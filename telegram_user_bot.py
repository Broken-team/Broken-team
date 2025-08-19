import os
import sys
import subprocess
import asyncio
import json
import logging
import requests
import base64
import re
import time
import random
import html
from datetime import datetime, timedelta
from telethon import TelegramClient, events, errors as telethon_errors
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError
import aiohttp
from aiohttp import ClientError, ClientTimeout

# Configuration
CONFIG_FILE = "team_broken_ai_config.json"
LOG_FILE = "team_broken_ai.log"
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 15
MAX_CONTEXT_LENGTH = 10
MAX_MESSAGE_HISTORY = 2000

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class APIError(Exception):
    pass

# Utility Functions
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def install_dependencies():
    """Install required packages"""
    required_packages = [
        "telethon",
        "python-telegram-bot",
        "requests",
        "aiohttp"
    ]
    
    for package in required_packages:
        try:
            __import__(package.split("==")[0])
            logger.info(f"✅ {package} already installed")
        except ImportError:
            logger.info(f"📥 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError:
                logger.error(f"❌ Failed to install {package}")

# Configuration Management
def validate_config(config):
    """Validate configuration parameters"""
    required_fields = ["api_id", "api_hash", "gemini_keys"]
    for field in required_fields:
        if not config.get(field):
            raise ConfigError(f"Missing required field: {field}")
    
    if config.get("api_id") and not str(config["api_id"]).isdigit():
        raise ConfigError("Invalid Telegram API ID")
    if config.get("api_hash") and len(config["api_hash"]) != 32:
        raise ConfigError("Invalid Telegram API Hash")
    return True

def load_config():
    """Load configuration from file or environment variables"""
    # First try environment variables (for Railway)
    env_config = load_config_from_env()
    if env_config:
        return env_config
    
    # Then try config file
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                validate_config(config)
                return config
    except (json.JSONDecodeError, ConfigError, IOError) as e:
        logger.error(f"Config load error: {str(e)}")
    
    return None

def load_config_from_env():
    """Load configuration from environment variables"""
    config = {}
    
    if os.environ.get('API_ID'):
        config = {
            "api_id": os.environ.get('API_ID'),
            "api_hash": os.environ.get('API_HASH'),
            "gemini_keys": os.environ.get('GEMINI_KEYS', '').split(','),
            "deepseek_keys": os.environ.get('DEEPSEEK_KEYS', '').split(','),
            "bot_token": os.environ.get('BOT_TOKEN', ''),
            "admin_ids": [int(x) for x in os.environ.get('ADMIN_IDS', '').split(',') if x],
            "preferred_api": os.environ.get('PREFERRED_API', 'gemini'),
            "enable_image_caption": os.environ.get('ENABLE_IMAGE_CAPTION', 'false').lower() == 'true'
        }
        
        # Validate
        try:
            validate_config(config)
            logger.info("✅ Configuration loaded from environment variables")
            return config
        except ConfigError:
            return None
    
    return None

def save_config(config):
    """Save configuration to file"""
    try:
        validate_config(config)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("Configuration saved successfully")
        return True
    except (IOError, ConfigError) as e:
        logger.error(f"Failed to save config: {str(e)}")
        return False

# Free API Services
class FreeAPIServices:
    @staticmethod
    async def translate_text(text, target_lang="en"):
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {"q": text, "langpair": f"auto|{target_lang}"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("responseData", {}).get("translatedText", text)
            return text
        except:
            return text

    @staticmethod
    async def get_joke():
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the chicken cross the road? To get to the other side!",
            "What do you call a fake noodle? An impasta!",
        ]
        return random.choice(jokes)

    @staticmethod
    async def get_quote():
        quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Life is what happens when you're busy making other plans. - John Lennon",
        ]
        return random.choice(quotes)

# API Manager
class EnhancedAPIManager:
    def __init__(self, config):
        self.config = config
        self.api_stats = {
            "gemini": {"success": 0, "failures": 0, "last_used": None},
            "deepseek": {"success": 0, "failures": 0, "last_used": None}
        }
        self.key_index = {"gemini": 0, "deepseek": 0}

    def get_next_key(self, api_type):
        keys = self.config.get(f"{api_type}_keys", [])
        if not keys:
            raise APIError(f"No {api_type} keys available")
        key = keys[self.key_index[api_type]]
        self.key_index[api_type] = (self.key_index[api_type] + 1) % len(keys)
        return key

    async def query_api(self, prompt: str, api_type: str):
        for _ in range(len(self.config[f"{api_type}_keys"])):
            key = self.get_next_key(api_type)
            try:
                if api_type == "gemini":
                    result = await self.query_gemini(prompt, key)
                else:
                    result = await self.query_deepseek(prompt, key)
                
                self.api_stats[api_type]["success"] += 1
                self.api_stats[api_type]["last_used"] = datetime.now().isoformat()
                return result, api_type
                
            except APIError as e:
                logger.warning(f"{api_type} key failed: {str(e)}")
                self.api_stats[api_type]["failures"] += 1
                continue
        
        raise APIError(f"All {api_type} keys exhausted")

    async def query_gemini(self, prompt: str, api_key: str):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        params = {"key": api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
        }
        
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as session:
            async with session.post(url, json=payload, params=params) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                    raise APIError(f"Gemini API error: {error_msg}")
                
                data = await response.json()
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
                raise APIError("No response from Gemini")

    async def query_deepseek(self, prompt: str, api_key: str):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                    raise APIError(f"DeepSeek API error: {error_msg}")
                
                data = await response.json()
                if 'choices' in data and data['choices']:
                    return data['choices'][0]['message']['content']
                raise APIError("No response from DeepSeek")

# Conversation Manager
class EnhancedConversationManager:
    def __init__(self):
        self.conversations = {}
        self.message_history = []

    def get_context(self, user_id):
        return self.conversations.get(user_id, [])

    def add_message(self, user_id, role, content):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversations[user_id]) > MAX_CONTEXT_LENGTH:
            self.conversations[user_id] = self.conversations[user_id][-MAX_CONTEXT_LENGTH:]

# Main Bot Class
class UltimateAITelegramBot:
    def __init__(self, config):
        self.config = config
        self.api_manager = EnhancedAPIManager(config)
        self.conversation_manager = EnhancedConversationManager()
        self.free_services = FreeAPIServices()
        
        self.user_status = {}
        self.client = None
        self.bot_app = None
        self.start_time = datetime.now()
        self.daily_stats = {"messages_processed": 0, "users_active": 0, "api_calls": 0}

    async def setup(self):
        try:
            # For Railway, use bot token only (no phone number)
            if self.config.get("bot_token"):
                print("🚀 Setting up Bot client...")
                self.bot_app = await self.setup_bot_app()
                self.register_handlers()
                logger.info("Bot setup completed!")
            else:
                raise ConfigError("Bot token required for Railway deployment")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

    async def setup_bot_app(self):
        for attempt in range(MAX_RETRIES):
            try:
                application = Application.builder().token(self.config["bot_token"]).build()
                await application.initialize()
                await application.start()
                print("✅ Bot client started successfully!")
                return application
            except TelegramError as e:
                logger.error(f"Bot setup failed (attempt {attempt + 1}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        raise ConnectionError("Failed to establish bot connection")

    def register_handlers(self):
        if self.bot_app:
            async def bot_handler(update, context):
                await self.handle_message(
                    user_id=update.effective_user.id,
                    message=update.message.text,
                    reply_func=update.message.reply_text,
                    is_admin=update.effective_user.id in self.config.get("admin_ids", []),
                    message_obj=update.message
                )
            
            self.bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handler))

    async def handle_message(self, user_id, message, reply_func, is_admin=False, message_obj=None):
        try:
            self.daily_stats["messages_processed"] += 1
            if user_id not in self.user_status:
                self.daily_stats["users_active"] += 1
                self.user_status[user_id] = True

            if await self.handle_commands(user_id, message, reply_func, is_admin):
                return

            if await self.handle_special_features(user_id, message, reply_func):
                return

            if self.user_status.get(user_id, True):
                await self.process_enhanced_ai_response(user_id, message, reply_func, message_obj)

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await reply_func("❌ An error occurred while processing your message.")

    async def handle_commands(self, user_id, message, reply_func, is_admin):
        command = message.lower().strip()
        
        if command == '/start':
            self.user_status[user_id] = True
            self.conversation_manager.clear_context(user_id)
            await reply_func("✅ AI activated! Send /help for available commands.")
            return True
        elif command == '/stop':
            self.user_status[user_id] = False
            await reply_func("🚫 AI responses paused. Send /start to enable again.")
            return True
        elif command == '/help':
            help_text = self.get_help_text(is_admin)
            await reply_func(help_text)
            return True
        elif command == '/status':
            status = self.get_system_status()
            await reply_func(status)
            return True
        elif command == '/context':
            self.conversation_manager.clear_context(user_id)
            await reply_func("🧹 Conversation context cleared!")
            return True
        return False

    async def handle_special_features(self, user_id, message, reply_func):
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['joke', 'funny']):
            joke = await self.free_services.get_joke()
            await reply_func(f"😄 {joke}")
            return True
        elif any(word in message_lower for word in ['quote', 'inspiration']):
            quote = await self.free_services.get_quote()
            await reply_func(f"💫 {quote}")
            return True
        return False

    async def process_enhanced_ai_response(self, user_id, message, reply_func, message_obj):
        try:
            self.conversation_manager.add_message(user_id, "user", message)
            
            context = self.conversation_manager.get_context(user_id)
            context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-5:]])
            
            enhanced_prompt = f"""Conversation context:
{context_text}

Current message: {message}

Please provide a helpful response:"""
            
            start_time = time.time()
            response, api_used, response_time = await self.api_manager.query_api(
                enhanced_prompt, self.config.get("preferred_api", "gemini")
            )
            
            self.daily_stats["api_calls"] += 1
            self.conversation_manager.add_message(user_id, "assistant", response)
            
            formatted_response = await self.format_response(response)
            await reply_func(formatted_response)
            
        except APIError as e:
            await reply_func(f"❌ API Error: {str(e)}")
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}")
            await reply_func("❌ Failed to generate response")

    async def format_response(self, response):
        if len(response) > 4000:
            response = response[:4000] + "...\n\n(Response truncated)"
        return response

    def get_help_text(self, is_admin=False):
        help_text = """
🤖 *AI Bot Commands:*
/start - Activate AI responses
/stop - Pause AI responses
/help - Show this help
/status - Show bot status
/context - Clear conversation history

*Special Features:*
• Request jokes: "tell me a joke"
• Get quotes: "inspirational quote"
"""
        if is_admin:
            help_text += """
👑 *Admin Commands:*
/stats - Detailed statistics
"""
        return help_text

    def get_system_status(self):
        return f"""
⚡ *System Status:*
• Active Users: {len(self.user_status)}
• Total Messages: {self.daily_stats['messages_processed']}
• API Calls: {self.daily_stats['api_calls']}
• Uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours
• Default API: {self.config.get('preferred_api', 'gemini').upper()}
"""

    async def run(self):
        try:
            print("\n" + "="*50)
            print("🤖 AI Telegram Bot is now running on Railway!")
            print("📍 Press Ctrl+C to stop the bot")
            print("="*50 + "\n")
            
            while True:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Bot runtime error: {str(e)}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("Shutting down clients...")
        try:
            if self.bot_app:
                await self.bot_app.stop()
                await self.bot_app.shutdown()
            logger.info("Clients shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

# Main function
async def main():
    try:
        clear_screen()
        print("🤖" + "="*50)
        print("          ULTIMATE AI TELEGRAM BOT")
        print("              Railway Optimized")
        print("="*50 + "🤖")
        
        # Install dependencies
        install_dependencies()
        
        # Load configuration
        config = load_config()
        if not config:
            print("❌ No valid configuration found!")
            print("Please set environment variables or create config.json")
            return
        
        # Initialize and run bot
        bot = UltimateAITelegramBot(config)
        await bot.setup()
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        print(f"\n❌ Critical error: {str(e)}")
    finally:
        print("\n👋 Session ended")

if __name__ == "__main__":
    asyncio.run(main())