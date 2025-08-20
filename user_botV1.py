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
    """Install required packages for Termux"""
    clear_screen()
    print("📦 Installing dependencies...")
    
    required_packages = [
        "telethon",
        "python-telegram-bot",
        "requests",
        "aiohttp"
    ]
    
    for package in required_packages:
        try:
            __import__(package.split("==")[0])
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    
    print("\n✅ All dependencies installed!")
    time.sleep(2)
    clear_screen()

# Configuration Management
def validate_config(config):
    """Validate configuration parameters"""
    if not config.get("api_id") or not str(config["api_id"]).isdigit():
        raise ConfigError("Invalid Telegram API ID")
    if not config.get("api_hash") or len(config["api_hash"]) != 32:
        raise ConfigError("Invalid Telegram API Hash")
    if not config.get("gemini_keys") and not config.get("deepseek_keys"):
        raise ConfigError("At least one API key set is required")
    return True

def load_config():
    """Load configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                validate_config(config)
                return config
    except (json.JSONDecodeError, ConfigError, IOError) as e:
        logger.error(f"Config load error: {str(e)}")
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

# Free API Services for Termux
class FreeAPIServices:
    """Free API services that work on Termux"""
    
    @staticmethod
    async def translate_text(text, target_lang="en"):
        """Free translation using MyMemory API"""
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
    async def get_weather(city):
        """Free weather API"""
        try:
            # Using openweathermap API (need to get free API key)
            return f"Weather information for {city} is currently unavailable. Please configure weather API key."
        except:
            return "Could not fetch weather"

    @staticmethod
    async def get_joke():
        """Free joke API"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the chicken cross the road? To get to the other side!",
            "What do you call a fake noodle? An impasta!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a bear with no teeth? A gummy bear!"
        ]
        return random.choice(jokes)

    @staticmethod
    async def get_quote():
        """Free quote API"""
        quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Life is what happens when you're busy making other plans. - John Lennon",
            "Be the change that you wish to see in the world. - Mahatma Gandhi",
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt"
        ]
        return random.choice(quotes)

# Enhanced Image Processing without PIL
class AdvancedImageProcessor:
    """Advanced image processing using free APIs"""
    
    @staticmethod
    async def analyze_image(image_url):
        """Comprehensive image analysis"""
        try:
            return "Image analysis: This feature requires API configuration. Please set up image analysis API keys."
        except Exception as e:
            return f"Image analysis error: {str(e)}"

# Voice Message Processing
class VoiceProcessor:
    """Voice message processing using free APIs"""
    
    @staticmethod
    async def transcribe_voice(audio_url):
        """Transcribe voice message using free API"""
        try:
            return "Voice transcription: This feature requires speech-to-text API configuration."
        except Exception as e:
            return f"Transcription error: {str(e)}"

# Web Search Integration
class WebSearch:
    """Free web search functionality"""
    
    @staticmethod
    async def search_web(query):
        """Search web using free APIs"""
        try:
            return f"Web search for '{query}': This feature requires search API configuration."
        except Exception as e:
            return f"Search error: {str(e)}"

# API Manager with Enhanced Features
class EnhancedAPIManager:
    """Enhanced API manager with more features"""
    
    def __init__(self, config):
        self.config = config
        self.api_stats = {
            "gemini": {"success": 0, "failures": 0, "last_used": None},
            "deepseek": {"success": 0, "failures": 0, "last_used": None}
        }
        self.key_index = {"gemini": 0, "deepseek": 0}
        self.last_fail_time = {"gemini": 0, "deepseek": 0}
        self.response_times = []

    def get_next_key(self, api_type):
        """Get next API key with round-robin"""
        keys = self.config.get(f"{api_type}_keys", [])
        if not keys:
            raise APIError(f"No {api_type} keys available")
        
        key = keys[self.key_index[api_type]]
        self.key_index[api_type] = (self.key_index[api_type] + 1) % len(keys)
        return key

    async def query_api(self, prompt: str, api_type: str):
        """Query API with automatic failover"""
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
                logger.warning(f"{api_type} key {key[:5]}... failed: {str(e)}")
                self.api_stats[api_type]["failures"] += 1
                continue
        
        raise APIError(f"All {api_type} keys exhausted")

    async def query_gemini(self, prompt: str, api_key: str):
        """Query Gemini API"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        params = {"key": api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
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
        """Query DeepSeek API"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
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

    async def query_with_enhancements(self, prompt: str, user_id: int):
        """Enhanced query with additional features"""
        try:
            # Add context awareness
            enhanced_prompt = await self._enhance_prompt(prompt, user_id)
            
            # Get response
            start_time = time.time()
            response, api_used = await self.query_api(enhanced_prompt, self.config.get("preferred_api", "gemini"))
            response_time = time.time() - start_time
            
            # Store performance metrics
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            # Add emotional intelligence
            response = await self._add_emotional_touch(response)
            
            return response, api_used, response_time
            
        except Exception as e:
            raise APIError(f"Enhanced query failed: {str(e)}")

    async def _enhance_prompt(self, prompt: str, user_id: int):
        """Enhance prompt with context and intelligence"""
        enhanced = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        enhanced += "Please provide helpful, accurate, and friendly response.\n"
        enhanced += f"User query: {prompt}"
        return enhanced

    async def _add_emotional_touch(self, response: str):
        """Add emotional intelligence to responses"""
        positive_words = ['great', 'awesome', 'wonderful', 'amazing', 'happy', 'excellent']
        negative_words = ['sad', 'angry', 'frustrated', 'problem', 'issue', 'sorry']
        
        if any(word in response.lower() for word in positive_words):
            response = f"😊 {response}"
        elif any(word in response.lower() for word in negative_words):
            response = f"🤗 {response}"
        
        return response

# Enhanced Conversation Manager
class EnhancedConversationManager:
    """Enhanced conversation management with more features"""
    
    def __init__(self):
        self.conversations = {}
        self.user_preferences = {}
        self.message_history = []
        self.user_metrics = {}

    def get_context(self, user_id):
        """Get conversation context for user"""
        return self.conversations.get(user_id, [])

    def add_message(self, user_id, role, content):
        """Add message to conversation context"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim context to max length
        if len(self.conversations[user_id]) > MAX_CONTEXT_LENGTH:
            self.conversations[user_id] = self.conversations[user_id][-MAX_CONTEXT_LENGTH:]
        
        # Store in global history
        self.message_history.append({
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim global history
        if len(self.message_history) > MAX_MESSAGE_HISTORY:
            self.message_history = self.message_history[-MAX_MESSAGE_HISTORY:]

    def clear_context(self, user_id):
        """Clear conversation context for user"""
        if user_id in self.conversations:
            del self.conversations[user_id]

    def track_conversation_quality(self, user_id, message, response, response_time):
        """Track conversation quality metrics"""
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = {
                "total_messages": 0,
                "avg_response_time": 0,
                "engagement_score": 0
            }
        
        metrics = self.user_metrics[user_id]
        metrics["total_messages"] += 1
        metrics["avg_response_time"] = (
            (metrics["avg_response_time"] * (metrics["total_messages"] - 1) + response_time) 
            / metrics["total_messages"]
        )

# Enhanced Main Bot Class with All Features
class UltimateAITelegramBot:
    """Ultimate AI Telegram Bot with all features"""
    
    def __init__(self, config):
        self.config = config
        self.api_manager = EnhancedAPIManager(config)
        self.conversation_manager = EnhancedConversationManager()
        self.free_services = FreeAPIServices()
        self.image_processor = AdvancedImageProcessor()
        self.voice_processor = VoiceProcessor()
        self.web_search = WebSearch()
        
        self.user_status = {}
        self.client = None
        self.bot_app = None
        self.start_time = datetime.now()
        self.daily_stats = {
            "messages_processed": 0,
            "users_active": 0,
            "api_calls": 0
        }

    async def setup(self):
        """Setup Telegram clients"""
        try:
            phone_number = input("\n📱 Enter your phone number (with country code): ").strip()
            
            print("🚀 Setting up Telegram clients...")
            self.client = await self.setup_telegram_client(phone_number)
            self.bot_app = await self.setup_bot_app()
            
            self.register_handlers()
            logger.info("AI Telegram Bot setup completed!")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

    async def setup_telegram_client(self, phone_number):
        """Setup Telethon client"""
        for attempt in range(MAX_RETRIES):
            try:
                client = TelegramClient(
                    'team_broken_ai_session',
                    int(self.config["api_id"]),
                    self.config["api_hash"]
                )
                await client.start(phone=phone_number)
                print("✅ Userbot client started successfully!")
                return client
            except telethon_errors.PhoneNumberInvalidError:
                raise ConfigError("❌ Invalid phone number format")
            except telethon_errors.ApiIdInvalidError:
                raise ConfigError("❌ Invalid API ID or Hash")
            except Exception as e:
                logger.error(f"Telegram client setup failed (attempt {attempt + 1}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
        raise ConnectionError("Failed to establish Telegram connection")

    async def setup_bot_app(self):
        """Setup python-telegram-bot application"""
        if not self.config.get("bot_token"):
            print("ℹ️  Bot token not set, running in userbot mode only")
            return None
        
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
        """Register message handlers"""
        
        # Userbot handler
        @self.client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private))
        async def userbot_handler(event):
            await self.handle_message(
                user_id=event.sender_id,
                message=event.text,
                reply_func=event.reply,
                is_admin=event.sender_id in self.config.get("admin_ids", []),
                message_obj=event
            )

        # Bot handler
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
        """Enhanced message handling with all features"""
        try:
            # Update daily stats
            self.daily_stats["messages_processed"] += 1
            if user_id not in self.user_status or not self.user_status.get(user_id, True):
                self.daily_stats["users_active"] += 1
                self.user_status[user_id] = True

            # Handle commands
            if await self.handle_commands(user_id, message, reply_func, is_admin):
                return

            # Handle special features
            if await self.handle_special_features(user_id, message, reply_func):
                return

            # Process AI response
            if self.user_status.get(user_id, True):
                await self.process_enhanced_ai_response(user_id, message, reply_func, message_obj)

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await reply_func("❌ An error occurred while processing your message.")

    async def handle_commands(self, user_id, message, reply_func, is_admin):
        """Handle bot commands"""
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

        elif command == '/apistats':
            stats = self.get_api_stats()
            await reply_func(stats)
            return True

        elif command == '/stats' and is_admin:
            stats = self.get_detailed_stats()
            await reply_func(stats)
            return True

        elif command.startswith('/broadcast ') and is_admin:
            await reply_func("📢 Broadcast feature would be implemented here")
            return True

        elif command.startswith('/setapi ') and is_admin:
            api_type = command[8:].strip().lower()
            if api_type in ['gemini', 'deepseek']:
                self.config['preferred_api'] = api_type
                save_config(self.config)
                await reply_func(f"✅ Default API set to {api_type.upper()}")
            else:
                await reply_func("❌ Invalid API type. Use 'gemini' or 'deepseek'")
            return True

        return False

    async def handle_special_features(self, user_id, message, reply_func):
        """Handle special features like weather, jokes, etc."""
        message_lower = message.lower()
        
        # Weather queries
        if any(word in message_lower for word in ['weather', 'temperature']):
            city_match = re.search(r'weather in (\w+)', message_lower)
            if city_match:
                weather = await self.free_services.get_weather(city_match.group(1))
                await reply_func(weather)
                return True

        # Joke requests
        if any(word in message_lower for word in ['joke', 'funny']):
            joke = await self.free_services.get_joke()
            await reply_func(f"😄 {joke}")
            return True

        # Quote requests
        if any(word in message_lower for word in ['quote', 'inspiration']):
            quote = await self.free_services.get_quote()
            await reply_func(f"💫 {quote}")
            return True

        # Translation requests
        if 'translate' in message_lower:
            await reply_func("🌍 Translation: Please configure translation API for this feature")
            return True

        # Web search
        if any(word in message_lower for word in ['search', 'google']):
            search_match = re.search(r'search (.+)', message_lower)
            if search_match:
                result = await self.web_search.search_web(search_match.group(1))
                await reply_func(f"🔍 {result}")
                return True

        return False

    async def process_enhanced_ai_response(self, user_id, message, reply_func, message_obj):
        """Enhanced AI response processing"""
        try:
            # Add to conversation context
            self.conversation_manager.add_message(user_id, "user", message)
            
            # Build enhanced prompt
            context = self.conversation_manager.get_context(user_id)
            context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-5:]])
            
            enhanced_prompt = f"""Conversation context:
{context_text}

Current message: {message}

Please provide a helpful response:"""
            
            # Get AI response
            start_time = time.time()
            response, api_used, response_time = await self.api_manager.query_with_enhancements(
                enhanced_prompt, user_id
            )
            
            # Track metrics
            self.conversation_manager.track_conversation_quality(user_id, message, response, response_time)
            self.daily_stats["api_calls"] += 1
            
            # Add to context and send response
            self.conversation_manager.add_message(user_id, "assistant", response)
            
            # Format response
            formatted_response = await self.format_response(response)
            await reply_func(formatted_response)
            
        except APIError as e:
            await reply_func(f"❌ API Error: {str(e)}")
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}")
            await reply_func("❌ Failed to generate response")

    async def format_response(self, response):
        """Format response with emojis and better presentation"""
        # Limit response length for Telegram
        if len(response) > 4000:
            response = response[:4000] + "...\n\n(Response truncated)"
        
        return response

    def get_help_text(self, is_admin=False):
        """Generate help text"""
        help_text = """
🤖 *AI Bot Commands:*
/start - Activate AI responses
/stop - Pause AI responses
/help - Show this help
/status - Show bot status
/apistats - Show API usage
/context - Clear conversation history

*Special Features:*
• Ask for weather: "weather in London"
• Request jokes: "tell me a joke"
• Get quotes: "inspirational quote"
• Web search: "search python programming"
"""
        if is_admin:
            help_text += """
👑 *Admin Commands:*
/stats - Detailed statistics
/broadcast - Send to all users
/setapi - Change default API
"""
        return help_text

    def get_system_status(self):
        """Get system status"""
        return f"""
⚡ *System Status:*
• Active Users: {len(self.user_status)}
• Total Messages: {self.daily_stats['messages_processed']}
• API Calls: {self.daily_stats['api_calls']}
• Uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours
• Default API: {self.config.get('preferred_api', 'gemini').upper()}
"""

    def get_api_stats(self):
        """Get API statistics"""
        stats = self.api_manager.api_stats
        response = "📊 *API Statistics:*\n"
        
        for api, data in stats.items():
            success = data.get('success', 0)
            failures = data.get('failures', 0)
            total = success + failures
            success_rate = (success / total * 100) if total > 0 else 0
            
            response += f"\n*{api.upper()}:*\n"
            response += f"✅ Success: {success}\n"
            response += f"❌ Failures: {failures}\n"
            response += f"📈 Rate: {success_rate:.1f}%\n"
        
        return response

    def get_detailed_stats(self):
        """Get detailed statistics for admin"""
        return f"""
📈 *Detailed Statistics:*
• Total Users: {len(self.user_status)}
• Active Today: {self.daily_stats['users_active']}
• Messages Today: {self.daily_stats['messages_processed']}
• API Calls Today: {self.daily_stats['api_calls']}

🔑 *API Keys:*
• Gemini Keys: {len(self.config.get('gemini_keys', []))}
• DeepSeek Keys: {len(self.config.get('deepseek_keys', []))}
"""

    async def run(self):
        """Main bot loop"""
        try:
            print("\n" + "="*50)
            print("🤖 AI Telegram Bot is now running!")
            print("📍 Press Ctrl+C to stop the bot")
            print("="*50 + "\n")
            
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Bot runtime error: {str(e)}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown bot gracefully"""
        logger.info("Shutting down clients...")
        try:
            if self.client:
                await self.client.disconnect()
            if self.bot_app:
                await self.bot_app.stop()
                await self.bot_app.shutdown()
            logger.info("Clients shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

# Configuration Menu
def config_menu():
    """Interactive configuration menu"""
    existing_config = load_config() or {
        "api_id": "",
        "api_hash": "",
        "gemini_keys": [],
        "deepseek_keys": [],
        "bot_token": "",
        "admin_ids": [],
        "preferred_api": "gemini",
        "enable_image_caption": False
    }
    
    while True:
        clear_screen()
        print("🎛️  Advanced Configuration Menu")
        print("=" * 40)
        print(f"1. API ID: {existing_config['api_id'] or 'Not set'}")
        print(f"2. API Hash: {existing_config['api_hash'] and 'Set' or 'Not set'}")
        print(f"3. Gemini Keys: {len(existing_config['gemini_keys'])} keys")
        print(f"4. DeepSeek Keys: {len(existing_config['deepseek_keys'])} keys")
        print(f"5. Bot Token: {'Set' if existing_config['bot_token'] else 'Not set'}")
        print(f"6. Admin IDs: {len(existing_config['admin_ids'])} admins")
        print(f"7. Preferred API: {existing_config.get('preferred_api', 'gemini').upper()}")
        print("8. Save and Run")
        print("9. Exit")
        print("=" * 40)
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            existing_config["api_id"] = input("Enter Telegram API ID: ").strip()
        elif choice == "2":
            existing_config["api_hash"] = input("Enter Telegram API Hash: ").strip()
        elif choice == "3":
            keys = input("Enter Gemini API keys (comma separated): ").strip()
            existing_config["gemini_keys"] = [k.strip() for k in keys.split(",") if k.strip()]
        elif choice == "4":
            keys = input("Enter DeepSeek API keys (comma separated): ").strip()
            existing_config["deepseek_keys"] = [k.strip() for k in keys.split(",") if k.strip()]
        elif choice == "5":
            existing_config["bot_token"] = input("Enter Bot Token: ").strip()
        elif choice == "6":
            ids = input("Enter Admin IDs (comma separated): ").strip()
            existing_config["admin_ids"] = [int(id.strip()) for id in ids.split(",") if id.strip().isdigit()]
        elif choice == "7":
            api = input("Preferred API (gemini/deepseek): ").strip().lower()
            if api in ["gemini", "deepseek"]:
                existing_config["preferred_api"] = api
        elif choice == "8":
            try:
                validate_config(existing_config)
                if save_config(existing_config):
                    print("✅ Configuration saved successfully!")
                    time.sleep(2)
                    return existing_config
            except ConfigError as e:
                input(f"❌ {str(e)}\nPress Enter to continue...")
        elif choice == "9":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            input("❌ Invalid option! Press Enter to continue...")

# Main function
async def main():
    """Main entry point"""
    try:
        clear_screen()
        print("🤖" + "="*50)
        print("          ULTIMATE AI TELEGRAM BOT")
        print("              Termux Compatible")
        print("="*50 + "🤖")
        print("\n🚀 Initializing system...")
        
        # Install dependencies
        install_dependencies()
        
        # Load or create configuration
        config = config_menu()
        
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