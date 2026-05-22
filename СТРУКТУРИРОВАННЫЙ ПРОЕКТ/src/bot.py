import os
import re
import joblib
import telebot
from telebot import apihelper
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
PROXY_URL = os.getenv("PROXY_URL")

if PROXY_URL:
    apihelper.proxy = {'http': PROXY_URL, 'https': PROXY_URL}

if not TOKEN:
    raise ValueError("Ошибка: Токен бота не найден! Проверьте файл .env в корне проекта.")
bot = telebot.TeleBot(TOKEN)

print("Загружаю модели ИИ...")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(CUR_DIR, 'models', 'rf_models.pkl')
VECTORIZER_PATH = os.path.join(CUR_DIR, 'models', 'vectorizer.pkl')

if not os.path.exists(MODELS_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Файлы моделей не найдены в папке {os.path.join(CUR_DIR, 'models')}. Запустите сначала create_models.py.")

rf_models = joblib.load(MODELS_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Бот успешно запущен и готов к работе!")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "Привет! Отправь текст от 50 слов, чтобы я определил твой тип личности.")

@bot.message_handler(func=lambda m: True)
def handle(m):
    if len(m.text.split()) < 50:
        bot.reply_to(m, f"В вашем тексте сейчас {len(m.text.split())} слов. Нужно отправить текст длиной 50+ слов.")
        return
        
    text = clean_text(m.text)
    vec = vectorizer.transform([text])
    
    ie = rf_models['IE'].predict(vec)
    ns = rf_models['NS'].predict(vec)
    tf = rf_models['TF'].predict(vec)
    jp = rf_models['JP'].predict(vec)
    
    res = ''
    res += 'I' if ie == 1 else 'E'
    res += 'N' if ns == 1 else 'S'
    res += 'T' if tf == 1 else 'F'
    res += 'J' if jp == 1 else 'P'
    
    bot.reply_to(m, f"Твой тип личности по тексту: {res}")

bot.infinity_polling()