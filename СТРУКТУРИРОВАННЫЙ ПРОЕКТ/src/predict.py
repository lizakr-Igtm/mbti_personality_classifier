import os
import joblib
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models', 'rf_models.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

rf_models = joblib.load(MODELS_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_mbti(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    
    ie = rf_models['IE'].predict(vec)[0]
    ns = rf_models['NS'].predict(vec)[0]
    tf = rf_models['TF'].predict(vec)[0]
    jp = rf_models['JP'].predict(vec)[0]
    
    mbti = ''
    mbti += 'I' if ie == 1 else 'E'
    mbti += 'N' if ns == 1 else 'S'
    mbti += 'T' if tf == 1 else 'F'
    mbti += 'J' if jp == 1 else 'P'
    return mbti

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = ' '.join(sys.argv[1:])
        print(predict_mbti(input_text))
    else:
        print("Использование: python predict.py 'ваш текст'")
