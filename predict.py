import joblib
import re
import sys

rf_models = joblib.load('D:/учёба/mbti_personality_project/models/rf_models.pkl')
vectorizer = joblib.load('D:/учёба/mbti_personality_project/models/vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
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
        text = ' '.join(sys.argv[1:])
        print(predict_mbti(text))
    else:
        print("Использование: python predict.py 'текст'")