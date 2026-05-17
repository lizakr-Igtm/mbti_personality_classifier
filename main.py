import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import os

os.makedirs('../models', exist_ok=True)
os.makedirs('../plots', exist_ok=True)

print("="*50)
print("MBTI Personality Classifier")
print("="*50)

print("\n1. Loading data...")
df = pd.read_csv('../data/mbti_1.csv')
print(f"   Loaded {len(df)} rows")

print("\n2. Cleaning text...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['full_text'] = df['posts'].str.replace('|||', ' ')
df['full_text'] = df['full_text'].apply(clean_text)
df['word_count'] = df['full_text'].str.split().str.len()
df = df[df['word_count'] >= 50]
print(f"   After filtering: {len(df)} rows")

print("\n3. Converting MBTI to binary...")
def mbti_to_binary(mbti):
    return {
        'IE': 1 if mbti[0] == 'I' else 0,
        'NS': 1 if mbti[1] == 'N' else 0,
        'TF': 1 if mbti[2] == 'T' else 0,
        'JP': 1 if mbti[3] == 'J' else 0
    }

binary = df['type'].apply(mbti_to_binary).apply(pd.Series)
df = pd.concat([df, binary], axis=1)

print("\n4. Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['full_text'])
y = df[['IE', 'NS', 'TF', 'JP']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

print("\n5. Training Logistic Regression...")
lr_models = {}
for col in ['IE', 'NS', 'TF', 'JP']:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train[col])
    lr_models[col] = model
    pred = model.predict(X_test)
    f1 = f1_score(y_test[col], pred)
    print(f"   {col}: F1 = {f1:.3f}")

print("\n6. Training Random Forest...")
rf_models = {}
for col in ['IE', 'NS', 'TF', 'JP']:
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train[col])
    rf_models[col] = model
    pred = model.predict(X_test)
    f1 = f1_score(y_test[col], pred)
    print(f"   {col}: F1 = {f1:.3f}")

print("\n7. Saving models...")
joblib.dump(rf_models, '../models/rf_models.pkl')
joblib.dump(lr_models, '../models/logreg_models.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
print("   Models saved to models/")

print("\n8. Creating plot...")
plt.figure(figsize=(10, 5))
df['type'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top-10 MBTI Types in Dataset')
plt.xlabel('MBTI Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../plots/mbti_distribution.png')
print("   Plot saved to plots/")

print("\n" + "="*50)
print("DONE. Models ready.")
print("="*50)