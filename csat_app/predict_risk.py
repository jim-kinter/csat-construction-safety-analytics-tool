# csat_app/predict_risk.py
import sys
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csat_project.settings')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from csat_app.models import Incident

print("Training predictive model on all incident data...")

# Load data
incidents = Incident.objects.all().values('weather', 'equipment_involved', 'severity', 'date')
df = pd.DataFrame(list(incidents))

if len(df) == 0:
    print("No data — skipping training.")
    sys.exit()

# Feature engineering
df = df.dropna(subset=['weather', 'equipment_involved', 'severity', 'date'])
df['month'] = pd.to_datetime(df['date']).dt.month
df['state'] = 'Unknown'  # placeholder — if you have state in data, use it

# Encode categorical features
le_weather = LabelEncoder()
le_equipment = LabelEncoder()
le_severity = LabelEncoder()

df['weather_enc'] = le_weather.fit_transform(df['weather'])
df['equip_enc'] = le_equipment.fit_transform(df['equipment_involved'])
df['month_enc'] = df['month']  # month is numeric
df['severity_enc'] = le_severity.fit_transform(df['severity'])

# Features
X = df[['weather_enc', 'equip_enc', 'month_enc']]
y = df['severity_enc']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights to handle imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained! Accuracy: {accuracy:.3%}")
print(classification_report(y_test, y_pred, target_names=le_severity.classes_))

# Save model and encoders
joblib.dump(model, 'risk_model.joblib')
joblib.dump(le_weather, 'le_weather.joblib')
joblib.dump(le_equipment, 'le_equipment.joblib')
joblib.dump(le_severity, 'le_severity.joblib')

print("Model and encoders saved. Ready for predictions!")