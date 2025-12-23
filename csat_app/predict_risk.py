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
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import random
from faker import Faker
from csat_app.models import Incident

print("Training predictive model on injury data only...")

# Load ONLY injury data for consistent severity labels
incidents = list(Incident.objects.filter(data_source='injury').values('weather', 'equipment_involved', 'severity', 'date'))

# Fallback if no injury data (should not happen)
if not incidents:
    print("No injury incidents found — using simulated data for training")
    fake = Faker()
    incidents = []
    for _ in range(10000):
        incidents.append({
            'weather': random.choice(['Clear', 'Rainy', 'Hot', 'Cold']),
            'equipment_involved': random.choice(['Crane', 'Scaffold', 'Welder', 'Ladder']),
            'severity': random.choice(['Low', 'Medium', 'High']),
            'date': fake.date_time_this_decade()
        })

df = pd.DataFrame(incidents)

if len(df) == 0:
    print("No data available — skipping training.")
    sys.exit()

# Feature engineering — safe .assign() (no FutureWarnings)
df = df.assign(
    month = pd.to_datetime(df['date']).dt.month,
    weather_enc = LabelEncoder().fit_transform(df['weather']),
    equip_enc = LabelEncoder().fit_transform(df['equipment_involved']),
    month_enc = pd.to_datetime(df['date']).dt.month,
    severity_enc = LabelEncoder().fit_transform(df['severity'])
)

# Save encoders
le_weather = LabelEncoder().fit(df['weather'])
le_equipment = LabelEncoder().fit(df['equipment_involved'])
le_severity = LabelEncoder().fit(df['severity'])

joblib.dump(le_weather, 'le_weather.joblib')
joblib.dump(le_equipment, 'le_equipment.joblib')
joblib.dump(le_severity, 'le_severity.joblib')

# Binary classification: Serious (High/Fatal) vs Non-Serious (Low/Medium)
# Binary classification: Serious (High) vs Non-Serious (Low/Medium)
df = df.assign(
    is_serious = (df['severity'] == 'High').astype(int)
)

X = df[['weather_enc', 'equip_enc', 'month_enc']]
y = df['is_serious']

# Update encoder for binary target (not needed anymore, but keep for consistency)
le_severity = LabelEncoder().fit(['Non-Serious', 'Serious'])
joblib.dump(le_severity, 'le_severity.joblib')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Class weights for imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# XGBoost
# Calculate ratio (non-serious / serious)
ratio = (len(y_train) - y_train.sum()) / y_train.sum()  # ~326

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=ratio  # ← this balances the classes
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Serious recall:", recall_score(y_test, y_pred))
print("Serious F1:", f1_score(y_test, y_pred))
print(f"Model trained! Accuracy: {accuracy:.3%}")
print(classification_report(y_test, y_pred, target_names=['Non-Serious', 'Serious'],zero_division=0))

# Save model
joblib.dump(model, 'risk_model.joblib')

print("Model and encoders saved. Ready for predictions!")