# csat_app/predict_risk.py
import sys
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csat_project.settings')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from csat_app.models import Incident

print("Training predictive model on all incident data...")

# Load data
incidents = Incident.objects.all().values('weather', 'equipment_involved', 'severity')
df = pd.DataFrame(list(incidents))

if len(df) == 0:
    print("No data — skipping training.")
    sys.exit()

# Clean & encode
df = df.fillna('Unknown')
le_weather = LabelEncoder()
le_equipment = LabelEncoder()
le_severity = LabelEncoder()

df['weather_enc'] = le_weather.fit_transform(df['weather'])
df['equip_enc'] = le_equipment.fit_transform(df['equipment_involved'])
df['severity_enc'] = le_severity.fit_transform(df['severity'])

X = df[['weather_enc', 'equip_enc']]
y = df['severity_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained! Accuracy: {accuracy:.3%}")

# Save everything
joblib.dump(model, 'risk_model.joblib')
joblib.dump(le_weather, 'le_weather.joblib')
joblib.dump(le_equipment, 'le_equipment.joblib')
joblib.dump(le_severity, 'le_severity.joblib')

print("Model and encoders saved. Ready for predictions!")