import sys
import os
import django

# Set up Django environment (run from root)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csat_project.settings')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add root to path if needed
django.setup()

import pandas as pd
from sqlalchemy import create_engine
from csat_app.models import Incident, Location, IncidentType #,Employee,  IncidentEmployee
from faker import Faker
from datetime import datetime
import random
from django.utils import timezone
from django.db import transaction
import math
from tqdm import tqdm
import numpy as np

import urllib.request
import pathlib

# Auto-download the OSHA dataset if it's missing
DATA_FILE = pathlib.Path('/app/osha_injury_2016_2021.csv')
if not DATA_FILE.exists():
    print(f"OSHA dataset found ({DATA_FILE.stat().st_size / 1e6:.1f} MB)")
else:
    print("OSHA dataset not found — downloading (~180 MB)...")
    url = "https://github.com/WTAMU-CIDM6395/James-Kinter/releases/download/v1.0/osha_injury_2016_2021.csv"
    urllib.request.urlretrieve(url, DATA_FILE)
    print("Download complete!")
    
# Clear all existing records from the database
def clear_database():
    #IncidentEmployee.objects.all().delete()
    Incident.objects.all().delete()
    IncidentType.objects.all().delete()
    #Employee.objects.all().delete()
    Location.objects.all().delete()
    print("Cleared all existing records from the database.")

# Create SQLAlchemy engine
engine = create_engine('mysql+pymysql://root:P1nkbunn13s@db:3306/csat_db')

# Make datetime timezone-aware, handling NaT
def make_aware(dt):
    if pd.isna(dt):
        return timezone.now()  # Default to current time for NaT
    if not timezone.is_aware(dt):
        return timezone.make_aware(dt)
    return dt

clear_database()

# Load osha_injury_2016_2021.csv (File 1) directly to Incident table
file_path = '/app/osha_injury_2016_2021.csv'
df_injury = pd.read_csv(file_path, low_memory=False)
print(f"Processing osha_injury_2016_2021.csv - Total rows: {len(df_injury)}")

# Map columns to Incident model fields (dummy for FKs, handle NaN)
df_injury['date'] = pd.to_datetime(df_injury['created_timestamp'], errors='coerce').apply(make_aware)
df_injury['description'] = "Total injuries: " + df_injury['total_injuries'].fillna(0).astype(str) + ", Deaths: " + df_injury['total_deaths'].fillna(0).astype(str) + ", Illnesses: " + df_injury['total_other_illnesses'].fillna(0).astype(str)
df_injury['severity'] = np.where(df_injury['total_deaths'].fillna(0) > 0, 'High', np.where(df_injury['total_injuries'].fillna(0) > 0, 'Medium', 'Low'))
df_injury['cause'] = "DAFW cases: " + df_injury['total_dafw_cases'].fillna(0).astype(str) + ", DJTR cases: " + df_injury['total_djtr_cases'].fillna(0).astype(str)
df_injury['outcome'] = "DAFW days: " + df_injury['total_dafw_days'].fillna(0).astype(str) + ", DJTR days: " + df_injury['total_djtr_days'].fillna(0).astype(str)
df_injury['location_id'] = None
df_injury['type_id'] = None

# Select and rename columns to match Incident model (add other fields as null if needed)
df_injury_incident = df_injury[['date', 'description', 'severity', 'cause', 'outcome', 'location_id', 'type_id']]

# Load in batches with progress (chunksize for to_sql)
batch_size = 10000  # Larger for speed
total_batches = math.ceil(len(df_injury_incident) / batch_size)
for batch_num in tqdm(range(total_batches), desc="Processing batches"):
    start = batch_num * batch_size
    end = start + batch_size
    batch_df = df_injury_incident.iloc[start:end]
    batch_df.to_sql('csat_app_incident', con=engine, if_exists='append', index=False)

print("Loaded osha_injury_2016_2021 data.")

# Load osha_sir.csv (File 2) directly to Incident table
file_path = '/app/osha_sir.csv'
df_sir = pd.read_csv(file_path, low_memory=False)
print(f"Processing osha_sir.csv - Total rows: {len(df_sir)}")

# Map columns to Incident model fields
df_sir['date'] = pd.to_datetime(df_sir['EventDate'], format='%m/%d/%Y', errors='coerce').apply(make_aware)
df_sir['description'] = df_sir['Final Narrative'].fillna('')
df_sir['severity'] = np.where((df_sir['Hospitalized'].fillna(0) > 0) | (df_sir['Amputation'].fillna(0) > 0) | (df_sir['Loss of Eye'].fillna(0) > 0), 'High', 'Medium')
df_sir['equipment_involved'] = df_sir['Secondary Source Title'].fillna(df_sir['SourceTitle'].fillna('None'))
df_sir['cause'] = df_sir['EventTitle'].fillna('')
df_sir['outcome'] = df_sir['Part of Body Title'].fillna('')
df_sir['location_id'] = None
df_sir['type_id'] = None

df_sir_incident = df_sir[['date', 'description', 'severity', 'equipment_involved', 'cause', 'outcome', 'location_id', 'type_id']]

# Load in batches with progress
batch_size = 10000
total_batches = math.ceil(len(df_sir_incident) / batch_size)
for batch_num in tqdm(range(total_batches), desc="Processing batches"):
    start = batch_num * batch_size
    end = start + batch_size
    batch_df = df_sir_incident.iloc[start:end]
    batch_df.to_sql('csat_app_incident', con=engine, if_exists='append', index=False)

print("Loaded osha_sir data.")

# Load osha_abstracts_2015_2017.csv (File 3) directly to Incident table
file_path = '/app/osha_abstracts_2015_2017.csv'
df_abstracts = pd.read_csv(file_path, low_memory=False)
print(f"Processing osha_abstracts_2015_2017.csv - Total rows: {len(df_abstracts)}")

# Map columns to Incident model fields
df_abstracts['date'] = pd.to_datetime(df_abstracts['Event Date'], format='%m/%d/%Y', errors='coerce').apply(make_aware)
df_abstracts['description'] = df_abstracts['Abstract Text'].fillna('')
df_abstracts['severity'] = df_abstracts['Degree of Injury'].fillna('Medium')
df_abstracts['equipment_involved'] = df_abstracts['hazsub'].fillna('None')
df_abstracts['cause'] = df_abstracts['evn_factor'].fillna('')
df_abstracts['outcome'] = df_abstracts['Part of Body'].fillna('')
df_abstracts['location_id'] = None
df_abstracts['type_id'] = None

df_abstracts_incident = df_abstracts[['date', 'description', 'severity', 'equipment_involved', 'cause', 'outcome', 'location_id', 'type_id']]

# Load in batches with progress
batch_size = 10000
total_batches = math.ceil(len(df_abstracts_incident) / batch_size)
for batch_num in tqdm(range(total_batches), desc="Processing batches"):
    start = batch_num * batch_size
    end = start + batch_size
    batch_df = df_abstracts_incident.iloc[start:end]
    batch_df.to_sql('csat_app_incident', con=engine, if_exists='append', index=False)

print("Loaded osha_abstracts_2015_2017 data.")

# Generate simulated data (50000 records representing industrial construction incidents)
fake = Faker()
incident_types_list = ['Falls', 'Hand Injury', 'Rigging Failure', 'Equipment Failure', 'Weld Failure', 'Confined Space Hazard', 'Chemical Exposure', 'Arc Flash', 'Missing Barricades', 'LOTO Failure', 'Spotting Failure', 'Human-Machine Interface Conflict']
total_simulated = 50000
print(f"Generating simulated data - Total records: {total_simulated}")

for i in tqdm(range(total_simulated), desc="Generating records"):
    location = Location.objects.create(name=fake.city(), address=fake.address(), site_manager=fake.name())

    incident_type_name = random.choice(incident_types_list)
    incident_type = IncidentType.objects.create(
        name=incident_type_name,
        description=fake.sentence(nb_words=10) + f" related to {incident_type_name.lower()} in industrial construction."
    )

    incident = Incident.objects.create(
        date=make_aware(fake.date_time_this_decade()),
        time=fake.time(),
        description=fake.paragraph(nb_sentences=3) + f" This incident involved {incident_type_name.lower()} typical in industrial construction sites.",
        severity=random.choice(['Low', 'Medium', 'High']),
        weather=random.choice(['Clear', 'Rainy', 'Windy', 'Foggy', 'Hot', 'Cold']),
        equipment_involved=random.choice(['Crane', 'Welder', 'Rigging Gear', 'Scaffold', 'Tank/Vessel', 'Electrical Panel', 'Barricade', 'Lockout Device', 'Vehicle', 'Dozer', 'Grader', 'Dump Truck', 'Excavator', 'H2S Detector', 'Spotter Equipment']),
        cause=fake.sentence(nb_words=8) + f" leading to {incident_type_name.lower()}.",
        outcome=fake.sentence(nb_words=6) + f" with outcome from {incident_type_name.lower()}.",
        location=location,
        type=incident_type
    )

    #employee = Employee.objects.create(name=fake.name(), role=random.choice(['Welder', 'Rigger', 'Operator', 'Electrician', 'Laborer', 'Supervisor', 'Spotter', 'LOTO Technician']), experience_years=random.randint(1, 30))

    #IncidentEmployee.objects.create(incident=incident, employee=employee)

print("Loaded simulated data.")