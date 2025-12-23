import sys
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csat_project.settings')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

import pandas as pd
from sqlalchemy import create_engine
from csat_app.models import Incident, Location, IncidentType
from faker import Faker
import random
from django.utils import timezone
from tqdm import tqdm
import numpy as np
import pathlib
import urllib.request
from django.db import connection

# Paths for all three datasets
INJURY_FILE = pathlib.Path('/app/osha_injury_2016_2021.csv')
SIR_FILE = pathlib.Path('/app/osha_sir.csv')
ABSTRACTS_FILE = pathlib.Path('/app/osha_abstracts_2015_2017.csv')

# Auto-download helper with browser user-agent
def download_file(url, dest_path):
    print(f"Downloading {dest_path.name} (~unknown MB)...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path)
    print("Download complete!")

# Download all three if missing
for file_path, url in [
    (INJURY_FILE, "https://github.com/jim-kinter/csat-construction-safety-analytics-tool/releases/download/v1.0/osha_injury_2016_2021.csv"),
    (SIR_FILE, "https://github.com/jim-kinter/csat-construction-safety-analytics-tool/releases/download/v1.0/osha_sir.csv"),
    (ABSTRACTS_FILE, "https://github.com/jim-kinter/csat-construction-safety-analytics-tool/releases/download/v1.0/osha_abstracts_2015_2017.csv")
]:
    if file_path.exists():
        print(f"{file_path.name} found ({file_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        download_file(url, file_path)

# Clear database
def clear_database():
    Incident.objects.all().delete()
    IncidentType.objects.all().delete()
    Location.objects.all().delete()
    print("Cleared existing records.")

clear_database()

# Robust CSV loader
def load_csv_safely(file_path, name):
    print(f"Loading {name}...")
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1', 'windows-1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=enc)
            print(f"Loaded {len(df):,} rows with {enc} encoding")
            return df
        except UnicodeDecodeError:
            continue
    print(f"Failed to load {name} with any encoding")
    sys.exit(1)

# Load all datasets
df_injury = load_csv_safely(INJURY_FILE, "OSHA Injury")
df_sir = load_csv_safely(SIR_FILE, "OSHA SIR")
df_abstracts = load_csv_safely(ABSTRACTS_FILE, "OSHA Abstracts")

# Reconnect helper
def reconnect_db():
    connection.close()
    connection.connect()

# Batch bulk_create helper
def batch_create(model_class, objects, batch_size=10000):
    for i in range(0, len(objects), batch_size):
        reconnect_db()
        batch = objects[i:i + batch_size]
        model_class.objects.bulk_create(batch)
        print(f"Inserted {len(batch)} records")

# Process Injury data
print("Processing injury data...")
df_injury['date'] = pd.to_datetime(df_injury.get('created_timestamp', pd.Series()), errors='coerce').apply(lambda dt: timezone.make_aware(dt) if pd.notna(dt) else timezone.now())
df_injury['description'] = "Injuries: " + df_injury.get('total_injuries', 0).fillna(0).astype(str) + " | Deaths: " + df_injury.get('total_deaths', 0).fillna(0).astype(str)
df_injury['severity'] = np.where(df_injury.get('total_deaths', 0).fillna(0) > 0, 'High', np.where(df_injury.get('total_injuries', 0).fillna(0) > 0, 'Medium', 'Low'))
df_injury['equipment_involved'] = 'Unknown'

incidents = []
for _, row in tqdm(df_injury.iterrows(), total=len(df_injury), desc="Injury incidents"):
    location = Location.objects.create(name="Unknown", address="N/A", site_manager="N/A")
    itype = IncidentType.objects.create(name="General Injury", description="OSHA reported")
    incidents.append(Incident(
        date=row['date'],
        description=row['description'],
        severity=row['severity'],
        equipment_involved=row['equipment_involved'],
        location=location,
        type=itype,
        data_source='injury'
    ))

batch_create(Incident, incidents)

# Process SIR data
print("Processing SIR data...")
df_sir['date'] = pd.to_datetime(df_sir.get('EventDate', pd.Series()), format='%m/%d/%Y', errors='coerce').apply(lambda dt: timezone.make_aware(dt) if pd.notna(dt) else timezone.now())
df_sir['description'] = df_sir.get('Final Narrative', '').fillna('No narrative')
df_sir['severity'] = np.where((df_sir.get('Hospitalized', 0) > 0) | (df_sir.get('Amputation', 0) > 0), 'High', 'Medium')
df_sir['equipment_involved'] = df_sir.get('Secondary Source Title', df_sir.get('SourceTitle', 'None')).fillna('None')

incidents = []
for _, row in tqdm(df_sir.iterrows(), total=len(df_sir), desc="SIR incidents"):
    location = Location.objects.create(name="Unknown", address="N/A", site_manager="N/A")
    itype = IncidentType.objects.create(name=row.get('EventTitle', 'Unknown'), description="SIR reported")
    incidents.append(Incident(
        date=row['date'],
        description=row['description'],
        severity=row['severity'],
        equipment_involved=row['equipment_involved'],
        location=location,
        type=itype,
        data_source='sir'
    ))

batch_create(Incident, incidents)

# Process Abstracts data
print("Processing Abstracts data...")
df_abstracts['date'] = pd.to_datetime(df_abstracts.get('Event Date', pd.Series()), format='%m/%d/%Y', errors='coerce').apply(lambda dt: timezone.make_aware(dt) if pd.notna(dt) else timezone.now())
df_abstracts['description'] = df_abstracts.get('Abstract Text', '').fillna('No abstract')
df_abstracts['severity'] = df_abstracts.get('Degree of Injury', 'Medium').fillna('Medium')
df_abstracts['equipment_involved'] = df_abstracts.get('hazsub', 'None').fillna('None')

incidents = []
for _, row in tqdm(df_abstracts.iterrows(), total=len(df_abstracts), desc="Abstracts incidents"):
    location = Location.objects.create(name="Unknown", address="N/A", site_manager="N/A")
    itype = IncidentType.objects.create(name="Abstract Incident", description="OSHA abstracts")
    incidents.append(Incident(
        date=row['date'],
        description=row['description'],
        severity=row['severity'],
        equipment_involved=row['equipment_involved'],
        location=location,
        type=itype,
        data_source='abstracts'
    ))

batch_create(Incident, incidents)

# Optional simulated data - EPCs would replace this section with a pull from their own historical data
print("Generating 10K simulated incidents...")
fake = Faker()
types = ['Falls', 'Struck By', 'Caught In', 'Electrical', 'Chemical']
for i in tqdm(range(10000), desc="Simulated"):
    location = Location.objects.create(name=fake.city(), address=fake.address(), site_manager=fake.name())
    itype_name = random.choice(types)
    itype = IncidentType.objects.create(name=itype_name, description=f"Simulated {itype_name.lower()}")
    Incident.objects.create(
        date=timezone.make_aware(fake.date_time_this_decade()),
        description=fake.paragraph(nb_sentences=3),
        severity=random.choice(['Low', 'Medium', 'High']),
        weather=random.choice(['Clear', 'Rainy', 'Hot', 'Cold']),
        equipment_involved=random.choice(['Crane', 'Scaffold', 'Welder', 'Ladder']),
        location=location,
        type=itype,
        data_source='simulated'
    )

print("All data loaded successfully!")