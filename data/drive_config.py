import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')
DATA_DIR = Path('data')

DATA_DIR.mkdir(exist_ok=True)
LAST_SYNC_FILE = DATA_DIR / '.last_sync'

def get_last_sync_time():
    if LAST_SYNC_FILE.exists():
        with open(LAST_SYNC_FILE, 'r') as f:
            return f.read().strip()
    return None

def update_last_sync_time(timestamp):
    with open(LAST_SYNC_FILE, 'w') as f:
        f.write(timestamp)

if not os.getenv('GOOGLE_CREDENTIALS'):
    print("Warning: GOOGLE_CREDENTIALS not found in environment")