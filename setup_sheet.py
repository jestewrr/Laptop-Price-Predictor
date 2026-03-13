import gspread
import json
import os
from google.oauth2.service_account import Credentials

# Load credentials from environment variable
creds_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
if creds_json:
    creds_dict = json.loads(creds_json)
else:
    # Fallback: load from file
    with open('credentials.json', 'r') as f:
        creds_dict = json.load(f)

scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(creds)

print("Creating database spreadsheet...")
sheet = client.create("Laptop Price Predictions").sheet1

print("Writing headers...")
sheet.append_row(["timestamp", "ram_size", "storage_rom", "processor", "display_quality", "human_model_price", "ai_model_price"])

print(f"\nSheet has been successfully created!")
print(f"URL: https://docs.google.com/spreadsheets/d/{sheet.spreadsheet.id}\n")

import sys
print("Done. It is now linked to your App!")
sys.exit(0)
