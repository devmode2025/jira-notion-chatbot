import os
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

print(f"NOTION_API_KEY: {NOTION_API_KEY[:10]}...")  # Show first 10 chars
print(f"NOTION_DATABASE_ID: {NOTION_DATABASE_ID}")

headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

# Test Notion connection
try:
    response = requests.get(f'https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}', headers=headers)
    print(f"Notion API Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Notion connection successful!")
    else:
        print(f"❌ Notion error: {response.text}")
except Exception as e:
    print(f"❌ Notion connection failed: {e}")