import os
import requests
from dotenv import load_dotenv
import json
from openai import OpenAI
import psycopg2
import numpy as np

load_dotenv()

# Your Notion API configuration
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """Generates an embedding for a given text using OpenAI's API."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def query_notion_database():
    """Queries the Notion database and returns the raw JSON response."""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()

def parse_notion_page(page):
    """Parses a Notion page object into a clean string based on our template."""
    props = page['properties']
    
    # Extract properties - adjust these names to match your Notion DB
    title = props.get('Title', {}).get('title', [{}])[0].get('text', {}).get('content', 'No Title')
    date = props.get('Date', {}).get('date', {}).get('start', 'No Date')
    
    # Extract rich text fields safely
    def extract_rich_text(property_name):
        rich_text_list = props.get(property_name, {}).get('rich_text', [])
        return " ".join([rt.get('text', {}).get('content', '') for rt in rich_text_list])
    
    completed = extract_rich_text('Completed')
    in_progress = extract_rich_text('In Progress')
    blockers = extract_rich_text('Blockers')

    # Combine into a clean text chunk
    clean_text = f"""
    Date: {date}
    Title: {title}
    Completed: {completed}
    In Progress: {in_progress}
    Blockers: {blockers}
    """
    return clean_text, date

def main():
    print("🚀 Starting ingestion script...")
    print("📋 Querying Notion database...")
    try:
        data = query_notion_database()
        pages = data['results']
        print(f"📄 Found {len(pages)} pages")
    except Exception as e:
        print(f"❌ Notion query failed: {e}")
        return

    print("🗄️ Connecting to database...")
    try:
        connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = connection.cursor()
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Create table if it doesn't exist
    print("📊 Creating table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notion_entries (
            id SERIAL PRIMARY KEY,
            date TEXT,
            content TEXT,
            embedding VECTOR(1536)
        );
    """)

    for i, page in enumerate(pages):
        print(f"📝 Processing page {i+1}/{len(pages)}...")
        try:
            content, date = parse_notion_page(page)
            print("  🤖 Generating embedding...")
            embedding = get_embedding(content)
            
            # Convert to PostgreSQL compatible format
            embedding_array = np.array(embedding)
            embedding_str = "[" + ",".join(map(str, embedding_array)) + "]"
            
            # Insert into the database
            print("  💾 Inserting into database...")
            cursor.execute(
                "INSERT INTO notion_entries (date, content, embedding) VALUES (%s, %s, %s::vector)",
                (date, content, embedding_str)
            )
        except Exception as e:
            print(f"  ❌ Error processing page: {e}")
            continue

    connection.commit()
    cursor.close()
    connection.close()
    print("✅ Data ingestion complete!")

if __name__ == "__main__":
    main()
