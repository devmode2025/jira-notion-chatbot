import streamlit as st
import psycopg2
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import time
from datetime import datetime, timedelta  # Add timedelta to existing import
import threading
from contextlib import contextmanager
from functools import lru_cache

# Add this after your imports, before function definitions
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page config for better performance
st.set_page_config(
    page_title="Jira/Notion Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection pool
_db_connection = None
_db_lock = threading.Lock()

@contextmanager
def get_db_connection():
    global _db_connection
    with _db_lock:
        if _db_connection is None or _db_connection.closed:
            _db_connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        yield _db_connection

# Cache embeddings to avoid repeated OpenAI calls
@lru_cache(maxsize=100)
def get_embedding_cached(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    try:
        return client.embeddings.create(
            input=[text], 
            model=model,
            timeout=30
        ).data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return []

def get_relevant_context(query_embedding, limit=3):
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            cursor.execute(
                "SELECT content FROM notion_entries ORDER BY embedding <=> %s::vector LIMIT %s;",
                (embedding_str, limit)
            )
            results = cursor.fetchall()
            return "\n\n".join([result[0] for result in results])
    except Exception as e:
        st.error(f"Database error: {e}")
        return ""

def daily_update_form():
    st.header("📝 Daily Progress Form")
    
    with st.form("daily_update"):
        entry_date = st.date_input("Date", datetime.now())
        
        st.subheader("Today's Progress")
        completed = st.text_area("Completed Tasks", 
                               placeholder="MVAI-51 React Foundation Setup ✅, MVAI-52 AI Chat Interface ✅")
        in_progress = st.text_area("In Progress", 
                                  placeholder="MVAI-53 Session Management UI (25% complete)")
        blockers = st.text_input("Blockers", placeholder="None")
        
        st.subheader("Tomorrow's Plan")
        priority1 = st.text_input("Priority 1", placeholder="Complete MVAI-53 Session Management UI")
        priority2 = st.text_input("Priority 2", placeholder="Begin MVAI-54 Real-time Messaging")
        priority3 = st.text_input("Priority 3", placeholder="UX polish and testing")
        
        st.subheader("Epic Impact")
        epic_progress = st.text_input("Epic Progress", placeholder="Epic 5 Progress: 40% complete (2/5 stories done)")
        key_milestone = st.text_input("Key Milestone", placeholder="Frontend AI Integration foundation solid")
        
        submitted = st.form_submit_button("Save Daily Update")
        
        if submitted:
            return save_daily_update(
                entry_date, completed, in_progress, blockers,
                priority1, priority2, priority3, epic_progress, key_milestone
            )
    return None

def save_daily_update(date, completed, in_progress, blockers, priority1, priority2, priority3, epic_progress, key_milestone):
    content = f"""
**Today's Progress:**
- Completed: {completed}
- In Progress: {in_progress}
- Blockers: {blockers}

**Tomorrow's Plan:**
- Priority 1: {priority1}
- Priority 2: {priority2}
- Priority 3: {priority3}

**Epic Impact:**
- {epic_progress}
- Key Milestone: {key_milestone}
"""
    
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            
            embedding = get_embedding_cached(content)
            embedding_array = np.array(embedding)
            embedding_str = "[" + ",".join(map(str, embedding_array)) + "]"
            
            cursor.execute(
                "INSERT INTO notion_entries (date, content, embedding) VALUES (%s, %s, %s::vector)",
                (date.strftime("%Y-%m-%d"), content, embedding_str)
            )
            
            connection.commit()
        
        st.success("✅ Daily update saved to database!")
        return content
        
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return None

def display_daily_updates():
    st.header("📊 Advanced Entry Browser")
    
    # Create search and controls layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search content:", 
                                  placeholder="Search keywords, tickets, blockers...")
    
    with col2:
        items_per_page = st.selectbox("Entries per page:", [5, 10, 25], index=0)
    
    with col3:
        sort_order = st.selectbox("Sort by:", ["Newest first", "Oldest first"])
    
    # Date range filter
    st.subheader("📅 Date Filter", help="Filter entries by date range")
    date_col1, date_col2 = st.columns(2)
    
    with date_col1:
        start_date = st.date_input("From date:", value=None, help="Leave empty for no start date filter")
    
    with date_col2:
        end_date = st.date_input("To date:", value=None, help="Leave empty for no end date filter")
    
    # Quick date presets
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    with preset_col1:
        if st.button("This Week", use_container_width=True):
            today = datetime.now().date()
            start_date = today - timedelta(days=today.weekday())
            end_date = today
    with preset_col2:
        if st.button("Last 7 Days", use_container_width=True):
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
    with preset_col3:
        if st.button("This Month", use_container_width=True):
            today = datetime.now().date()
            start_date = today.replace(day=1)
            end_date = today
    with preset_col4:
        if st.button("Clear Dates", use_container_width=True):
            start_date = None
            end_date = None
            st.rerun()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic query based on filters
            query_params = []
            query_conditions = []
            
            # Text search condition
            if search_term:
                query_conditions.append("content ILIKE %s")
                query_params.append(f"%{search_term}%")
            
            # Date range conditions
            if start_date:
                query_conditions.append("date >= %s")
                query_params.append(start_date.strftime("%Y-%m-%d"))
            if end_date:
                query_conditions.append("date <= %s")
                query_params.append(end_date.strftime("%Y-%m-%d"))
            
            # Build the WHERE clause
            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""
            
            # Build the complete query
            query = f"""
                SELECT id, date, content 
                FROM notion_entries 
                {where_clause}
                ORDER BY date {'DESC' if sort_order == 'Newest first' else 'ASC'}
            """
            
            cursor.execute(query, query_params)
            entries = cursor.fetchall()
            
            if not entries:
                st.info("📭 No entries found" + 
                       (" matching your search" if search_term else "") +
                       (" in selected date range" if start_date or end_date else ""))
                return
            
            # Pagination setup
            total_entries = len(entries)
            total_pages = max(1, (total_entries + items_per_page - 1) // items_per_page)
            
            # Page control with session state
            page = st.session_state.current_page
            
            # Hidden number input for page control
            page_input = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=page, 
                step=1,
                key="page_selector",
                label_visibility="collapsed"
            )
            
            # Update session state if user changes the hidden input
            if page_input != st.session_state.current_page:
                st.session_state.current_page = page_input
                st.rerun()
            
            # Calculate slice indices
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_entries)
            
            # Show results info with filter summary
            st.subheader(f"📋 Results {start_idx + 1}-{end_idx} of {total_entries}")
            
            # Filter summary
            filter_summary = []
            if search_term:
                filter_summary.append(f"search: '{search_term}'")
            if start_date:
                filter_summary.append(f"from: {start_date}")
            if end_date:
                filter_summary.append(f"to: {end_date}")
            
            if filter_summary:
                st.caption("Filters: " + " | ".join(filter_summary))
            
            # Display entries for current page
            for i, (entry_id, entry_date, content) in enumerate(entries[start_idx:end_idx], start_idx + 1):
                with st.expander(f"#️⃣ {i}. 📅 {entry_date} (Entry #{entry_id})", expanded=False):
                    st.markdown(content)
                    st.caption(f"Entry ID: {entry_id} | Date: {entry_date}")
            
            # Page navigation controls
            if total_pages > 1:
                st.markdown("---")
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                
                with nav_col1:
                    if page > 1:
                        if st.button("⬅️ Previous", use_container_width=True):
                            st.session_state.current_page = page - 1
                            st.rerun()
                
                with nav_col2:
                    st.markdown(f"**Page {page} of {total_pages}**", help=f"Total entries: {total_entries}")
                
                with nav_col3:
                    if page < total_pages:
                        if st.button("Next ➡️", use_container_width=True):
                            st.session_state.current_page = page + 1
                            st.rerun()
                
    except Exception as e:
        st.error(f"❌ Error loading entries: {e}")
        st.info("Please check your database connection and try again.")
                
    except Exception as e:
        st.error(f"❌ Error loading entries: {e}")
        st.info("Please check your database connection and try again.")

def chat_interface():
    st.title("🤖 Jira/Notion Progress Chat")
    st.write("Ask me about your past progress, blockers, or plans!")

    user_query = st.text_input("What would you like to know?", "What was I working on last week?")

    if user_query:
        with st.spinner("Searching your progress..."):
            try:
                start_time = time.time()
                
                # Generate embedding
                embedding_start = time.time()
                query_embedding = get_embedding_cached(user_query)
                embedding_time = time.time() - embedding_start
                
                # Database query
                db_start = time.time()
                context = get_relevant_context(query_embedding)
                db_time = time.time() - db_start
                
                # LLM call
                llm_start = time.time()
                if context:
                    prompt = f"""You are an AI assistant that helps a software developer review their daily progress updates.
                    Answer the user's question based solely on the context provided below from their past Notion entries.

                    Context: {context}

                    User Question: {user_query}

                    If the context doesn't contain information relevant to the question, politely say you don't know.
                    Answer: """

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        timeout=30
                    )
                    answer = response.choices[0].message.content
                else:
                    answer = "No relevant context found."
                
                llm_time = time.time() - llm_start
                total_time = time.time() - start_time
                
                st.write("### Answer:")
                st.write(answer)
                
                with st.expander("⚡ Performance Metrics"):
                    st.write(f"**Total Time:** {total_time:.2f}s")
                    st.write(f"- Embedding Generation: {embedding_time:.2f}s")
                    st.write(f"- Database Query: {db_time:.2f}s") 
                    st.write(f"- LLM Response: {llm_time:.2f}s")
                    
                with st.expander("See relevant context used"):
                    st.write(context)
                    
            except Exception as e:
                st.error(f"Error: {e}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat with Data", "Add Daily Update", "View Entries"])
    
    if page == "Chat with Data":
        chat_interface()
            
    elif page == "Add Daily Update":
        content = daily_update_form()
        if content:
            st.markdown("### Preview:")
            st.markdown(content)
            
    elif page == "View Entries":
        display_daily_updates()

if __name__ == "__main__":
    main()