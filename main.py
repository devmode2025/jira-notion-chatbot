import streamlit as st
from app import daily_update_form, display_daily_updates

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat with Data", "Add Daily Update", "View Entries"])
    
    if page == "Chat with Data":
        st.title("🤖 Jira/Notion Progress Chat")
        # Your existing chat code here...
        user_query = st.text_input("What would you like to know?", "What was I working on last week?")
        if user_query:
            # Your existing chat logic...
            pass
            
    elif page == "Add Daily Update":
        content = daily_update_form()
        if content:
            st.success("✅ Daily update saved!")
            st.markdown("### Preview:")
            st.markdown(content)
            
            # Here you would add code to actually save to database
            st.info("Note: Add database saving code in save_daily_update() function")
            
    elif page == "View Entries":
        display_daily_updates()

def save_daily_update(date, completed, in_progress, blockers, priority1, priority2, priority3, epic_progress, key_milestone):
    """Format and save the daily update to the database"""
    
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
        connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = connection.cursor()
        
        # Generate embedding
        embedding = get_embedding(content)
        embedding_array = np.array(embedding)
        embedding_str = "[" + ",".join(map(str, embedding_array)) + "]"
        
        # Insert into database
        cursor.execute(
            "INSERT INTO notion_entries (date, content, embedding) VALUES (%s, %s, %s::vector)",
            (date.strftime("%Y-%m-%d"), content, embedding_str)
        )
        
        connection.commit()
        cursor.close()
        connection.close()
        
        st.success("✅ Daily update saved to database!")
        return content
        
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return None
    
if __name__ == "__main__":
    main()