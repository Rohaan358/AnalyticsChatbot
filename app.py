import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any

# --- CONFIGURATION ---
load_dotenv(override=True)
DB_URL = os.getenv("DATABASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY").strip() if os.getenv("LLM_API_KEY") else None
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")

client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

import json

CACHE_FILE = "query_cache.json"

def load_query_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_to_query_cache(question, sql):
    cache = load_query_cache()
    normalized_q = question.lower().strip()
    cache[normalized_q] = sql
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

CHATS_DIR = "chats"
if not os.path.exists(CHATS_DIR):
    os.makedirs(CHATS_DIR)

def save_session(session_id, messages):
    if not messages: return
    # Find or generate a title based on the first user message
    title = session_id
    if len(messages) >= 1:
        first_q = messages[0]["content"]
        # If it's a new session, we might want to get a cleaner title from LLM
        if session_id.startswith("New_Session_"):
            try:
                res = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": f"Briefly title this pharma data question (max 4 words): {first_q}"}],
                    timeout=5.0
                )
                title = res.choices[0].message.content.strip().replace(" ", "_").replace('"', '').replace("'", "")
            except:
                title = first_q[:20].replace(" ", "_")
    
    file_path = os.path.join(CHATS_DIR, f"{title}.json")
    # Save with dataframes converted to dict for JSON
    serializable_msgs = []
    for m in messages:
        m_copy = m.copy()
        if "data" in m_copy and isinstance(m_copy["data"], pd.DataFrame):
            m_copy["data"] = m_copy["data"].to_dict(orient="records")
        serializable_msgs.append(m_copy)
        
    with open(file_path, "w") as f:
        json.dump(serializable_msgs, f, indent=4)
    return title

def load_session(filename):
    with open(os.path.join(CHATS_DIR, filename), "r") as f:
        msgs = json.load(f)
        for m in msgs:
            if "data" in m and m["data"] is not None:
                m["data"] = pd.DataFrame(m["data"])
        return msgs

# --- DATABASE ENGINE ---
def run_sql_query(query: str):
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
    query_u = query.upper()
    if not query_u.strip().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed."}
    for word in forbidden:
        if re.search(rf'\b{word}\b', query_u):
            return {"error": f"Keyword {word} is not allowed."}

    try:
        clean_url = DB_URL.split('?')[0] if DB_URL else ""
        conn = psycopg2.connect(clean_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def get_schema():
    try:
        with open("prisma/schema.prisma", "r") as f:
            return f.read()
    except:
        return "Schema unavailable."

@st.cache_data(ttl=600)
def get_rag_context(user_query: str):
    """Retrieves relevant business logic and SQL examples."""
    context = ""
    knowledge_dir = "knowledge"
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".md"):
                with open(os.path.join(knowledge_dir, filename), "r") as f:
                    content = f.read()
                    context += f"\n--- From {filename} ---\n{content}\n"
    
    # Add explicit warning about table existence and column mapping based on schema analysis
    context += "\n--- DATABASE SCHEMA MAPPINGS & LIMITATIONS (DO NOT IGNORE) ---\n"
    context += "- !! IMPORTANT !!: Table 'orders' is EMPTY (0 rows). DO NOT USE 'orders' or 'order_details' for sales queries.\n"
    context += "- !! RULE !!: Use 'invoice_details' for ALL Internal Sales/Quantity queries.\n"
    context += "- RULE: Join 'invoice' -> 'invoice_details' to get 'product_quantity'.\n"
    context += "- RULE: Join 'customer_details' -> 'invoice' using cd.customer_id = inv.cust_id.\n"
    context += "- RULE: Always CAST \"invoice_details\".\"product_quantity\" AS NUMERIC for SUM operations.\n"
    context += "- TABLE 'doctors': Use 'category' for doctor segments (A, B, C, D).\n"
    context += "- TABLE 'doctor_calls': DOES NOT EXIST. Use 'doctor_plan' for all visit-related queries.\n"
    context += "- TABLE 'ims_sale': Use this for Market units. Column: 'unit'.\n"
    return context

@st.cache_data(ttl=1800)
def get_executive_kpis():
    """Fetches high-level business metrics for the homepage."""
    kpis = {
        "internal_sales": 0,
        "market_sales": 0,
        "top_brick": "N/A",
        "doc_count": 0
    }
    
    # 1. Total Internal Units (from Invoices)
    res = run_sql_query('SELECT SUM(CAST("product_quantity" AS NUMERIC)) as total FROM "invoice_details"')
    if res and not isinstance(res, dict): kpis["internal_sales"] = res[0]["total"] or 0
    
    # 2. Total Market Units (from IMS)
    res = run_sql_query('SELECT SUM("unit") as total FROM "ims_sale"')
    if res and not isinstance(res, dict): kpis["market_sales"] = res[0]["total"] or 0
    
    # 3. Top Performing Brick
    res = run_sql_query('''
        SELECT "b"."name", SUM(CAST("id"."product_quantity" AS NUMERIC)) as total 
        FROM "ims_brick" "b"
        JOIN "customer_details" "cd" ON "b"."id" = "cd"."ims_brick_id"
        JOIN "invoice" "inv" ON "cd"."customer_id" = "inv"."cust_id"
        JOIN "invoice_details" "id" ON "inv"."id" = "id"."invoice_id"
        GROUP BY "b"."name" ORDER BY total DESC LIMIT 1
    ''')
    if res and not isinstance(res, dict): kpis["top_brick"] = res[0]["name"]
    
    # 4. Total Active Doctors
    res = run_sql_query('SELECT COUNT(*) as total FROM "doctors"')
    if res and not isinstance(res, dict): kpis["doc_count"] = res[0]["total"] or 0
    
    return kpis

# --- SESSION INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session" not in st.session_state:
    st.session_state.current_session = f"New_Session_{int(pd.Timestamp.now().timestamp())}"
if "prompt_trigger" not in st.session_state:
    st.session_state.prompt_trigger = None

# --- PAGE CONFIG ---
st.set_page_config(page_title="Antigravity Pharma AI", page_icon="💊", layout="wide")

# ... CSS remains same ...

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTextInput > div > div > input { background-color: #262730; color: white; border-radius: 10px; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("💊 Pharma Intelligence ChatBot")
st.caption("AI Agent with RAG (Retrieval-Augmented Generation)")
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Chat Sessions")
    chat_files = [f for f in os.listdir(CHATS_DIR) if f.endswith(".json")]
    
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.current_session = f"New_Session_{int(pd.Timestamp.now().timestamp())}"
        st.rerun()

    if chat_files:
        selected_chat = st.selectbox("Past Conversations", ["Select..."] + chat_files)
        if selected_chat != "Select...":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load"):
                    st.session_state.messages = load_session(selected_chat)
                    st.session_state.current_session = selected_chat.replace(".json", "")
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete"):
                    file_to_del = os.path.join(CHATS_DIR, selected_chat)
                    if os.path.exists(file_to_del):
                        os.remove(file_to_del)
                        st.session_state.messages = []
                        st.session_state.current_session = f"New_Session_{int(pd.Timestamp.now().timestamp())}"
                        st.toast(f"Deleted {selected_chat}")
                        st.rerun()
    
    st.divider()
    with st.expander("📊 Data Health Check"):
        st.success("IMS Market Sales: 156k+ records")
        st.success("Internal Sales (Invoices): 55k+ records")
        st.success("Doctors: 382 records")
        st.warning("⚠️ Orders & Targets: 0 records")
    
    st.divider()
    if st.button("Clear History"):
        st.session_state.messages = []
    

# --- HELPER: Handle question submission ---
def submit_question(q):
    st.session_state.prompt_trigger = q

# --- STARTER QUESTIONS (Show only if no messages) ---
if not st.session_state.messages:
    st.write("### 💡 Start with a sample report:")
    all_starters = [
        "Compare top 5 bricks by internal units vs market units",
        "Show me top 5 Category A doctors",
        "Which 3 products have the highest invoice quantity?",
        "Compare internal sales vs market sales in F.B.AREA",
        "Which brick has the highest internal units sold?",
        "List top 5 doctors by visit count in doctor_plan",
        "Compare market units of 'Product A' vs 'Product B' across bricks",
        "Show internal sales trend for F.B.AREA region",
        "Which Team has the highest target vs achievement?",
        "What is the market share of Karachi brick?"
    ]
    # Free shuffling - No Tokens used!
    random.shuffle(all_starters)
    starters = all_starters[:4]
    
    cols = st.columns(2)
    for i, s in enumerate(starters):
        with cols[i % 2]:
            if st.button(s, key=f"start_{i}"):
                submit_question(s)
                st.rerun()

# --- CHAT INTERFACE ---
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message:
            df_hist = pd.DataFrame(message["data"])
            # USE COERCE to handle 'None' values seen in screenshot
            # This ensures the column type becomes numeric (with NaNs) even with empty rows
            for c in df_hist.columns:
                if "id" not in c.lower() or c.lower().endswith("_id"):
                    df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')
            
            hist_num_cols = df_hist.select_dtypes(include=['number']).columns.tolist()
            st.dataframe(
                df_hist, 
                column_config={col: st.column_config.NumberColumn(format="%,d") for col in hist_num_cols},
                use_container_width=True
            )
        if message.get("chart_data") is not None:
            # Re-generate chart to avoid session state serialization issues
            x_col, y_cols = message["chart_data"]
            fig = px.bar(message["data"], x=x_col, y=y_cols, barmode='group', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True, key=f"chart_hist_{idx}")
        
        # Display FOLLOW-UP buttons if they exist in the message metadata
        if "follow_ups" in message and message["follow_ups"] and idx == len(st.session_state.messages) - 1:
            st.write("---")
            st.write("🔍 **Suggested Follow-ups:**")
            num_follow_ups = len(message["follow_ups"])
            if num_follow_ups > 0:
                f_cols = st.columns(num_follow_ups)
                for f_idx, f_text in enumerate(message["follow_ups"]):
                    with f_cols[f_idx]:
                        if st.button(f_text, key=f"follow_{idx}_{f_idx}"):
                            submit_question(f_text)
                            st.rerun()

# Use either chat_input or a click from starters/follow-ups
user_input = st.chat_input("Ask about your pharma data...")
prompt = user_input or st.session_state.prompt_trigger

if prompt:
    st.session_state.prompt_trigger = None # Reset
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Retrieving Data..."):
        try:
            # --- 1. QUICKEST CACHE CHECK (No File Reads) ---
            cache = load_query_cache()
            normalized_q = prompt.lower().strip()
            
            sql_query = ""
            results = None
            is_cached = False
            
            if normalized_q in cache:
                sql_query = cache[normalized_q]
                results = run_sql_query(sql_query)
                if not (isinstance(results, dict) and "error" in results):
                    is_cached = True

            # --- 2. LLM GENERATION ONLY ON CACHE MISS ---
            if not is_cached:
                # Cache misses perform file reads
                schema = get_schema()
                rag_context = get_rag_context(prompt)
                
                max_retries = 3
                current_attempt = 0
                last_error = ""
                
                while current_attempt < max_retries:
                    current_attempt += 1
                    error_feedback = f"\n\nLAST ERROR WAS: {last_error}" if last_error else ""
                    gen_prompt = (
                        f"KNOWLEDGE BASE & BUSINESS LOGIC:\n{rag_context}\n\n"
                        f"DATABASE SCHEMA:\n{schema}\n\n"
                        f"IMPORTANT DATA STATS (REAL-TIME):\n"
                        f"- 'invoice_details' for INTERNAL SALES. 'ims_sale' for MARKET.\n"
                        f"- 'orders' and 'targets' are EMPTY. DO NOT USE.\n\n"
                        f"USER QUESTION: {prompt}"
                        f"{error_feedback}\n\n"
                        "RULES: Return ONLY valid PostgreSQL SELECT query inside triple backticks. Use double quotes for identifiers. Use LIMIT 10."
                    )
                    
                    # ATTEMPT GENERATION
                    try:
                        response = client.chat.completions.create(
                            model=LLM_MODEL,
                            messages=[{"role": "system", "content": "You are a professional database agent."}, {"role": "user", "content": gen_prompt}],
                            timeout=30.0
                        )
                    except Exception as api_err:
                        # AUTO FALLBACK if primary is rate-limited (429) or fails
                        if "429" in str(api_err) or "rate-limit" in str(api_err).lower():
                            st.toast("⚠️ Primary Model busy, trying fallback (Llama 8B)...", icon="🔀")
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[{"role": "system", "content": "You are a professional database agent."}, {"role": "user", "content": gen_prompt}],
                                timeout=20.0
                            )
                        else:
                            raise api_err
                    
                    content = response.choices[0].message.content
                    sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
                    sql_query = sql_match.group(1).strip() if sql_match else content.strip()

                    results = run_sql_query(sql_query)
                    
                    if isinstance(results, dict) and "error" in results:
                        last_error = results["error"]
                        continue
                    else:
                        save_to_query_cache(prompt, sql_query)
                        break

            # Final Output Handling
            if isinstance(results, dict) and "error" in results:
                st.error(f"Failed after {max_retries} attempts. Final Error: {results['error']}")
            else:
                if is_cached:
                    st.toast("⚡ Result served from Cache (SQL generation skipped)", icon="🔥")
                
                df = pd.DataFrame(results)
                
                if df.empty:
                    final_answer = "I executed the query, but it returned no data. This usually means the specific filters (like region name) didn't match anything in the database."
                else:
                    # IF CACHED: Skip AI summary for 1-second performance
                    if is_cached:
                        final_answer = f"⚡ **(Cached Response)**\n\nHere are the latest results from the database for your request. The SQL logic was retrieved from history for high-speed performance."
                    else:
                        sum_prompt = (
                            f"User asked: {prompt}\nDB Result: {results}\n\n"
                            "Task: Provide a concise (1-2 sentence) business summary of the results.\n"
                            "STRICT RULES:\n"
                            "- Just summarize the numeric findings simply. BE VERY BRIEF."
                        )
                        summary_res = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": sum_prompt}], timeout=30.0)
                        final_answer = summary_res.choices[0].message.content
                
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                    # st.code(sql_query, language="sql") # Optional: Show SQL for debugging
                    chart_data = None
                    if not df.empty:
                        # Force conversion to ensure commas work (COERCE handles None/NaNs correctly)
                        for col in df.columns:
                            if "id" not in col.lower() or col.lower().endswith("_id"):
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Identify numeric columns for comma formatting
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        # Display with commas using Streamlit's Column Config
                        st.dataframe(
                            df, 
                            column_config={col: st.column_config.NumberColumn(format="%,d") for col in numeric_cols},
                            use_container_width=True
                        )
                        
                        # --- Enhanced Multi-Column Auto-Charting ---
                        chart_data = None
                        if len(df) > 1:
                            # Use the first column as X
                            x_col = df.columns[0]
                            
                            if len(numeric_cols) > 0:
                                # Ensure we have multiple bars if there are multiple stats
                                fig = px.bar(
                                    df, 
                                    x=x_col, 
                                    y=numeric_cols, 
                                    barmode='group',
                                    template="plotly_dark",
                                    title=f"Comparison: {', '.join(numeric_cols)} per {x_col}"
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(st.session_state.messages)}")
                                
                                # Store for history (Original raw numeric cols list)
                                chart_data = (x_col, numeric_cols)
                
                # --- GENERATE FOLLOW-UPS ---
                follow_ups = []
                try:
                    f_prompt = f"Based on this answer: '{final_answer}', suggest 2 very short (max 5 words) follow-up questions about the data."
                    f_res = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": f_prompt}], timeout=15.0)
                    raw_f = f_res.choices[0].message.content
                    follow_ups = [q.strip('1234. -').strip() for q in raw_f.split('\n') if '?' in q][:2]
                except:
                    follow_ups = []

                st.session_state.messages.append({"role": "assistant", "content": final_answer, "data": df, "chart_data": chart_data, "follow_ups": follow_ups})
                # Auto-save
                new_id = save_session(st.session_state.current_session, st.session_state.messages)
                if st.session_state.current_session.startswith("New_Session_"):
                    st.session_state.current_session = new_id
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
