import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

import decimal
import json

# Custom JSON encoder to handle PostgreSQL Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

CACHE_FILE = "query_cache.json"

def smart_format_dataframe(df: pd.DataFrame):
    """
    Returns:
        df_numeric  -> pure numeric (for charts / calculations)
        df_display  -> formatted with commas strings (for st.dataframe UI)
    """
    if df.empty or len(df.columns) == 0:
        return df, df
        
    df_numeric = df.copy()
    df_display = df.copy() # Start with original data to avoid data loss

    # 🔹 Identify and convert numeric columns safely
    num_cols = []
    for col in df_numeric.columns:
        # SKIP the first column - it's our X-axis category (e.g., Brick Name, Product)
        if col == df_numeric.columns[0]:
            continue
            
        # Try to convert to numeric, if it fails, it's a category
        tmp = pd.to_numeric(df_numeric[col], errors='coerce')
        if tmp.notnull().any():
            df_numeric[col] = tmp.fillna(0).astype(float)
            num_cols.append(col)

    # 🔹 Apply formatting to display dataframe for numeric columns only
    for col in num_cols:
        # Detect integer-like vs float
        if (df_numeric[col].dropna() % 1 == 0).all():
            df_display[col] = df_numeric[col].apply(
                lambda x: f"{int(x):,}" if pd.notnull(x) else ""
            )
        else:
            df_display[col] = df_numeric[col].apply(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
            )

    # Force the first column (Category) to be string to ensure Plotly treats it correctly
    df_numeric[df_numeric.columns[0]] = df_numeric[df_numeric.columns[0]].astype(str)

    return df_numeric, df_display

def plot_smart_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str, key: str):
    """
    Plots a chart with support for dual Y-axes if columns have vastly different scales
    (e.g., Quantity in millions vs Revenue in trillions).
    """
    if len(y_cols) == 2:
        v1 = df[y_cols[0]].abs().max()
        v2 = df[y_cols[1]].abs().max()
        
        # Check for Scale mismatch (10x or more) or Pharma specific keywords
        is_qty_val = any(k in y_cols[0].lower() for k in ["qty", "unit"]) and \
                     any(k in y_cols[1].lower() for k in ["rev", "val", "price", "sale"])
        is_val_qty = any(k in y_cols[1].lower() for k in ["qty", "unit"]) and \
                     any(k in y_cols[0].lower() for k in ["rev", "val", "price", "sale"])
        
        if is_qty_val or is_val_qty or (v1 > 0 and v2 > 0 and (v1/v2 > 10 or v2/v1 > 10)):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Smaller scale usually goes on Primary Y as Bars, Larger scale on Secondary Y as Line
            if v1 < v2:
                p_col, s_col = y_cols[0], y_cols[1]
            else:
                p_col, s_col = y_cols[1], y_cols[0]
                
            fig.add_trace(go.Bar(x=df[x_col], y=df[p_col], name=p_col, marker_color='#636EFA'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df[x_col], y=df[s_col], name=s_col, marker_color='#EF553B', mode='lines+markers'), secondary_y=True)
            
            fig.update_layout(
                title=title, template="plotly_dark", hovermode="x unified",
                xaxis={'type':'category', 'categoryorder':'total descending'},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text=p_col, secondary_y=False)
            fig.update_yaxes(title_text=s_col, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True, key=key)
            return

    # Standard Grouped Bar
    fig = px.bar(df, x=x_col, y=y_cols, barmode='group', template="plotly_dark", title=title)
    fig.update_layout(xaxis={'type':'category', 'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True, key=key)

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
        json.dump(serializable_msgs, f, indent=4, cls=DecimalEncoder)
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
    
    # Add explicit instructions for complex joins and case-insensitive matching
    context += "\n--- MASTER SQL RULES (MANDATORY) ---\n"
    context += "1. USE 'master_sale' FOR INTERNAL METRICS: It has product_name, product_quantity, total_amount, region_name, zone_name.\n"
    context += "2. JOINING BRICK NAMES: 'master_sale' does NOT have brick_name. For brick-level internal sales, use this join:\n"
    context += "   SELECT ib.name, SUM(ms.product_quantity) FROM master_sale ms JOIN customer_details cd ON ms.customer_id = cd.customer_id JOIN ims_brick ib ON cd.ims_brick_id = ib.id GROUP BY ib.name\n"
    context += "3. NO CARTESIAN PRODUCTS: Never join 'ims_sale' and 'invoice_details' (or 'master_sale') in a single table join. It will multiply the numbers incorrectly.\n"
    context += "   - To compare Internal vs Market, use CTEs (Common Table Expressions):\n"
    context += "   WITH Internal AS (SELECT ib.name, SUM(ms.product_quantity) as qty FROM master_sale ms JOIN customer_details cd ON ms.customer_id = cd.customer_id JOIN ims_brick ib ON cd.ims_brick_id = ib.id GROUP BY 1),\n"
    context += "        Market AS (SELECT ib.name, SUM(unit) as qty FROM ims_sale s JOIN ims_brick ib ON s.\"brickId\" = ib.id GROUP BY 1)\n"
    context += "   SELECT i.name, i.qty AS \"Internal\", m.qty AS \"Market\" FROM Internal i JOIN Market m ON i.name = m.name\n"
    context += "4. FUZZY SEARCH (ILIKE): For all user filters (like 'Gulshan'), use ILIKE with wildcards: WHERE ib.name ILIKE '%gulshan%'.\n"
    context += "5. GULSHAN AGGREGATION: Gulshan has many blocks (Block 5, 6, etc.). Always use ILIKE '%gulshan%' to aggregate all of them.\n"
    context += "6. EMPTY TABLES: 'orders' and 'targets' are EMPTY. For sales, always use 'master_sale' or 'invoice_details'.\n"
    
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
st.set_page_config(page_title="Pharma Intelligence", page_icon="💊", layout="wide")

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
            df_raw = pd.DataFrame(message["data"])
            df_numeric_hist, df_display_hist = smart_format_dataframe(df_raw)
            # Table use formatted
            st.dataframe(df_display_hist, use_container_width=True)
            
            # --- Historical Charts ---
            if message.get("chart_data") is not None:
                x_col, y_cols = message["chart_data"]
                # Filter to ensure Y columns exist and are numeric
                y_cols_valid = [c for c in y_cols if c in df_numeric_hist.columns]
                if y_cols_valid:
                    plot_smart_chart(
                        df_numeric_hist, 
                        x_col, 
                        y_cols_valid, 
                        f"History: {', '.join(y_cols_valid)}", 
                        f"chart_hist_{idx}"
                    )
        
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
                        f"QUERY VALIDATION RULES (CRITICAL BEFORE EXECUTING):\n"
                        f"1. Verify table exists in schema (e.g. 'orders' is EMPTY, 'master_sale' is NOT).\n"
                        f"2. Verify all column names exist exactly as written.\n"
                        f"3. If filter value may differ, ALWAYS use ILIKE '%value%'.\n"
                        f"4. If your previous query returned 0 rows, you MUST generate a simple query to inspect available values: SELECT DISTINCT <column_name> FROM <table> LIMIT 10;\n"
                        f"5. Never return constant values like 0 for 'brick_name'. Always select the actual column.\n"
                        f"6. Group data by the real column name. Never use numeric indexes.\n"
                        f"7. Exclude NULL or missing categories using WHERE/HAVING filters.\n\n"
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
                        last_error = str(api_err)
                        continue
                    
                    content = response.choices[0].message.content
                    sql_match = re.search(r"```sql\n(.*?)\n```", content, re.DOTALL)
                    sql_query = sql_match.group(1).strip() if sql_match else content.strip()

                    results = run_sql_query(sql_query)
                    
                    if isinstance(results, dict) and "error" in results:
                        last_error = results["error"]
                        continue
                        
                    if not results and current_attempt < max_retries:
                        last_error = f"The query returned 0 rows. As per Validation Rule 4, generate a query to SELECT DISTINCT values from the filtered column to see what data is actually available, so the user knows what they can search for."
                        continue
                        
                    else:
                        if results: # Only save to cache if it actually returned data
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
                    # PROACTIVE DISCOVERY: Try to find similar bricks
                    discovery_msg = ""
                    potential_brick = re.search(r"ILIKE '%(.*?)%'", sql_query, re.I)
                    if potential_brick:
                        search_term = potential_brick.group(1)
                        discovery_results = run_sql_query(f"SELECT name FROM ims_brick WHERE name ILIKE '%{search_term}%' LIMIT 3")
                        if discovery_results:
                            names_list = "\n".join([f"- {r['name']}" for r in discovery_results])
                            discovery_msg = f"Available similar options in database:\n{names_list}"

                    # Let the LLM generate a friendly "no data" message
                    empty_prompt = (
                        f"User asked: {prompt}\n\n"
                        f"Your SQL query returned exactly 0 rows.\n"
                        f"To assist the user, explain politely that the specific filter they used might not match exactly.\n"
                        f"If I found similar available options in the database, here they are: {discovery_msg}\n"
                        f"Incorporate these options into your advice if they exist."
                    )
                    empty_res = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": empty_prompt}], timeout=30.0)
                    final_answer = empty_res.choices[0].message.content
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
                    with st.expander("🛠️ Show SQL Logic (Debug)"):
                        st.code(sql_query, language="sql")
                    chart_data = None
                    if not df.empty:
                        # Use Smart Formatting
                        df_numeric, df_display = smart_format_dataframe(df)
                        
                        # Table use formatted
                        st.dataframe(df_display, use_container_width=True)
                        
                        # --- Enhanced Multi-Column Auto-Charting ---
                        chart_data = None
                        if len(df_numeric) > 1:
                            # 1. First column is the category (X-axis)
                            x_col = df_numeric.columns[0]
                            
                            # 2. Identify numeric stats (Y-axis)
                            num_cols = df_numeric.select_dtypes(include=['number']).columns.tolist()
                            y_cols = [c for c in num_cols if c != x_col]
                            
                            # 3. Validation: X-axis must have valid categories (Not '0', NULL, or empty)
                            valid_rows = df_numeric.copy()
                            valid_rows[x_col] = valid_rows[x_col].astype(str).str.strip()
                            valid_rows = valid_rows[
                                (valid_rows[x_col] != "0") & 
                                (valid_rows[x_col] != "None") & 
                                (valid_rows[x_col] != "")
                            ]
                            if not valid_rows.empty and len(y_cols) > 0:
                                # --- Live Chart (Plotly with Smart Scaling) ---
                                plot_smart_chart(
                                    valid_rows, 
                                    x_col, 
                                    y_cols, 
                                    f"Analysis: {', '.join(y_cols)} per {x_col}", 
                                    f"chart_new_{len(st.session_state.messages)}"
                                )
                                chart_data = (x_col, y_cols)
                            else:
                                if len(y_cols) > 0:
                                    st.warning(f"⚠️ Cannot plot chart: Category column ('{x_col}') appears empty or invalid.")
                
                # --- GENERATE FOLLOW-UPS ---
                follow_ups = []
                try:
                    f_prompt = f"Based on this answer: '{final_answer}', suggest 2 very short (max 5 words) follow-up questions about the data."
                    f_res = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": f_prompt}], timeout=15.0)
                    raw_f = f_res.choices[0].message.content
                    follow_ups = [q.strip('1234. -').strip() for q in raw_f.split('\n') if '?' in q][:2]
                except:
                    follow_ups = []

                st.session_state.messages.append({"role": "assistant", "content": final_answer, "data": df_numeric if not df.empty else df, "chart_data": chart_data, "follow_ups": follow_ups})
                # Auto-save
                new_id = save_session(st.session_state.current_session, st.session_state.messages)
                if st.session_state.current_session.startswith("New_Session_"):
                    st.session_state.current_session = new_id
                st.rerun()
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
