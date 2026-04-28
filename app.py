"""
app.py — Pharma Intelligence Streamlit App
Refactored to use:
  - db.py          → connection pool, parameterized auth, chat persistence
  - agent_core.py  → RAG context, SQL self-correction, streaming summaries
"""

import os
import re
import json
import random
import datetime
import decimal

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Internal modules (fixes live here) ──────────────────────────────────────
from db import verify_login, run_sql_query, upsert_db_chat, delete_db_chat, load_db_chat
from agent_core import (
    generate_and_run_sql,
    summarise_results_stream,
    suggest_followups,
    client as llm_client,
    LLM_MODEL,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pharma Intelligence", page_icon="💊", layout="wide")

# ── JSON encoder ─────────────────────────────────────────────────────────────
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STORAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_chats_dir() -> str:
    username = st.session_state.get("username", "default")
    d = os.path.join("chats", str(username).strip())
    os.makedirs(d, exist_ok=True)
    return d


# Shared geo cache (not per-user) — fixes the per-user cache bug
GEO_CACHE_FILE = os.path.join("chats", "geo_cache.json")
os.makedirs("chats", exist_ok=True)


def load_geo_cache() -> dict:
    try:
        with open(GEO_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_geo_cache(cache: dict):
    try:
        with open(GEO_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def reverse_geocode(lat: float, lng: float) -> str:
    key = f"{lat:.4f},{lng:.4f}"
    cache = load_geo_cache()
    if key in cache:
        return cache[key]
    try:
        import urllib.request, json as _json
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat:.4f}&lon={lng:.4f}&zoom=18"
        req = urllib.request.Request(url, headers={"User-Agent": "PharmaBot/2.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            addr = _json.loads(resp.read().decode()).get("display_name", "Address not found")
            cache[key] = addr
            save_geo_cache(cache)
            return addr
    except Exception:
        return f"Lat: {lat:.4f}, Lng: {lng:.4f}"


def get_query_cache_path() -> str:
    return os.path.join(get_chats_dir(), "query_cache.json")


def load_query_cache() -> dict:
    p = get_query_cache_path()
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}


def save_to_query_cache(question: str, sql: str):
    cache = load_query_cache()
    cache[question.lower().strip()] = sql
    with open(get_query_cache_path(), "w") as f:
        json.dump(cache, f, indent=4)


def save_session(session_id: str, messages: list) -> str:
    if not messages:
        return session_id
    username = st.session_state.get("username", "default")

    # Auto-generate session title from first question
    title = session_id
    if session_id.startswith("New_Session_"):
        try:
            first_q = messages[0]["content"]
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": f"Title this pharma question in max 4 words. Output ONLY the words: {first_q}"}],
                timeout=5.0,
            )
            raw = resp.choices[0].message.content.strip()
            title = re.sub(r'[\\/\*?:"<>|\n\r]+', "", raw).replace(" ", "_")[:50]
        except Exception:
            title = re.sub(r'[\\/\*?:"<>|\n\r]+', "", messages[0]["content"][:30]).replace(" ", "_")

    # Serialize (DataFrames → dicts)
    serialisable = []
    for m in messages:
        mc = m.copy()
        if "data" in mc and isinstance(mc["data"], pd.DataFrame):
            mc["data"] = mc["data"].to_dict(orient="records")
        serialisable.append(mc)

    json_str = json.dumps(serialisable, cls=DecimalEncoder)

    # Write local JSON
    os.makedirs(get_chats_dir(), exist_ok=True)
    with open(os.path.join(get_chats_dir(), f"{title}.json"), "w") as f:
        f.write(json_str)

    # Sync to DB
    upsert_db_chat(username, title, json_str)
    return title


def load_session(filename: str) -> list:
    title = filename.replace(".json", "")
    full_path = os.path.join(get_chats_dir(), filename)
    msgs = None

    if os.path.exists(full_path):
        try:
            with open(full_path, "r") as f:
                msgs = json.load(f)
        except Exception:
            pass

    if not msgs:
        raw = load_db_chat(title)
        if raw:
            msgs = raw if isinstance(raw, list) else json.loads(raw)
            with open(full_path, "w") as f:
                json.dump(msgs, f, cls=DecimalEncoder)

    if not msgs or not isinstance(msgs, list):
        return []

    result = []
    for m in msgs:
        if isinstance(m, dict):
            if "data" in m and m["data"] is not None:
                m["data"] = pd.DataFrame(m["data"])
            result.append(m)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# DATA FORMATTING & CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def smart_format_dataframe(df: pd.DataFrame):
    if df.empty:
        return df, df
    df_numeric = df.copy()
    df_display = df.copy()

    for col in df_numeric.columns:
        if df_numeric[col].dtype == object:
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors="ignore")
            except Exception:
                pass

    for col in df_numeric.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df_numeric[col])
        is_category = col == df_numeric.columns[0] and len(df_numeric.columns) > 1
        is_metric = any(k in str(col).lower() for k in ["sale","rev","price","total","qty","amount"])

        if is_numeric and (not is_category or is_metric):
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce").fillna(0).astype(float)
            is_all_int = (df_numeric[col] % 1 == 0).all()
            if is_all_int:
                df_display[col] = df_numeric[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
            else:
                df_display[col] = df_numeric[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
        elif "date" in str(col).lower() or "time" in str(col).lower():
            try:
                df_display[col] = pd.to_datetime(df_numeric[col]).dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    return df_numeric, df_display


def plot_smart_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str, key: str):
    is_time = any(k in x_col.lower() for k in ["month","year","date","day","week"])
    xaxis = {"type": None} if is_time else {"type": "category", "categoryorder": "total descending"}

    if len(y_cols) == 2:
        v1, v2 = df[y_cols[0]].abs().max(), df[y_cols[1]].abs().max()
        qty_val = any(k in y_cols[0].lower() for k in ["qty","unit"]) and any(k in y_cols[1].lower() for k in ["rev","val","price","sale"])
        val_qty = any(k in y_cols[1].lower() for k in ["qty","unit"]) and any(k in y_cols[0].lower() for k in ["rev","val","price","sale"])

        if qty_val or val_qty or (v1 > 0 and v2 > 0 and (v1/v2 > 10 or v2/v1 > 10)):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            p, s = (y_cols[0], y_cols[1]) if v1 < v2 else (y_cols[1], y_cols[0])
            fig.add_trace(go.Bar(x=df[x_col], y=df[p], name=p, marker_color="#636EFA"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df[x_col], y=df[s], name=s, marker_color="#EF553B", mode="lines+markers"), secondary_y=True)
            fig.update_layout(title=title, template="plotly_dark", hovermode="x unified", xaxis=xaxis,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text=p, secondary_y=False)
            fig.update_yaxes(title_text=s, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True, key=key)
            return

    fig = px.bar(df, x=x_col, y=y_cols, barmode="group", template="plotly_dark", title=title)
    fig.update_layout(xaxis=xaxis)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE KPIs  (cached 30 min)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def get_executive_kpis() -> dict:
    kpis = {"internal_sales": 0, "market_sales": 0, "top_brick": "N/A", "doc_count": 0}
    try:
        r = run_sql_query('SELECT SUM(CAST("product_quantity" AS NUMERIC)) as total FROM "invoice_details"')
        if isinstance(r, list) and r: kpis["internal_sales"] = r[0].get("total") or 0
        r = run_sql_query('SELECT SUM("unit") as total FROM "ims_sale"')
        if isinstance(r, list) and r: kpis["market_sales"] = r[0].get("total") or 0
        r = run_sql_query('''
            SELECT b.name, SUM(CAST(id.product_quantity AS NUMERIC)) as total
            FROM ims_brick b
            JOIN customer_details cd ON b.id = cd.ims_brick_id
            JOIN invoice inv ON cd.customer_id = inv.cust_id
            JOIN invoice_details id ON inv.id = id.invoice_id
            GROUP BY b.name ORDER BY total DESC LIMIT 1
        ''')
        if isinstance(r, list) and r: kpis["top_brick"] = r[0].get("name", "N/A")
        r = run_sql_query('SELECT COUNT(*) as total FROM "doctors"')
        if isinstance(r, list) and r: kpis["doc_count"] = r[0].get("total") or 0
    except Exception as exc:
        print(f"[KPI Error] {exc}")
    return kpis


@st.cache_data(ttl=3600)
def get_globe_data_cached(show_sales: bool):
    import psycopg2
    from db import _CLEAN_URL
    try:
        conn = psycopg2.connect(_CLEAN_URL, connect_timeout=10)
    except psycopg2.OperationalError as exc:
        raise RuntimeError(f"Database connection timed out. Please try again in a moment. ({exc})") from exc
    if show_sales:
        h_sql = """SELECT hc.latitude::float, hc.longitude::float, hc.name,
                          'Health Centre' as type,
                          SUM(CAST(ms.total_amount AS NUMERIC)) as sales,
                          COALESCE(hc.address, hc.name, '') as address
                   FROM healthcentres hc
                   LEFT JOIN master_sale ms ON ms.customer_name ILIKE hc.name
                   WHERE hc.latitude IS NOT NULL
                   GROUP BY 1,2,3,4,6"""
        c_sql = """SELECT c.latitude::float, c.longitude::float, c.name,
                          'Customer' as type,
                          SUM(CAST(ms.total_amount AS NUMERIC)) as sales,
                          COALESCE(ib.name,'') as brick_name, '' as address
                   FROM customers c
                   LEFT JOIN master_sale ms ON ms.customer_name ILIKE c.name
                   LEFT JOIN customer_details cd ON cd.customer_id = c.id
                   LEFT JOIN ims_brick ib ON ib.id = cd.ims_brick_id
                   WHERE c.latitude IS NOT NULL
                   GROUP BY 1,2,3,4,6,7"""
    else:
        h_sql = "SELECT latitude::float, longitude::float, name, 'Health Centre' as type, 0 as sales, COALESCE(address,name,'') as address FROM healthcentres WHERE latitude IS NOT NULL"
        c_sql = "SELECT c.latitude::float, c.longitude::float, c.name, 'Customer' as type, 0 as sales, COALESCE(ib.name,'') as brick_name, '' as address FROM customers c LEFT JOIN customer_details cd ON cd.customer_id=c.id LEFT JOIN ims_brick ib ON ib.id=cd.ims_brick_id WHERE c.latitude IS NOT NULL"

    h_df = pd.read_sql(h_sql, conn)
    c_df = pd.read_sql(c_sql, conn)

    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='doctors'")
        cols = [r[0].lower() for r in cur.fetchall()]
    link_col = next((c for c in ["customersid","customerid"] if c in cols), None)
    if link_col:
        d_df = pd.read_sql(
            f'SELECT c.latitude::float, c.longitude::float, d.name, \'Doctor\' as type, 0 as sales, \'\' as address, \'\' as brick_name FROM doctors d JOIN customers c ON d."{link_col}"=c.id WHERE c.latitude IS NOT NULL LIMIT 80',
            conn,
        )
    else:
        d_df = pd.DataFrame(columns=["latitude","longitude","name","type","sales","address","brick_name"])

    conn.close()
    return h_df, c_df, d_df


# ═══════════════════════════════════════════════════════════════════════════
# MAP / LOCATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def is_map_intent(text: str) -> bool:
    keywords = ["location","map","dikhao map","map pr","map par","nakshe","kahan hai",
                "kahan ha","show on map","locate","address","coordinates","gps",
                "kahan hain","location show"]
    return any(k in text.lower() for k in keywords)


def extract_entity_name(prompt: str) -> str:
    noise = ["ki location dikhao","ka location dikhao","location dikhao","location show kro",
             "ko map par dikhao","ko map pr dikhao","map par dikhao","map pr dikhao",
             "show on map","locate karo","kahan hai","kahan ha","kahan hain",
             "location","map","dikhao","dikha","show","locate","address","coordinates","gps"]
    cleaned = prompt.strip()
    for n in sorted(noise, key=len, reverse=True):
        cleaned = re.sub(n, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(ki|ka|ke|mujhe|mujhay|is|iss)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+(ki|ka|ke|pr|par)$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(".,?!'\"").strip()
    if cleaned and cleaned.lower() not in ["none","","dr","doctor"]:
        return cleaned
    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"user","content":f"Extract ONLY the doctor or clinic name from: '{prompt}'. Return ONLY the name."}],
            timeout=8.0,
        )
        name = resp.choices[0].message.content.strip().strip("\"'")
        if name and name.lower() not in ["none","null","","n/a"]:
            return name
    except Exception:
        pass
    words = [w for w in prompt.split() if len(w) > 2 and (w[0].isupper() or w.isupper())]
    return " ".join(words) if words else prompt


def extract_multiple_entities(prompt: str) -> list:
    clean = re.sub(r"\s+(teeno|dono|sab|ke sath|k sath)\s+", " ", prompt, flags=re.IGNORECASE)
    parts = re.split(r"\s+(?:or|and|aur)\s+|\s*[+&/,،]\s*", clean, flags=re.IGNORECASE)
    names = [extract_entity_name(p.strip()) for p in parts]
    return [n for n in names if n and n.lower() not in ["none","","null"]] or [extract_entity_name(prompt)]


def fetch_location_for_entity(entity_name: str) -> list:
    import psycopg2
    from db import _CLEAN_URL
    results = []
    try:
        conn = psycopg2.connect(_CLEAN_URL, connect_timeout=8)
    except psycopg2.OperationalError as exc:
        print(f"[Location DB Timeout] {exc}")
        return results
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for table, type_label in [
                ("customers",    "Customer"),
                ("healthcentres","Health Centre"),
            ]:
                cur.execute(
                    f'SELECT name, latitude::float, longitude::float, %s as entity_type FROM "{table}" WHERE name ILIKE %s AND latitude IS NOT NULL LIMIT 3',
                    (type_label, f"%{entity_name}%"),
                )
                rows = cur.fetchall()
                for row in rows:
                    addr = reverse_geocode(row["latitude"], row["longitude"])
                    results.append({**dict(row), "address": addr, "pin_color": "red" if type_label == "Customer" else "blue"})
    except Exception as e:
        print(f"[Location Error] {e}")
    finally:
        conn.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MESSAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def submit_question(q: str):
    st.session_state.prompt_trigger = q


def delete_message(msg_id: str):
    msgs = st.session_state.messages
    target = next((i for i, m in enumerate(msgs) if m.get("msg_id") == msg_id), -1)
    if target == -1:
        return
    role = msgs[target]["role"]
    if role == "assistant" and target > 0 and msgs[target - 1]["role"] == "user":
        msgs.pop(target); msgs.pop(target - 1)
    elif role == "user" and target < len(msgs) - 1 and msgs[target + 1]["role"] == "assistant":
        msgs.pop(target + 1); msgs.pop(target)
    else:
        msgs.pop(target)
    save_session(st.session_state.current_session, msgs)
    st.rerun()


def render_map(map_rows: list, key: str):
    map_df = pd.DataFrame(map_rows)
    try:
        import folium
        from streamlit_folium import st_folium
        vlat, vlng = map_df["latitude"].mean(), map_df["longitude"].mean()
        m = folium.Map(
            location=[vlat, vlng], zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
        )
        for _, row in map_df.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(
                    f"<b>{row['name']}</b><br>Type: {row.get('entity_type','')}<br>📍 {row.get('address','')}",
                    max_width=300,
                ),
                tooltip=row["name"],
                icon=folium.Icon(color=row.get("pin_color", "blue"), icon="info-sign"),
            ).add_to(m)
        st_folium(m, width=700, height=400, key=key)
    except Exception:
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

for key, default in [
    ("messages",        []),
    ("current_session", f"New_Session_{int(pd.Timestamp.now().timestamp())}"),
    ("prompt_trigger",  None),
    ("username",        None),
    ("conv_history",    []),   # LLM conversation history for follow-ups
    ("globe_loaded",    False),  # lazy-load flag for the Intelligence Globe tab
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════════════════════════════

if not st.session_state.username:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);
        background-attachment: fixed;
    }
    .login-card {
        background: rgba(30,41,59,0.7); backdrop-filter: blur(12px);
        border-radius: 20px; padding: 40px;
        box-shadow: 0 15px 35px rgba(0,0,0,.4);
        border: 1px solid rgba(255,255,255,.1);
        max-width: 450px; margin: 50px auto; text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg,#6366f1,#a855f7); color: white;
        border: none; border-radius: 10px; padding: 12px; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("<h1 style='color:white;font-family:Outfit,sans-serif;font-size:2.5rem'>👤</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;font-family:Outfit,sans-serif'>Intelligence Login</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8'>Pharma Insights & Strategy Portal</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        user_in = st.text_input("Email / Username", placeholder="e.g. admin@pharma.com")
        pass_in = st.text_input("Access Key", type="password", placeholder="••••••••")
        st.write("")
        submitted = st.form_submit_button("SIGN IN TO DASHBOARD", use_container_width=True)

    if submitted:
        if not user_in.strip():
            st.error("Please enter a username.")
        elif verify_login(user_in.strip(), pass_in):   # ← uses parameterized db.verify_login
            st.session_state.username = user_in.strip()
            st.rerun()
        else:
            st.error("Authentication failed. Check your credentials.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN THEME
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
.main { background:#0f172a; color:#f8fafc; font-family:'Inter',sans-serif; }
h1,h2,h3,.stButton { font-family:'Outfit',sans-serif; }
[data-testid="stSidebar"] { background-color:#1e293b!important; border-right:1px solid rgba(255,255,255,.05); }
.stChatMessage { border-radius:16px!important; padding:20px!important; margin-bottom:12px!important; border:1px solid rgba(255,255,255,.05)!important; }
[data-testid="stChatMessage-user"] { background:rgba(99,102,241,.1)!important; border-left:4px solid #6366f1!important; }
[data-testid="stChatMessage-assistant"] { background:rgba(30,41,59,.6)!important; border-right:4px solid #a855f7!important; }
.stDataFrame { border-radius:12px; border:1px solid rgba(255,255,255,.05); }
.stMetric { background:rgba(255,255,255,.03); padding:15px; border-radius:12px; border:1px solid rgba(255,255,255,.05); }
::-webkit-scrollbar{width:8px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:#334155;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:white;margin-bottom:0'>💊 Pharma Intel Agent</h1>", unsafe_allow_html=True)
st.caption("Strategic Decision Support · RAG Memory · Self-Correcting SQL")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header(f"🧑‍💼 {st.session_state.username}")
    if st.button("🚪 Logout"):
        for k in ["username","messages","current_session","conv_history"]:
            st.session_state[k] = None if k == "username" else ([] if "messages" in k or "history" in k else st.session_state[k])
        st.rerun()

    st.divider()
    st.header("📂 Sessions")

    chat_files = [
        f for f in os.listdir(get_chats_dir())
        if f.endswith(".json") and f not in ["query_cache.json","users.json"]
    ]

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.session_state.current_session = f"New_Session_{int(pd.Timestamp.now().timestamp())}"
        st.rerun()

    if chat_files:
        sel = st.selectbox("Past Conversations", ["Select..."] + chat_files)
        if sel != "Select...":
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📂 Load"):
                    st.session_state.messages = load_session(sel)
                    st.session_state.conv_history = []
                    st.session_state.current_session = sel.replace(".json", "")
                    st.rerun()
            with c2:
                if st.button("🗑️ Delete"):
                    title = sel.replace(".json", "")
                    fp = os.path.join(get_chats_dir(), sel)
                    if os.path.exists(fp): os.remove(fp)
                    delete_db_chat(title)
                    st.session_state.messages = []
                    st.session_state.current_session = f"New_Session_{int(pd.Timestamp.now().timestamp())}"
                    st.toast(f"Deleted {sel}")
                    st.rerun()

    st.divider()
    with st.expander("📊 Data Health"):
        st.success("IMS Market Sales: live")
        st.success("Internal Sales: live")
        st.success("Doctors: live")
        st.warning("⚠️ Orders: empty table")

    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.conv_history = []


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

user_input = st.chat_input("Ask about your pharma data...")
prompt = user_input or st.session_state.prompt_trigger

tab1, tab2 = st.tabs(["💬 AI Chat Agent", "🌍 Intelligence Globe"])

# ─────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────
with tab1:
    st.caption("Ask questions in English or Roman Urdu")
    st.divider()

    # Welcome state — shown only when no messages exist yet
    if not st.session_state.messages:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.12) 100%);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 16px;
            padding: 32px 36px;
            margin-bottom: 24px;
            text-align: center;
        ">
            <div style="font-size: 3rem; margin-bottom: 8px;">💊</div>
            <h2 style="color: #f8fafc; font-family: Outfit, sans-serif; margin: 0 0 8px 0;">
                Welcome to Pharma Intelligence
            </h2>
            <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
                Ask questions in <strong style="color:#a5b4fc">English</strong> or
                <strong style="color:#a5b4fc">Roman Urdu</strong> — your data answers instantly.
                <br>Data loads only when you ask, keeping the app fast.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("#### 💡 Try a sample report to get started:")
        starters = random.sample([
            "Compare top 5 bricks by internal units vs market units",
            "Show me top 5 Category A doctors",
            "Which 3 products have the highest invoice quantity?",
            "Compare internal sales vs market sales in F.B.AREA",
            "Which brick has the highest internal units sold?",
            "List top 5 doctors by visit count in doctor_plan",
            "Show internal sales trend for F.B.AREA region",
            "Which Team has the highest target vs achievement?",
            "What is the market share of Karachi brick?",
        ], 4)
        cols = st.columns(2)
        for i, s in enumerate(starters):
            with cols[i % 2]:
                st.button(s, key=f"starter_{i}", on_click=submit_question, args=(s,))

    # Render existing messages
    for idx, message in enumerate(st.session_state.messages):
        # Determine avatar: standard for AI, Globe for web search
        avatar = "🌐" if message.get("is_web") else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            m_id = message.get("msg_id", f"static_{idx}")

            # Re-render map from history
            if message.get("map_data"):
                render_map(message["map_data"], key=f"hist_map_{idx}")

            # Re-render table + chart from history
            if "data" in message and message["data"] is not None:
                df_raw = pd.DataFrame(message["data"])
                df_num, df_disp = smart_format_dataframe(df_raw)

                h1, h2, _ = st.columns([1, 1, 5])
                with h1:
                    if st.button("🗑️", key=f"del_{m_id}", help="Delete"):
                        delete_message(m_id)
                with h2:
                    csv = df_disp.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 CSV", csv, f"data_{idx}.csv", key=f"dl_{idx}")

                st.dataframe(df_disp, use_container_width=True)

                # Auto-map if coords present
                if "latitude" in df_raw.columns and "longitude" in df_raw.columns:
                    coord_df = df_raw.dropna(subset=["latitude","longitude"])
                    if not coord_df.empty:
                        with st.expander("🗺️ Map", expanded=True):
                            render_map(coord_df.to_dict("records"), key=f"sql_map_{idx}")

                # SQL expander
                if message.get("sql"):
                    with st.expander("🔍 View SQL"):
                        st.code(message["sql"], language="sql")

                if message.get("insight"):
                    st.info(f"💡 **AI Insights:**\n{message['insight']}")

                # Charts
                split_meta = message.get("split_charts_metadata")
                if split_meta:
                    for i, cat in enumerate(df_num[split_meta["group_col"]].unique()[:5]):
                        subset = df_num[df_num[split_meta["group_col"]] == cat].copy()
                        plot_smart_chart(subset, split_meta["x_axis_col"], split_meta["y_metrics"], f"📊 {cat}", f"ch_{idx}_{i}")
                elif message.get("chart_data"):
                    x_col, y_cols = message["chart_data"]
                    valid_y = [c for c in y_cols if c in df_num.columns]
                    if valid_y:
                        plot_smart_chart(df_num, x_col, valid_y, f"Trends: {', '.join(valid_y)}", f"ch_{idx}")

            # Follow-up buttons (last message only)
            if message.get("follow_ups") and idx == len(st.session_state.messages) - 1:
                st.write("---")
                st.write("🔍 **Suggested Follow-ups:**")
                fcols = st.columns(len(message["follow_ups"]))
                for fi, ft in enumerate(message["follow_ups"]):
                    with fcols[fi]:
                        if st.button(ft, key=f"f_{idx}_{fi}"):
                            submit_question(ft)
                            st.rerun()

    # ── Process new prompt ───────────────────────────────────────────────
    if prompt:
        st.session_state.prompt_trigger = None
        msg_id = f"msg_{int(pd.Timestamp.now().timestamp()*1000)}"

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "msg_id": msg_id + "_q"})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            # ── MAP INTENT ───────────────────────────────────────────────
            if is_map_intent(prompt):
                entities = extract_multiple_entities(prompt)
                all_locs = []
                try:
                    with st.spinner(f"📍 Looking up location(s) for: {', '.join(entities)}…"):
                        for ent in entities:
                            all_locs.extend(fetch_location_for_entity(ent))
                except Exception as loc_err:
                    st.warning(f"⚠️ Location lookup failed: {loc_err}")

                if all_locs:
                    reply = f"📍 Found **{len(all_locs)}** location(s) for: {', '.join(entities)}"
                    st.markdown(reply)
                    render_map(all_locs, key=f"new_map_{msg_id}")
                    st.session_state.messages.append({
                        "role": "assistant", "content": reply,
                        "map_data": all_locs, "msg_id": msg_id,
                    })
                else:
                    reply = f"❌ No coordinates found for: {', '.join(entities)}"
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply, "msg_id": msg_id})

            # ── SQL / DATA INTENT ────────────────────────────────────────
            else:
                with st.spinner("🔍 Generating SQL..."):
                    result = generate_and_run_sql(
                        prompt,
                        history=st.session_state.conv_history,
                    )

                is_conv = result.get("is_conversational", False)
                is_web  = result.get("is_web", False)
                
                if result["retries"] > 1:
                    st.caption(f"⚠️ Self-corrected after {result['retries']} attempt(s)")

                if result["error"]:
                    err_msg = f"❌ **Query failed:** {result['error']}"
                    if result["sql"]:
                        err_msg += f"\n\n```sql\n{result['sql']}\n```"
                    st.markdown(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg, "msg_id": msg_id})

                else:
                    rows = result["results"]
                    sql  = result["sql"]

                    # Cache successful query
                    save_to_query_cache(prompt, sql)

                    df_raw = pd.DataFrame(rows) if rows else pd.DataFrame()
                    df_num, df_disp = smart_format_dataframe(df_raw)

                    # ── STREAMING SUMMARY ─────────────────────────────────
                    # Double check if web intent exists even if flag is missing (robustness)
                    web_keywords = ["google", "search", "latest", "news", "internet", "web"]
                    force_web = any(k in prompt.lower() for k in web_keywords)
                    
                    avatar = "🌐" if (is_web or force_web) else None
                    with st.chat_message("assistant", avatar=avatar):
                        full_answer = st.write_stream(
                            summarise_results_stream(prompt, rows[:50])   # cap rows for token safety
                        )

                    # Show table (Skip for web-search/conversational results to avoid redundancy)
                    is_conv = result.get("is_conversational", False)
                    if not df_disp.empty and not is_conv:
                        h1, h2, _ = st.columns([1, 1, 5])
                        with h1:
                            if st.button("🗑️", key=f"del_new_{msg_id}"):
                                delete_message(msg_id)
                        with h2:
                            csv = df_disp.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 CSV", csv, f"result_{msg_id}.csv", key=f"dl_new_{msg_id}")
                        st.dataframe(df_disp, use_container_width=True)

                    # SQL expander (Only for actual DB queries)
                    if sql:
                        with st.expander("🔍 View SQL"):
                            st.code(sql, language="sql")

                    # Auto-map
                    if "latitude" in df_raw.columns and "longitude" in df_raw.columns:
                        coord_df = df_raw.dropna(subset=["latitude","longitude"])
                        if not coord_df.empty:
                            with st.expander("🗺️ Map of Results", expanded=True):
                                render_map(coord_df.to_dict("records"), key=f"new_sql_map_{msg_id}")

                    # Smart chart
                    chart_data = None
                    if not df_num.empty and len(df_num.columns) >= 2:
                        x_col = df_num.columns[0]
                        blacklist = {"id","uuid","code","rank","row_number"}
                        y_cols = [
                            c for c in df_num.columns[1:]
                            if pd.api.types.is_numeric_dtype(df_num[c])
                            and c.lower() not in blacklist
                        ]
                        if y_cols:
                            chart_data = [x_col, y_cols]
                            plot_smart_chart(df_num, x_col, y_cols, f"📊 {prompt[:50]}", f"ch_new_{msg_id}")

                    # Follow-ups
                    with st.spinner("Generating follow-up suggestions..."):
                        follow_ups = suggest_followups(prompt, rows[:20])

                    if follow_ups:
                        st.write("---")
                        st.write("🔍 **Suggested Follow-ups:**")
                        fcols = st.columns(len(follow_ups))
                        for fi, ft in enumerate(follow_ups):
                            with fcols[fi]:
                                if st.button(ft, key=f"fu_new_{msg_id}_{fi}"):
                                    submit_question(ft)
                                    st.rerun()

                    # Persist message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "data": df_raw if not df_raw.empty else None,
                        "sql": sql,
                        "chart_data": chart_data,
                        "follow_ups": follow_ups,
                        "msg_id": msg_id,
                        "is_web": is_web,
                    })

                    # Update conversation history for next turn
                    st.session_state.conv_history.append({"role": "user",      "content": prompt})
                    st.session_state.conv_history.append({"role": "assistant",  "content": full_answer})
                    if len(st.session_state.conv_history) > 12:
                        st.session_state.conv_history = st.session_state.conv_history[-12:]

        # Auto-save session
        new_title = save_session(st.session_state.current_session, st.session_state.messages)
        if new_title != st.session_state.current_session:
            st.session_state.current_session = new_title


# ─────────────────────────────────────────────
# TAB 2 — GLOBE
# ─────────────────────────────────────────────
with tab2:
    st.subheader("🌍 Field Intelligence Globe (Satellite)")

    # ── Lazy-load: only fetch data when the user explicitly requests it ──
    if not st.session_state.globe_loaded:
        st.markdown("""
        <div style="
            background: rgba(30,41,59,0.6);
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 14px;
            padding: 36px;
            text-align: center;
            margin-top: 16px;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🗺️</div>
            <h3 style="color: #f8fafc; font-family: Outfit, sans-serif; margin: 0 0 8px 0;">
                Field Intelligence Map
            </h3>
            <p style="color: #94a3b8; margin: 0 0 20px 0; font-size: 0.9rem;">
                Visualise health centres, customers, and doctors on a satellite map.<br>
                Map data is fetched on demand to keep the app fast.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("🚀 Load Globe Data", use_container_width=True):
                st.session_state.globe_loaded = True
                st.rerun()

    else:
        show_sales = st.checkbox("💰 Show Sales Overlay", value=False)

        col_reload, _ = st.columns([1, 5])
        with col_reload:
            if st.button("🔄 Refresh Data"):
                get_globe_data_cached.clear()
                st.rerun()

        try:
            with st.spinner("🌍 Loading field intelligence data…"):
                h_df, c_df, d_df = get_globe_data_cached(show_sales)

            map_data = pd.concat([h_df, c_df, d_df])
            vlat = map_data["latitude"].mean() if not map_data.empty else 24.86
            vlng = map_data["longitude"].mean() if not map_data.empty else 67.00

            import folium
            from streamlit_folium import st_folium

            m = folium.Map(
                location=[vlat, vlng], zoom_start=11,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
            )
            for df, color, label in [(h_df,"red","HC"), (c_df,"orange","Cust")]:
                for _, row in df.iterrows():
                    addr = row.get("address") or row.get("brick_name", "")
                    popup = f"<b>{row['name']}</b><br>{label}<br>📍 {addr}"
                    if show_sales and row["sales"] > 0:
                        popup += f"<br><b style='color:green'>PKR {row['sales']:,.0f}</b>"
                    folium.CircleMarker(
                        [row["latitude"], row["longitude"]],
                        radius=7 if show_sales and row["sales"] > 20000 else 5,
                        popup=folium.Popup(popup, max_width=250),
                        color=color, fill=True,
                    ).add_to(m)

            for _, row in d_df.iterrows():
                folium.CircleMarker([row["latitude"], row["longitude"]], radius=4, popup=f"{row['name']} (Doc)", color="blue", fill=True).add_to(m)

            st_folium(m, width=900, height=500, key=f"globe_{show_sales}")
            st.info("🔴 Health Centres  |  🟠 Customers  |  🔵 Doctors")

            # ── Executive KPIs — loaded alongside globe data ──────────────
            st.divider()
            st.subheader("📊 Executive KPIs")
            try:
                with st.spinner("Loading KPIs…"):
                    kpis = get_executive_kpis()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("📦 Internal Units Sold", f"{int(kpis['internal_sales']):,}" if kpis["internal_sales"] else "N/A")
                k2.metric("🏪 Market Units Sold",   f"{int(kpis['market_sales']):,}"   if kpis["market_sales"]   else "N/A")
                k3.metric("🏆 Top Brick",           kpis["top_brick"])
                k4.metric("👨‍⚕️ Total Doctors",      f"{int(kpis['doc_count']):,}"      if kpis["doc_count"]      else "N/A")
            except Exception as kpi_err:
                st.warning(f"⚠️ KPIs unavailable: {kpi_err}")

        except RuntimeError as e:
            # DB connection timeout — show a friendly message instead of crashing
            st.warning(f"⏳ {e}")
            st.info("The database is taking longer than expected. Please click **Refresh Data** to try again.")
            if st.button("🔄 Try Again"):
                get_globe_data_cached.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Globe error: {e}")
            if st.button("🔄 Retry"):
                st.rerun()
