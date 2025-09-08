# app.py
from flask import Flask, render_template, request, jsonify, render_template_string, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import os, re, json, time, unicodedata, difflib
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
import pandas as pd
import pytz
from dotenv import load_dotenv
from openai import OpenAI
from models import db, User, Tui, Favorite
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate

# ==== (opcional) scraping simple para horarios ====
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# ======================
# Configuración básica
# ======================
load_dotenv()
app = Flask(__name__)

# --- CONFIG DB (PostgreSQL) ---
database_url = os.getenv("DATABASE_URL")
# Render y algunos proveedores antiguos usan 'postgres://' en vez de 'postgresql://'
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Opcional: evita conexiones muertas en despliegues
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

db.init_app(app)

# --- LOGIN ---
login_manager = LoginManager(app)
login_manager.login_view = "register"  # <- antes ponías "login"

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# Para que las rutas /api/* no redirijan en 302, sino devuelvan 401 (JSON)
@login_manager.unauthorized_handler
def unauthorized():
    from flask import request, jsonify, redirect, url_for
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "AUTH_REQUIRED"}), 401
    # Para páginas normales, redirige a /register
    return redirect(url_for("register"))

# --- MIGRATIONS ---
migrate = Migrate(app, db)


@app.after_request
def add_security_headers(resp):
    resp.headers['Permissions-Policy'] = 'geolocation=(self)'
    return resp

@app.context_processor
def inject_build_version():
    return {"build_version": int(time.time())}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAD_TZ = pytz.timezone("Europe/Madrid")

# ======================
# Utilidades de texto / fuzzy (ligeras)
# ======================
def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\x96","-").replace("–","-").replace("—","-")
    s = s.replace("?","")
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_query(s: str):
    s = norm_text(s)
    toks = re.findall(r"[a-z0-9ñ]+", s)
    stop = {
        "a","al","del","de","la","las","los","el","y","o","u","en","por","para","con","sin",
        "cerca","cercano","cercana","cercanos","cercanas","alrededor","sobre","entre","hacia","desde",
        "que","qué","donde","dónde","una","uno","un","unos","unas","mi","tu","su","me","te","se",
        "lo","le","les","nos","vos","ya","mas","más","esto","eso","aqui","aquí","alli","allí"
    }
    return [t for t in toks if t not in stop and len(t) >= 3]

def build_ngrams(tokens, nmin=1, nmax=4):
    grams = []
    for n in range(nmin, nmax+1):
        for i in range(len(tokens)-n+1):
            grams.append(" ".join(tokens[i:i+n]))
    return grams

def stem_es_light(t: str) -> str:
    t = norm_text(t)
    return re.sub(r"(as|os|a|o|es|s)$","", t)

def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

# ======================
# Utilidades num/geo
# ======================
def _to_float(v):
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(_to_float, [lat1, lon1, lat2, lon2])
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ======================
# Limpieza de datos
# ======================
def clean_url(s):
    if not isinstance(s, str): return ""
    s = s.strip()
    if not s: return ""
    if s.startswith(("http://","https://")) or "." in s:
        return s
    return ""

def clean_phone(s):
    if not isinstance(s, str): s = str(s) if s is not None else ""
    digits = re.sub(r"\D+", "", s)
    return digits if len(digits) >= 7 else ""

def clean_decimal_comma(s):
    if s is None: return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan","none","null"}: return None
    try:
        return float(ss.replace(",", "."))
    except Exception:
        return None

def clean_int(s):
    if s is None: return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan","none","null"}: return None
    try:
        return int(float(ss))
    except Exception:
        return None

def _split_csv_like(s: str):
    return [p.strip() for p in re.split(r"[;,]", s or "") if p.strip()]

def _sanitize_name(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().strip('"').strip("'")
    s = re.sub(r"[�]+", "", s)
    s = re.sub(r"\?{2,}", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _looks_gibberish(name: str) -> bool:
    s = (name or "").strip()
    if len(s) < 3: return True
    if re.fullmatch(r"[\W_¿?¡!.\-\"'()\s]+", s or ""): return True
    if s.count("?") >= max(3, len(s)//2): return True
    return False

# ======================
# Horarios
# ======================
DOW_MAP_ES = {"L":0,"LUN":0,"LUNES":0,"M":1,"MAR":1,"MARTES":1,"X":2,"MIE":2,"MIERCOLES":2,"J":3,"JUE":3,"JUEVES":3,"V":4,"VIE":4,"VIERNES":4,"S":5,"SAB":5,"SABADO":5,"D":6,"DOM":6,"DOMINGO":6}

def _to_min_24h(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    hh = max(0, min(23, hh)); mm = max(0, min(59, mm))
    return hh*60 + mm

def _preclean_horario(s: str) -> str:
    if not isinstance(s, str): return ""
    ss = s.strip()
    if norm_text(ss) in {"nan","none","null"}: return ""
    if ss.startswith("[") and ss.endswith("]"):
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", ss)
        parts = [p[0] or p[1] for p in parts]
        if parts: return " | ".join(parts)
    return ss

def parse_horarios_es(s: str):
    out = {i: [] for i in range(7)}
    if not s or not isinstance(s, str): return out
    s = _preclean_horario(s)
    ss = (s.replace("\x96","-").replace("–","-").replace("—","-")
            .replace(" a "," ").replace("h","").replace("?","").strip())
    bloques = re.split(r"[;|/]+", ss)
    for b in bloques:
        b = b.strip()
        if not b: continue
        m = re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\-\s,]+)\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", b)
        if not m: continue
        dias_txt, o, c = m.groups()
        dias = []
        for frag in re.split(r"[,\s]+", dias_txt.strip().upper()):
            if not frag: continue
            frag_n = norm_text(frag).upper()
            i = DOW_MAP_ES.get(frag_n[:3].upper(), None)
            if i is not None: dias.append(i)
        o_m, c_m = _to_min_24h(o), _to_min_24h(c)
        for d in set(dias):
            if c_m <= o_m:
                out[d].append((o_m, 24*60)); out[d].append((0, c_m))
            else:
                out[d].append((o_m, c_m))
    # merge
    for d in range(7):
        slots = sorted(out[d], key=lambda t: (t[0], t[1]))
        merged = []
        for s0, s1 in slots:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)
        out[d] = [(a, b) for a, b in merged]
    return out

def parse_horarios(s: str):
    return parse_horarios_es(s)

def is_open_now(slots, now_dt=None):
    now = now_dt or datetime.now(MAD_TZ)
    dow = now.weekday()
    mins = now.hour*60 + now.minute
    if not isinstance(slots, dict): return False
    for o,c in slots.get(dow, []):
        if o <= mins <= c:
            return True
    return False

# ======================
# Carga y limpieza CSV
# ======================
CSV_PATH = "data/BBDD_TUI.csv"

def load_and_clean_df(csv_path=CSV_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=",", dtype=str, low_memory=False)
    except Exception as e:
        print("❌ Error al cargar CSV:", e)
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.upper()

    COL_NOMBRE = "NOMBRE_TUI"
    COL_TIPOS  = "TIPOS_TUI"
    COL_CATEG  = "CATEGORIA_TUI"
    COL_DESC   = "DESCRIPCION_TUI" if "DESCRIPCION_TUI" in df.columns else None
    COL_DIR    = "DIRECCION" if "DIRECCION" in df.columns else ("DIRECCION_TUI" if "DIRECCION_TUI" in df.columns else None)
    COL_URL    = "URL" if "URL" in df.columns else "WEBSITE"
    COL_WEB    = "WEBSITE" if "WEBSITE" in df.columns else "URL"
    COL_TEL    = "TELEFONO" if "TELEFONO" in df.columns else None
    COL_HOR    = "HORARIO" if "HORARIO" in df.columns else None
    COL_LAT    = "LATITUD_TUI" if "LATITUD_TUI" in df.columns else None
    COL_LON    = "LONGITUD_TUI" if "LONGITUD_TUI" in df.columns else None
    COL_RATING = "RATING_TUI" if "RATING_TUI" in df.columns else None
    COL_REV    = "TOTAL_VALORACIONES_TUI" if "TOTAL_VALORACIONES_TUI" in df.columns else None
    COL_STATUS = "ESTADO_NEGOCIO" if "ESTADO_NEGOCIO" in df.columns else None
    COL_RES    = "RESERVA_POSIBLE" if "RESERVA_POSIBLE" in df.columns else None
    COL_ACC    = "ACCESIBILIDAD_SILLA_RUEDAS" if "ACCESIBILIDAD_SILLA_RUEDAS" in df.columns else None

    # limpieza básica
    for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC, COL_DIR, COL_URL, COL_WEB, COL_TEL, COL_HOR, COL_STATUS, COL_RES, COL_ACC]:
        if c and c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].where(~df[c].str.match(r"(?i)^\s*(nan|none|null)\s*$"), "")

    if COL_NOMBRE in df.columns:
        df[COL_NOMBRE] = df[COL_NOMBRE].apply(_sanitize_name)
        mask_bad = df[COL_NOMBRE].apply(_looks_gibberish)
        df = df.loc[~mask_bad].copy()

    if COL_WEB in df.columns:
        df[COL_WEB] = df[COL_WEB].apply(clean_url)
    if COL_URL in df.columns:
        df[COL_URL] = df[COL_URL].apply(clean_url)
    if COL_TEL and COL_TEL in df.columns:
        df[COL_TEL] = df[COL_TEL].apply(clean_phone)

    for c in [COL_LAT, COL_LON]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    if COL_RATING and COL_RATING in df.columns:
        rnum = df[COL_RATING].apply(clean_decimal_comma)
        rnum = rnum.where(rnum != 0.0, None)
        df["_RATING"] = pd.to_numeric(rnum, errors="coerce")
    if COL_REV and COL_REV in df.columns:
        df[COL_REV] = df[COL_REV].apply(clean_int)

    if COL_STATUS and COL_STATUS in df.columns:
        st = df[COL_STATUS].astype(str).str.lower()
        mask_open = st.str.contains("abierto", na=True) & ~st.str.contains("cerrado permanentemente|cerrado definitivamente", na=False)
        df = df.loc[mask_open].copy()

    if COL_ACC and COL_ACC in df.columns:
        df[COL_ACC] = df[COL_ACC].str.strip().str.upper().replace({"SI":"SI","SÍ":"SI","YES":"SI","NO":"NO"}).replace("", "NO")
    if COL_RES and COL_RES in df.columns:
        df[COL_RES] = df[COL_RES].str.strip().str.upper().replace({"SI":"SI","SÍ":"SI","YES":"SI","NO":"NO"}).replace("", "NO")

    # horarios -> estructura para "abiertos ahora"
    if COL_HOR and COL_HOR in df.columns:
        df["_CAL_HORAS"] = df[COL_HOR].apply(parse_horarios)
    else:
        df["_CAL_HORAS"] = [{} for _ in range(len(df))]

    # columnas de búsqueda
    present_txt_cols = [c for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC] if c and c in df.columns]
    if present_txt_cols:
        df["_SEARCH_RAW"] = df[present_txt_cols].fillna("").agg(" ".join, axis=1)
        def _tokens(s): return [norm_text(x) for x in _split_csv_like(s)]
        df["_TOK_TYPES"] = df.get(COL_TIPOS, pd.Series("", index=df.index)).apply(_tokens)
        df["_TOK_TYPE_HEAD"] = df.get(COL_TIPOS, pd.Series("", index=df.index)).apply(lambda s: norm_text((s or "").split(",")[0]))
        df["_TOK_CATEG"] = df.get(COL_CATEG, pd.Series("", index=df.index)).fillna("").apply(lambda s: [norm_text(x) for x in _split_csv_like(s)])
        df["_SEARCH"] = (df["_SEARCH_RAW"]).apply(norm_text)
        df["_TOKENS"] = (df["_TOK_TYPES"] + df["_TOK_CATEG"])
    else:
        df["_SEARCH_RAW"] = ""; df["_SEARCH"] = ""; df["_TOKENS"] = [[] for _ in range(len(df))]

    # deduplicar por claves si existen (sin priorizar rating)
    key_cols = []
    if "PLACE_ID" in df.columns: key_cols.append("PLACE_ID")
    if not key_cols:
        if "NOMBRE_TUI" in df.columns and ("DIRECCION" in df.columns or "DIRECCION_TUI" in df.columns):
            dir_col = "DIRECCION" if "DIRECCION" in df.columns else "DIRECCION_TUI"
            key_cols = ["NOMBRE_TUI", dir_col]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")

    return df

df = load_and_clean_df(CSV_PATH)

# Vocabulario dinámico desde el CSV
TYPE_VOCAB = set()
if not df.empty:
    toks_col = df["_TOKENS"] if "_TOKENS" in df.columns else pd.Series([[]]*len(df), index=df.index)
    for toks in toks_col:
        if not isinstance(toks, (list, tuple)): toks = []
        for t in toks:
            TYPE_VOCAB.add(stem_es_light(str(t)))

# ======================
# Áreas (ligero)
# ======================
DISTRICTS = {
    "centro","arganzuela","retiro","salamanca","chamartin","chamartín","tetuan","tetuán","chamberi","chamberí",
    "moncloa - aravaca","moncloa","aravaca","latina","carabanchel","usera","puente de vallecas","moratalaz",
    "ciudad lineal","hortaleza","villaverde","villa de vallecas","vicalvaro","vicálvaro",
    "san blas - canillejas","san blas","canillejas","barajas"
}
AREA_HINTS = {"por","en","cerca","cercanos","cercanas","zona","barrio","distrito","alrededor"}

def extract_area_from_query(q: str):
    tks = tokenize_query(q)
    # bigramas simples para "san blas" / "ciudad lineal"
    for i in range(len(tks)-1):
        two = f"{tks[i]} {tks[i+1]}"
        if two in {"san blas","ciudad lineal"}:
            return two
    # heurística con hints
    joined = tks[:]
    for idx, t in enumerate(joined):
        if t in AREA_HINTS and idx+1 < len(joined):
            cand = joined[idx+1]
            if idx+2 < len(joined):
                two = f"{cand} {joined[idx+2]}"
                if two in DISTRICTS: return two
            if cand in DISTRICTS: return cand
    for t in joined:
        if t in DISTRICTS: return t
    return ""


# ======================
# Búsqueda (sin priorizar rating)
# ======================
def buscar_top(pregunta, max_resultados=20):
    if df.empty:
        return df.head(0)

    q = (pregunta or "").strip()
    q_tokens = tokenize_query(q)
    if not q_tokens:
        # sin query → no ordenamos por rating; devolvemos primeros N (CSV base)
        return df.head(max_resultados)

    area = extract_area_from_query(q)
    base = df.copy()

    # filtrar por área en dirección si aparece
    if area:
        dir_col = "DIRECCION" if "DIRECCION" in base.columns else ("DIRECCION_TUI" if "DIRECCION_TUI" in base.columns else None)
        if dir_col:
            patt = re.escape(norm_text(area))
            addr_norm = base[dir_col].astype(str).apply(norm_text)
            mask = addr_norm.str.contains(patt, na=False)
            base = base.loc[mask] if mask.any() else df.copy()

    # score textual por TIPOS/CATEGORIA/NOMBRE (sin rating)
    def row_score(row):
        s = 0
        hay_tipo = " ".join(row.get("_TOK_TYPES", []))
        hay_cat  = " ".join(row.get("_TOK_CATEG", []))
        name     = str(row.get("NOMBRE_TUI", ""))
        for t in q_tokens:
            if re.search(rf"\b{re.escape(t)}\b", hay_tipo): s += 5
            if re.search(rf"\b{re.escape(t)}\b", hay_cat):  s += 3
            if re.search(rf"\b{re.escape(t)}\b", name, re.I): s += 4
        return s

    base["__score"] = base.apply(row_score, axis=1)
    hits = base.loc[base["__score"] > 0].copy()

    if hits.empty:
        # fuzzy minimal
        def fuzzy_points(row):
            points = 0
            nm = str(row.get("NOMBRE_TUI", ""))
            if any(similar(nm, t) >= 0.75 for t in q_tokens): points += 4
            return points
        base["__score"] = base.apply(fuzzy_points, axis=1)
        hits = base.loc[base["__score"] > 0].copy()

    # ordenar por __score y _RATING si existe
    if not hits.empty:
        if "_RATING" in hits.columns:
            hits["_R_REV"] = pd.to_numeric(hits.get("TOTAL_VALORACIONES_TUI"), errors="coerce").fillna(0)
            hits = hits.sort_values(["__score", "_RATING", "_R_REV"], ascending=[False, False, False])
        else:
            hits = hits.sort_values(["__score", "NOMBRE_TUI"], ascending=[False, True])
        return hits.head(max_resultados)

    return df.head(0)



# ======================
# Filtros y distancia
# ======================
def _aplica_filtros(hits, user_text, now_dt=None):
    if hits is None or hits.empty: return hits
    tx = norm_text(user_text or "")
    out = hits.copy()

    if any(k in tx for k in ["abierto ahora","abiertos ahora","open now"]):
        out = out.loc[out["_CAL_HORAS"].apply(lambda s: is_open_now(s, now_dt=now_dt))]
    if any(k in tx for k in ["accesible","silla de ruedas","wheelchair"]):
        if "ACCESIBILIDAD_SILLA_RUEDAS" in out.columns:
            out = out.loc[out["ACCESIBILIDAD_SILLA_RUEDAS"].str.upper().eq("SI")]
    if any(k in tx for k in ["reserva","reservar","booking","book"]):
        if "RESERVA_POSIBLE" in out.columns:
            out = out.loc[out["RESERVA_POSIBLE"].str.upper().eq("SI")]

    return out if not out.empty else hits

def _sort_by_distance(df_in, user_lat, user_lon):
    if df_in is None or df_in.empty: return df_in
    if user_lat is None or user_lon is None: return df_in
    dists = []
    for _, r in df_in.iterrows():
        d = haversine_km(user_lat, user_lon, r.get("LATITUD_TUI"), r.get("LONGITUD_TUI"))
        dists.append(d if d is not None else 1e9)
    out = df_in.copy()
    out["_DIST_KM"] = dists
    return out.sort_values(["_DIST_KM","NOMBRE_TUI"], ascending=[True, True])

# ======================
# Memoria para plan/mapa
# ======================
LAST_PLAN = {}

def _save_plan_context(conversation_id: str, hits: pd.DataFrame, limit=12):
    if not conversation_id or hits is None or hits.empty: return
    place_ids = []
    if "PLACE_ID" in hits.columns:
        place_ids = hits["PLACE_ID"].dropna().astype(str).str.strip().head(limit).tolist()
    names = hits["NOMBRE_TUI"].astype(str).str.strip().head(limit).tolist() if "NOMBRE_TUI" in hits.columns else []
    LAST_PLAN[conversation_id] = {"place_ids": place_ids, "names": names, "ts": int(time.time())}

def _load_plan_context(conversation_id: str):
    if not conversation_id: return None
    return LAST_PLAN.get(conversation_id)

CURRENT_PLACE_IDS = []
CURRENT_PLACE_TS = 0
def _update_current_place_ids(place_ids):
    global CURRENT_PLACE_IDS, CURRENT_PLACE_TS
    CURRENT_PLACE_IDS = [str(x).strip() for x in (place_ids or []) if str(x).strip()]
    CURRENT_PLACE_TS = int(time.time())

# ======================
# Heurística: intentar raspar horario de la web oficial
# ======================
TIME_RX = re.compile(r"\b(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})\b")
LABEL_RX = re.compile(r"(horario|hours|opening|apertura|abierto)", re.I)

def try_fetch_hours_from_web(row) -> str|None:
    url = (row.get("WEBSITE") or "").strip()
    if not url or url.lower() in {"nan","none","null"}: return None
    if not requests or not BeautifulSoup: return None
    try:
        r = requests.get(url, timeout=6)
        if r.status_code >= 400: return None
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        # buscar secciones con "Horario" o líneas con patrones HH:MM–HH:MM
        if LABEL_RX.search(text):
            # obtener algunas coincidencias
            matches = TIME_RX.findall(text)
            if matches:
                # devolver 1-2 tramos representativos
                spans = []
                for a,b in matches[:2]:
                    a = a.replace(".",":"); b = b.replace(".",":")
                    spans.append(f"{a}-{b}")
                return " / ".join(spans)
        else:
            matches = TIME_RX.findall(text)
            if matches:
                a,b = matches[0]
                return f"{a.replace('.',':')}-{b.replace('.',':')}"
    except Exception:
        return None
    return None

# ======================
# LLM
# ======================
def _llm(messages, model=None, temperature=0.6, max_tokens=1200):
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ======================
# Rutas
# ======================
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages")
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    conversation_id = (data.get("conversation_id") or data.get("conv_id") or "").strip()
    user_latest = [m["content"] for m in messages if m["role"] == "user"][-1]
    start_time  = data.get("start_time") or "09:30"
    start_lat   = data.get("start_lat")
    start_lon   = data.get("start_lon")
    prefer_open = bool(data.get("prefer_open"))
    bootstrap   = bool(data.get("bootstrap"))

    # hora base para “abiertos ahora”
    try:
        hh, mm = map(int, (start_time or "09:30").split(":"))
        base = datetime.now(MAD_TZ)
        now_dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except Exception:
        now_dt = datetime.now(MAD_TZ)

    # 1) Construir candidatos del CSV (NUNCA priorizamos rating)
    if bootstrap:
        hits = df.copy()  # vista general: no priorizar rating
    else:
        hits = buscar_top(user_latest, max_resultados=60)

    if prefer_open:
        user_latest = (user_latest + " abiertos ahora").strip()
    hits = _aplica_filtros(hits, user_latest, now_dt=now_dt)

    if start_lat is not None and start_lon is not None:
        hits = _sort_by_distance(hits, start_lat, start_lon)

    # 2) Compactar contexto para el modelo (y enriquecemos horarios si faltan)
    used_csv = not hits.empty
    top_rows = hits.head(10) if used_csv else hits
    csv_context = []
    for _, r in top_rows.iterrows():
        horario = (r.get("HORARIO","") or "").strip()
        # si no hay horario pero el sitio “encaja” por tipos, intento raspar
        if not horario and (r.get("TIPOS_TUI") or "").strip():
            fetched = try_fetch_hours_from_web(r)
            if fetched: horario = fetched
        csv_context.append({
            "nombre": r.get("NOMBRE_TUI",""),
            "tipo": r.get("TIPOS_TUI",""),
            "direccion": r.get("DIRECCION", r.get("DIRECCION_TUI","")),
            "telefono": r.get("TELEFONO",""),
            "web": r.get("WEBSITE","") or "",
            "mapa": r.get("URL","") or "",
            "horario": horario or "Horarios no disponibles; verifica la web oficial del sitio.",
            # rating solo informativo, NO para priorizar
            "rating": (float(r.get("_RATING")) if pd.notna(r.get("_RATING")) else None),
            "reseñas": (int(r.get("TOTAL_VALORACIONES_TUI")) if str(r.get("TOTAL_VALORACIONES_TUI","")).isdigit() else None),
            "lat": (float(r.get("LATITUD_TUI")) if pd.notna(r.get("LATITUD_TUI")) else None),
            "lon": (float(r.get("LONGITUD_TUI")) if pd.notna(r.get("LONGITUD_TUI")) else None),
        })

    # 3) Publicar place_ids SOLO si hay datos CSV en la respuesta
    if used_csv:
        _save_plan_context(conversation_id, hits, limit=12)
        place_ids = (hits["PLACE_ID"].dropna().astype(str).str.strip().head(20).tolist()
                     if "PLACE_ID" in hits.columns else [])
        _update_current_place_ids(place_ids)
    else:
        _update_current_place_ids([])

    # 4) Prompts
    overview_intro = (
        "Vista general narrativa y conversacional para Madrid:\n"
        "- Recomienda una época del año y comenta brevemente la temporada actual (clima/afluencia).\n"
        "- Overview con 2–3 frases por lugar (prioriza los del CSV; puedes añadir 1–2 sugerencias genéricas si faltan).\n"
        "- Ofrece 2–3 opciones temáticas (cultural, foodies, parques…).\n"
        "- Consejos de ahorro (días gratuitos), mejores horas, alternativas por clima extremo.\n"
        "- Cierra preguntando si quiere: **plan** (hoy/semanal) o **datos** generales.\n"
        "Reglas:\n"
        "- Evita datos sensibles exactos no confirmados (teléfono/precio exacto).\n"
        "- Tono cálido y cercano, estilo guía local.\n"
    )

    system = (
        "Eres un asistente turístico local de Madrid. Enfócate en turismo (planes, moverse, comer, cultura) "
        "en Madrid y alrededores. Usa PRIMERO los lugares del CSV cuando existan; si faltan datos, complementa con "
        "conocimiento general prudente. NO inventes teléfonos/precios exactos.\n\n"
        "Iconografía: usa iconos en planes y listados (p. ej., ⏰, 🎯, 🗺️, 🌐, 📞, 🦽, 🍽️, 🚇, 🚌, 🚶, 💡, ✨, 💶, 🌦️).\n\n"
        "Interpretación de intención (INFIERE por el texto):\n"
        "- Si piden un PLAN (plan/itinerario/agenda/ruta/hoy/mañana/semana):\n"
        "  • Plan para HOY (hora inicio dada), 5–7 paradas, tiempos HH:MM–HH:MM, orden lógico; entre paradas sugiere trayecto (a pie/metro/bus).\n"
        "  • Incluye datos prácticos del CSV por parada (dirección, horario —si falta: 'Horarios no disponibles; verifica la web oficial del sitio'—, web/mapa, accesibilidad si aplica).\n"
        "- Si es INFORMACIÓN o categorías: devuelve un LISTADO útil (≤10) con bullets y datos prácticos del CSV (sin priorizar rating).\n"
        "- Si preguntan CÓMO LLEGAR: explica rutas típicas (metro/bus/a pie), nodos de referencia y tiempos estimados; ofrece alternativas.\n"
        "- Si piden LUGARES MENOS CONCURRIDOS: horarios tranquilos, zonas alternativas, 3–6 sugerencias del CSV si hay.\n"
        "Siempre mantén foco local (Madrid) y tono cercano."
    )

    # En bootstrap, forzamos el “overview introductorio”
    user_block = {
        "modo": "overview" if bootstrap else "normal",
        "instrucciones_overview": overview_intro if bootstrap else None,
        "consulta_usuario": user_latest,
        "hora_inicio": start_time,
        "coords_usuario": f"{start_lat},{start_lon}" if (start_lat is not None and start_lon is not None) else None,
        "contexto_csv": csv_context
    }

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_block, ensure_ascii=False)}
    ]

    reply = _llm(msgs, temperature=0.6, max_tokens=1100)

    if (not used_csv) and (not reply or not reply.strip()):
        reply = ("No tengo datos locales suficientes ahora mismo. Dime tus intereses (museos, parques, bares, gastronomía) "
                 "y te preparo ideas enfocadas en Madrid.")

    return jsonify({"response": reply})

# ===== Vistas auxiliares =====
@app.route("/widget")
def widget():
    return render_template("chat_widget.html")

@app.route("/health")
def health():
    return jsonify({"status":"ok", "rows": int(len(df))})

# --- API: estado de selección para mapa ---
@app.route("/api/current_place_ids", methods=["GET"])
def api_current_place_ids():
    names = []
    try:
        if CURRENT_PLACE_IDS and "PLACE_ID" in df.columns and "NOMBRE_TUI" in df.columns:
            names = (df.loc[df["PLACE_ID"].astype(str).isin(CURRENT_PLACE_IDS), "NOMBRE_TUI"]
                        .astype(str).str.strip().tolist())
    except Exception:
        names = []
    return jsonify({"place_ids": CURRENT_PLACE_IDS, "names": names, "ts": CURRENT_PLACE_TS})

@app.route("/api/current_place_ids/clear", methods=["POST"])
def api_current_place_ids_clear():
    _update_current_place_ids([])
    return jsonify({"ok": True, "cleared": True, "ts": CURRENT_PLACE_TS})

# --- API: puntos por bounding box para mapa ---
# --- API: puntos por bounding box para mapa ---
@app.route("/api/poi")
def api_poi():
    try:
        south = float(request.args.get("south"))
        west  = float(request.args.get("west"))
        north = float(request.args.get("north"))
        east  = float(request.args.get("east"))
    except (TypeError, ValueError):
        return jsonify({"error": "parámetros bbox requeridos: south,west,north,east"}), 400

    zoom  = int(request.args.get("zoom", 12))
    limit = int(request.args.get("limit", 1200 if zoom >= 13 else 700))

    # NUEVO: filtros opcionales por ids/place_ids
    ids_raw  = (request.args.get("ids") or "").strip()
    pids_raw = (request.args.get("place_ids") or "").strip()
    filter_ids  = {x.strip() for x in ids_raw.split(",") if x.strip()}
    filter_pids = {x.strip() for x in pids_raw.split(",") if x.strip()}

    LAT, LON = "LATITUD_TUI", "LONGITUD_TUI"
    if LAT not in df.columns or LON not in df.columns:
        return jsonify([])

    # 1) BBOX (único filtro duro)
    bbox_df = df[df[LAT].between(south, north) & df[LON].between(west, east)].copy()

    # 2) FORZADOS por place_ids o ids (desde TODO el df, no solo bbox)
    forced = pd.DataFrame()
    if filter_ids and "ID" in df.columns:
        forced = pd.concat([forced, df[df["ID"].astype(str).isin(filter_ids)]], ignore_index=True)
    if filter_pids and "PLACE_ID" in df.columns:
        forced = pd.concat([forced, df[df["PLACE_ID"].astype(str).isin(filter_pids)]], ignore_index=True)

    # Deduplicar por claves comunes
    if not forced.empty:
        dedup_keys = [k for k in ["PLACE_ID", "ID", LAT, LON] if k in forced.columns]
        forced = forced.drop_duplicates(subset=dedup_keys, keep="first")

    # 3) Resto (no forzados) → orden por rating y limit
    others = bbox_df.copy()
    if not forced.empty:
        mask_excl = pd.Series(True, index=others.index)
        if "ID" in others.columns and "ID" in forced.columns:
            mask_excl &= ~others["ID"].astype(str).isin(set(forced["ID"].astype(str)))
        if "PLACE_ID" in others.columns and "PLACE_ID" in forced.columns:
            mask_excl &= ~others["PLACE_ID"].astype(str).isin(set(forced["PLACE_ID"].astype(str)))
        others = others[mask_excl]

    if "_RATING" in others.columns:
        others["_R_REV"] = pd.to_numeric(others.get("TOTAL_VALORACIONES_TUI"), errors="coerce").fillna(0)
        others = others.sort_values(by=["_RATING", "_R_REV"], ascending=[False, False])

    if len(others) > limit:
        others = others.head(limit)

    # 4) Resultado final = forced (sin límite) + others (limitados)
    result = pd.concat([forced, others], ignore_index=True)
    dedup_keys = [k for k in ["PLACE_ID", "ID", "NOMBRE_TUI", LAT, LON] if k in result.columns]
    if dedup_keys:
        result = result.drop_duplicates(subset=dedup_keys, keep="first")

    def row_to_obj(r):
        def as_list(s):
            return [x.strip() for x in str(s or "").split(",") if x.strip()]

        rid = r.get("ID")
        if pd.isna(rid):
            rid = abs(hash((r.get("NOMBRE_TUI", ""), r.get(LAT), r.get(LON)))) % (2**31)

        return {
            "id": int(rid),
            "place_id": str(r.get("PLACE_ID", "") or ""),
            "name": r.get("NOMBRE_TUI", ""),
            "description": r.get("DESCRIPCION_TUI", ""),
            "lat": float(r[LAT]),
            "lon": float(r[LON]),
            "address": r.get("DIRECCION", "") or r.get("DIRECCION_TUI", ""),
            "categories": [r.get("CATEGORIA_TUI", "")],
            "subcategories": as_list(r.get("TIPOS_TUI", "")),
            "email": r.get("EMAIL", ""),
            "phone": r.get("TELEFONO", ""),
            "website": r.get("WEBSITE", "") or r.get("CONTENT_URL", ""),
            "gmaps_url": r.get("URL", ""),
            "horario": r.get("HORARIO", ""),
            "precio": r.get("PRECIO", ""),
            "estado_negocio": r.get("ESTADO_NEGOCIO", ""),
            "reserva_posible": r.get("RESERVA_POSIBLE", ""),
            "accesibilidad_silla_ruedas": r.get("ACCESIBILIDAD_SILLA_RUEDAS", ""),
            "rating": r.get("_RATING", None),
            "total_reviews": (int(r["TOTAL_VALORACIONES_TUI"]) 
                              if pd.notna(r.get("TOTAL_VALORACIONES_TUI")) else None),
        }

    return jsonify([row_to_obj(r) for _, r in result.iterrows()])



@app.route("/api/lookup_place_ids", methods=["GET"])
def api_lookup_place_ids():
    ids_raw = request.args.get("place_ids", "") or ""
    ids = [x.strip() for x in ids_raw.split(",") if x and x.strip()]
    if not ids:
        return jsonify([])
    if "PLACE_ID" not in df.columns or "LATITUD_TUI" not in df.columns or "LONGITUD_TUI" not in df.columns:
        return jsonify([])

    sub = df.loc[df["PLACE_ID"].astype(str).isin(ids), ["PLACE_ID","NOMBRE_TUI","LATITUD_TUI","LONGITUD_TUI"]].copy()
    out = []
    for _, r in sub.iterrows():
        try:
            out.append({
                "place_id": str(r["PLACE_ID"]),
                "name": (r.get("NOMBRE_TUI") or ""),
                "lat": float(r["LATITUD_TUI"]),
                "lon": float(r["LONGITUD_TUI"]),
            })
        except Exception:
            continue
    return jsonify(out)

@app.route("/api/conversation_place_ids/<conversation_id>", methods=["GET"])
def api_conversation_place_ids(conversation_id):
    ctx = _load_plan_context(conversation_id)
    if ctx:
        return jsonify({
            "place_ids": ctx.get("place_ids", []),
            "names": ctx.get("names", []),
            "ts": ctx.get("ts", 0)
        })
    return jsonify({"place_ids": [], "names": [], "ts": 0})

# Puertos a usar
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
    

register_tpl = """
<!doctype html>
<title>Registro</title>
<h2>Crear cuenta</h2>
<form method="post">
  <input name="username" placeholder="Usuario" required>
  <input name="email" placeholder="Email" required type="email">
  <input name="password" placeholder="Contraseña" required type="password">
  <button type="submit">Registrarme</button>
</form>
<p><a href="{{ url_for('login') }}">Ya tengo cuenta</a></p>
"""

login_tpl = """
<!doctype html>
<title>Login</title>
<h2>Iniciar sesión</h2>
<form method="post">
  <input name="username" placeholder="Usuario o email" required>
  <input name="password" placeholder="Contraseña" required type="password">
  <button type="submit">Entrar</button>
</form>
<p><a href="{{ url_for('register') }}">Crear cuenta</a></p>
"""

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if User.query.filter((User.username==username)|(User.email==email)).first():
            flash("Usuario o email ya existe")
            return render_template_string(register_tpl)
        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("home") if "home" in [r.rule for r in app.url_map.iter_rules()] else url_for("index"))
    return render_template_string(register_tpl)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter((User.username==u)|(User.email==u)).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("home") if "home" in [r.rule for r in app.url_map.iter_rules()] else url_for("index"))
        flash("Credenciales no válidas")
    return render_template_string(login_tpl)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))



# --- Favoritos API ---

@app.route("/api/favorites", methods=["GET"])
@login_required
def get_favorites():
    # Devolvemos tanto ids de DB como place_ids para que el cliente pueda
    # emparejar correctamente los marcadores generados desde el CSV.
    favs = Favorite.query.filter_by(user_id=current_user.id).all()
    tui_ids = [f.tui_id for f in favs]
    # Join para extraer place_id asociados
    place_ids = []
    try:
        q = (
            db.session.query(Tui.place_id)
            .filter(Tui.id.in_(tui_ids))
            .all()
        )
        place_ids = [p[0] for p in q if p and p[0]]
    except Exception:
        place_ids = []
    return jsonify({"tui_ids": tui_ids, "place_ids": place_ids})

@app.route("/api/favorites/toggle", methods=["POST"])
@login_required
def toggle_favorite():
    data = request.get_json(force=True) or {}
    tui_id = data.get("tui_id")
    place_id = (data.get("place_id") or "").strip() or None
    name = (data.get("name") or "").strip() or None
    lat = data.get("lat")
    lon = data.get("lon")

    # 1) Buscar por id
    place = None
    if tui_id:
        try:
            place = Tui.query.get(int(tui_id))
        except Exception:
            place = None

    # 2) Si no, por place_id (del CSV / DB)
    if not place and place_id:
        place = Tui.query.filter_by(place_id=place_id).first()

    # 3) Último recurso: por nombre y coordenadas (tolerancia pequeña)
    if not place and name and lat is not None and lon is not None:
        try:
            lat = float(lat); lon = float(lon)
            place = (
                Tui.query
                .filter(Tui.NOMBRE_TUI == name)
                .filter(db.func.abs(Tui.LATITUD_TUI - lat) < 1e-5)
                .filter(db.func.abs(Tui.LONGITUD_TUI - lon) < 1e-5)
                .first()
            )
        except Exception:
            place = None

    if not place:
        return jsonify({"ok": False, "error": "TUI no encontrado"}), 404

    fav = Favorite.query.filter_by(user_id=current_user.id, tui_id=place.id).first()
    if fav:
        db.session.delete(fav)
        db.session.commit()
        return jsonify({"ok": True, "favorite": False, "tui_id": place.id, "place_id": place.place_id or ""})
    else:
        db.session.add(Favorite(user_id=current_user.id, tui_id=place.id))
        db.session.commit()
        return jsonify({"ok": True, "favorite": True, "tui_id": place.id, "place_id": place.place_id or ""})

# (Opcional) Home protegida: descomenta el decorador si quieres exigir login para ver la página principal.
# @app.route("/")
# @login_required
# def home():
#     return render_template("index.html")

