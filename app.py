# app.py
"""
Backend Flask para recomendaciones turísticas en Madrid.
- CSV como fuente de lugares (limpieza + búsqueda + horarios).
- Chat concurrente con LLM (plan solo si lo pides explícitamente).
- Estado de place_ids para mapa **por conversación** (no global) y acumulado reciente.
- Favoritos por usuario.
- 👇 Mejoras:
  * El LLM ahora recibe un historial compacto de la conversación y el último catálogo mostrado
    para entender referencias (“la segunda”, “el museo”, etc.) y profundizar en temas ya citados.
"""

# ======================
# Imports
# ======================
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections import deque
from datetime import datetime
from math import radians, sin, cos, asin, sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import threading

import difflib
import pandas as pd
import pytz
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    url_for,
    flash,
)
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_migrate import Migrate
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash

# Modelos propios
from models import db, Favorite, Tui, User, Conversation


# Scraping opcional (horarios)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # (raro en 3.10, pero por si acaso)

MAD_TZ = ZoneInfo("Europe/Madrid")

def _parse_hhmm(hhmm: str):
    try:
        h, m = map(int, hhmm.split(":")[:2])
        if 0 <= h < 24 and 0 <= m < 60:
            return h, m
    except Exception:
        pass
    now = datetime.now(MAD_TZ)
    return now.hour, now.minute

# ======================
# Configuración básica
# ======================
load_dotenv()
app = Flask(__name__)

# --- DB ---
database_url = os.getenv("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

db.init_app(app)

# --- Login ---
login_manager = LoginManager(app)
login_manager.login_view = "register"


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    """Carga de usuario para Flask-Login."""
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None


@login_manager.unauthorized_handler
def unauthorized():
    """401 JSON para /api/*; redirección a /register para vistas HTML."""
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "AUTH_REQUIRED"}), 401
    return redirect(url_for("register"))


# --- Migraciones ---
migrate = Migrate(app, db)


@app.after_request
def add_security_headers(resp):
    """Cabeceras de seguridad mínimas."""
    resp.headers["Permissions-Policy"] = "geolocation=(self)"
    return resp


@app.context_processor
def inject_build_version():
    """Inyecta timestamp para cache-busting en plantillas."""
    return {"build_version": int(time.time())}


# --- OpenAI / Zona horaria ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAD_TZ = pytz.timezone("Europe/Madrid")


# ======================
# Utilidades: texto / fuzzy
# ======================

def _save_conversation_snapshot(user_id: int, conversation_id: str, messages: list, reply: str):
    """
    Guarda (o actualiza) la fila de conversations para (user_id, conversation_id)
    con el snapshot completo de messages + la última reply del asistente.
    """
    if not conversation_id:
        conversation_id = "default"  # respaldo si el front no envía nada

    try:
        conv = Conversation.query.filter_by(
            user_id=user_id, conversation_id=conversation_id
        ).first()

        # Tomamos el historial completo que nos envía el front y añadimos nuestra reply
        snapshot = list(messages) + [{"role": "assistant", "content": reply}]

        # Últimos place_ids mostrados en el mapa para esta conversación
        last_map_ids = _conv_get(conversation_id).get("last_map_ids", [])

        # Un título útil (primer mensaje del usuario)
        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        title = (first_user or "")[:100] if first_user else None

        if conv:
            conv.messages = snapshot
            conv.last_place_ids = last_map_ids
            conv.last_user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
            conv.last_assistant_text = reply
        else:
            conv = Conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                title=title,
                messages=snapshot,
                last_place_ids=last_map_ids,
                last_user_text=next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None),
                last_assistant_text=reply,
            )
            db.session.add(conv)

        db.session.commit()

    except Exception as e:
        app.logger.exception("No se pudo guardar la conversación: %s", e)


def norm_text(s: str) -> str:
    """Normaliza: minúsculas, sin tildes, colapsa espacios, quita ? sueltos."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\x96", "-").replace("–", "-").replace("—", "-")
    s = s.replace("?", "")
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s)


def tokenize_query(s: str) -> List[str]:
    """Tokeniza español evitando stopwords y tokens cortos (<3)."""
    s = norm_text(s)
    toks = re.findall(r"[a-z0-9ñ]+", s)
    stop = {
        "a", "al", "del", "de", "la", "las", "los", "el", "y", "o", "u", "en", "por",
        "para", "con", "sin", "cerca", "cercano", "cercana", "cercanos", "cercanas",
        "alrededor", "sobre", "entre", "hacia", "desde", "que", "qué", "donde", "dónde",
        "una", "uno", "un", "unos", "unas", "mi", "tu", "su", "me", "te", "se", "lo",
        "le", "les", "nos", "vos", "ya", "mas", "más", "esto", "eso", "aqui", "aquí",
        "alli", "allí",
    }
    return [t for t in toks if t not in stop and len(t) >= 3]


def build_ngrams(tokens: Sequence[str], nmin: int = 2, nmax: int = 3) -> List[str]:
    """Genera n-gramas (>= bigramas) para boost de frases exactas."""
    grams: List[str] = []
    for n in range(nmin, nmax + 1):
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams


def similar(a: str, b: str) -> float:
    """Similitud difusa simple."""
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


# ======================
# Utilidades: num/geo
# ======================
def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    """Distancia Haversine en km; None si algún valor no es numérico."""
    lat1, lon1, lat2, lon2 = map(_to_float, [lat1, lon1, lat2, lon2])
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ======================
# Limpieza de datos
# ======================
def clean_url(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    if s.startswith(("http://", "https://")) or "." in s:
        return s
    return ""


def clean_phone(s: Any) -> str:
    s = str(s) if s is not None else ""
    digits = re.sub(r"\D+", "", s)
    return digits if len(digits) >= 7 else ""


def clean_decimal_comma(s: Any) -> Optional[float]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan", "none", "null"}:
        return None
    try:
        return float(ss.replace(",", "."))
    except Exception:
        return None


def clean_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss or norm_text(ss) in {"nan", "none", "null"}:
        return None
    try:
        return int(float(ss))
    except Exception:
        return None


def _split_csv_like(s: str) -> List[str]:
    return [p.strip() for p in re.split(r"[;,]", s or "") if p.strip()]


def _sanitize_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().strip('"').strip("'")
    s = re.sub(r"[�]+", "", s)
    s = re.sub(r"\?{2,}", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _looks_gibberish(name: str) -> bool:
    s = (name or "").strip()
    if len(s) < 3:
        return True
    if re.fullmatch(r"[\W_¿?¡!.\-\"'()\s]+", s or ""):
        return True
    if s.count("?") >= max(3, len(s) // 2):
        return True
    return False


# ======================
# Horarios
# ======================
DOW_MAP_ES = {
    "L": 0, "LUN": 0, "LUNES": 0, "M": 1, "MAR": 1, "MARTES": 1, "X": 2, "MIE": 2,
    "MIERCOLES": 2, "J": 3, "JUE": 3, "JUEVES": 3, "V": 4, "VIE": 4, "VIERNES": 4,
    "S": 5, "SAB": 5, "SABADO": 5, "D": 6, "DOM": 6, "DOMINGO": 6,
}


def _to_min_24h(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return hh * 60 + mm


def _preclean_horario(s: str) -> str:
    if not isinstance(s, str):
        return ""
    ss = s.strip()
    if norm_text(ss) in {"nan", "none", "null"}:
        return ""
    if ss.startswith("[") and ss.endswith("]"):
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", ss)
        parts = [p[0] or p[1] for p in parts]
        if parts:
            return " | ".join(parts)
    return ss


def parse_horarios_es(s: str) -> Dict[int, List[Tuple[int, int]]]:
    """Parsea horarios tipo 'Lun Mar 10:00-18:00 | Sab 11:00-14:00' -> dict por día."""
    out: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(7)}
    if not s or not isinstance(s, str):
        return out
    s = _preclean_horario(s)
    ss = (
        s.replace("\x96", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" a ", " ")
        .replace("h", "")
        .replace("?", "")
        .strip()
    )
    bloques = re.split(r"[;|/]+", ss)
    for b in bloques:
        b = b.strip()
        if not b:
            continue
        m = re.match(
            r"^([A-Za-zÁÉÍÓÚÑáéíóúñ\-\s,]+)\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})",
            b,
        )
        if not m:
            continue
        dias_txt, o, c = m.groups()
        dias: List[int] = []
        for frag in re.split(r"[,\s]+", dias_txt.strip().upper()):
            if not frag:
                continue
            frag_n = norm_text(frag).upper()
            i = DOW_MAP_ES.get(frag_n[:3].upper(), None)
            if i is not None:
                dias.append(i)
        o_m, c_m = _to_min_24h(o), _to_min_24h(c)
        for d in set(dias):
            if c_m <= o_m:
                out[d].append((o_m, 24 * 60))
                out[d].append((0, c_m))
            else:
                out[d].append((o_m, c_m))
    # merge
    for d in range(7):
        slots = sorted(out[d], key=lambda t: (t[0], t[1]))
        merged: List[List[int]] = []
        for s0, s1 in slots:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)
        out[d] = [(a, b) for a, b in merged]
    return out


def parse_horarios(s: str) -> Dict[int, List[Tuple[int, int]]]:
    return parse_horarios_es(s)


def is_open_now(slots: Dict[int, List[Tuple[int, int]]], now_dt: Optional[datetime] = None) -> bool:
    now = now_dt or datetime.now(MAD_TZ)
    dow = now.weekday()
    mins = now.hour * 60 + now.minute
    if not isinstance(slots, dict):
        return False
    for o, c in slots.get(dow, []):
        if o <= mins <= c:
            return True
    return False


# ======================
# Carga CSV + limpieza
# ======================
CSV_PATH = "data/BBDD_TUI.csv"


def load_and_clean_df(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Carga el CSV, normaliza columnas y añade campos auxiliares para búsqueda."""
    try:
        df = pd.read_csv(csv_path, sep=",", dtype=str, low_memory=False)
    except Exception as e:
        print("❌ Error al cargar CSV:", e)
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.upper()

    COL_NOMBRE = "NOMBRE_TUI"
    COL_TIPOS = "TIPOS_TUI"
    COL_CATEG = "CATEGORIA_TUI"
    COL_DESC = "DESCRIPCION_TUI" if "DESCRIPCION_TUI" in df.columns else None
    COL_DIR = (
        "DIRECCION"
        if "DIRECCION" in df.columns
        else ("DIRECCION_TUI" if "DIRECCION_TUI" in df.columns else None)
    )
    COL_URL = "URL" if "URL" in df.columns else "WEBSITE"
    COL_WEB = "WEBSITE" if "WEBSITE" in df.columns else "URL"
    COL_TEL = "TELEFONO" if "TELEFONO" in df.columns else None
    COL_HOR = "HORARIO" if "HORARIO" in df.columns else None
    COL_LAT = "LATITUD_TUI" if "LATITUD_TUI" in df.columns else None
    COL_LON = "LONGITUD_TUI" if "LONGITUD_TUI" in df.columns else None
    COL_RATING = "RATING_TUI" if "RATING_TUI" in df.columns else None
    COL_REV = (
        "TOTAL_VALORACIONES_TUI"
        if "TOTAL_VALORACIONES_TUI" in df.columns
        else None
    )
    COL_STATUS = "ESTADO_NEGOCIO" if "ESTADO_NEGOCIO" in df.columns else None
    COL_RES = "RESERVA_POSIBLE" if "RESERVA_POSIBLE" in df.columns else None
    COL_ACC = (
        "ACCESIBILIDAD_SILLA_RUEDAS"
        if "ACCESIBILIDAD_SILLA_RUEDAS" in df.columns
        else None
    )

    # Limpieza base
    for c in [
        COL_NOMBRE,
        COL_TIPOS,
        COL_CATEG,
        COL_DESC,
        COL_DIR,
        COL_URL,
        COL_WEB,
        COL_TEL,
        COL_HOR,
        COL_STATUS,
        COL_RES,
        COL_ACC,
    ]:
        if c and c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].where(
                ~df[c].str.match(r"(?i)^\s*(nan|none|null)\s*$"), ""
            )

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
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )

    if COL_RATING and COL_RATING in df.columns:
        rnum = df[COL_RATING].apply(clean_decimal_comma)
        rnum = rnum.where(rnum != 0.0, None)
        df["_RATING"] = pd.to_numeric(rnum, errors="coerce")
    if COL_REV and COL_REV in df.columns:
        df[COL_REV] = df[COL_REV].apply(clean_int)

    if COL_STATUS and COL_STATUS in df.columns:
        st = df[COL_STATUS].astype(str).str.lower()
        mask_open = st.str.contains("abierto", na=True) & ~st.str.contains(
            "cerrado permanentemente|cerrado definitivamente", na=False
        )
        df = df.loc[mask_open].copy()

    if COL_ACC and COL_ACC in df.columns:
        df[COL_ACC] = (
            df[COL_ACC]
            .str.strip()
            .str.upper()
            .replace({"SI": "SI", "SÍ": "SI", "YES": "SI", "NO": "NO"})
            .replace("", "NO")
        )
    if COL_RES and COL_RES in df.columns:
        df[COL_RES] = (
            df[COL_RES]
            .str.strip()
            .str.upper()
            .replace({"SI": "SI", "SÍ": "SI", "YES": "SI", "NO": "NO"})
            .replace("", "NO")
        )

    # Horarios preparseados
    if COL_HOR and COL_HOR in df.columns:
        df["_CAL_HORAS"] = df[COL_HOR].apply(parse_horarios)
    else:
        df["_CAL_HORAS"] = [{} for _ in range(len(df))]

    # Campos de búsqueda
    present_txt_cols = [
        c
        for c in [COL_NOMBRE, COL_TIPOS, COL_CATEG, COL_DESC]
        if c and c in df.columns
    ]
    if present_txt_cols:
        df["_SEARCH_RAW"] = df[present_txt_cols].fillna("").agg(" ".join, axis=1)

        def _tokens(s):  # inner helper
            return [norm_text(x) for x in _split_csv_like(s)]

        df["_TOK_TYPES"] = df.get(COL_TIPOS, pd.Series("", index=df.index)).apply(
            _tokens
        )
        df["_TOK_CATEG"] = (
            df.get(COL_CATEG, pd.Series("", index=df.index))
            .fillna("")
            .apply(lambda s: [norm_text(x) for x in _split_csv_like(s)])
        )
        df["_SEARCH"] = (df["_SEARCH_RAW"]).apply(norm_text)
        df["_TOKENS"] = df["_TOK_TYPES"] + df["_TOK_CATEG"]
    else:
        df["_SEARCH_RAW"] = ""
        df["_SEARCH"] = ""
        df["_TOKENS"] = [[] for _ in range(len(df))]

    # Deduplicado
    key_cols: List[str] = []
    if "PLACE_ID" in df.columns:
        key_cols.append("PLACE_ID")
    if not key_cols:
        if "NOMBRE_TUI" in df.columns and (
            "DIRECCION" in df.columns or "DIRECCION_TUI" in df.columns
        ):
            dir_col = "DIRECCION" if "DIRECCION" in df.columns else "DIRECCION_TUI"
            key_cols = ["NOMBRE_TUI", dir_col]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")

    return df


df = load_and_clean_df(CSV_PATH)


# ======================
# Áreas (ligero)
# ======================
DISTRICTS = {
    "centro", "arganzuela", "retiro", "salamanca", "chamartin", "chamartín",
    "tetuan", "tetuán", "chamberi", "chamberí", "moncloa - aravaca", "moncloa",
    "aravaca", "latina", "carabanchel", "usera", "puente de vallecas", "moratalaz",
    "ciudad lineal", "hortaleza", "villaverde", "villa de vallecas", "vicalvaro",
    "vicálvaro", "san blas - canillejas", "san blas", "canillejas", "barajas",
}
AREA_HINTS = {"por", "en", "cerca", "cercanos", "cercanas", "zona", "barrio", "distrito", "alrededor"}


def extract_area_from_query(q: str) -> str:
    """Heurística para detectar distrito/área en la consulta."""
    tks = tokenize_query(q)
    # Bigramas especiales
    for i in range(len(tks) - 1):
        two = f"{tks[i]} {tks[i+1]}"
        if two in {"san blas", "ciudad lineal"}:
            return two
    # Pistas simples
    joined = tks[:]
    for idx, t in enumerate(joined):
        if t in AREA_HINTS and idx + 1 < len(joined):
            cand = joined[idx + 1]
            if idx + 2 < len(joined):
                two = f"{cand} {joined[idx+2]}"
                if two in DISTRICTS:
                    return two
            if cand in DISTRICTS:
                return cand
    for t in joined:
        if t in DISTRICTS:
            return t
    return ""


# ======================
# Búsqueda (rating como desempate) + boost por n-gramas
# ======================
def buscar_top(pregunta: str, max_resultados: int = 60) -> pd.DataFrame:
    """Construye resultados candidatos a partir del CSV con scoring textual."""
    if df.empty:
        return df.head(0)

    q = (pregunta or "").strip()
    q_tokens = tokenize_query(q)
    if not q_tokens:
        return df.head(max_resultados)

    area = extract_area_from_query(q)
    base = df.copy()

    # Filtro por área en dirección (si hay match)
    if area:
        dir_col = (
            "DIRECCION"
            if "DIRECCION" in base.columns
            else ("DIRECCION_TUI" if "DIRECCION_TUI" in base.columns else None)
        )
        if dir_col:
            patt = re.escape(norm_text(area))
            addr_norm = base[dir_col].astype(str).apply(norm_text)
            mask = addr_norm.str.contains(patt, na=False)
            if mask.any():
                base = base.loc[mask]

    ngrams = set(build_ngrams(q_tokens, 2, 3))

    def row_score(row) -> int:
        s = 0
        hay_tipo = " ".join(row.get("_TOK_TYPES", []))
        hay_cat = " ".join(row.get("_TOK_CATEG", []))
        name = str(row.get("NOMBRE_TUI", ""))
        search_all = " ".join([hay_tipo, hay_cat, norm_text(name)])
        for t in q_tokens:
            if re.search(rf"\b{re.escape(t)}\b", hay_tipo):
                s += 5
            if re.search(rf"\b{re.escape(t)}\b", hay_cat):
                s += 3
            if re.search(rf"\b{re.escape(t)}\b", name, re.I):
                s += 4
        for g in ngrams:  # boost frases exactas
            if f" {g} " in f" {search_all} ":
                s += 6
        return s

    base["__score"] = base.apply(row_score, axis=1)
    hits = base.loc[base["__score"] > 0].copy()

    if hits.empty:
        def fuzzy_points(row) -> int:
            points = 0
            nm = str(row.get("NOMBRE_TUI", ""))
            if any(similar(nm, t) >= 0.75 for t in q_tokens):
                points += 4
            return points

        base["__score"] = base.apply(fuzzy_points, axis=1)
        hits = base.loc[base["__score"] > 0].copy()

    if not hits.empty:
        if "_RATING" in hits.columns:
            hits["_R_REV"] = pd.to_numeric(
                hits.get("TOTAL_VALORACIONES_TUI"), errors="coerce"
            ).fillna(0)
            hits = hits.sort_values(
                ["__score", "_RATING", "_R_REV"], ascending=[False, False, False]
            )
        else:
            hits = hits.sort_values(["__score", "NOMBRE_TUI"], ascending=[False, True])
        return hits.head(max_resultados)

    return df.head(0)


# ======================
# Filtros y distancia
# ======================
def _aplica_filtros(hits: pd.DataFrame, user_text: str, now_dt: Optional[datetime] = None) -> pd.DataFrame:
    """Aplica filtros semánticos a partir del texto del usuario."""
    if hits is None or hits.empty:
        return hits
    tx = norm_text(user_text or "")
    out = hits.copy()

    if any(k in tx for k in ["abierto ahora", "abiertos ahora", "open now"]):
        out = out.loc[out["_CAL_HORAS"].apply(lambda s: is_open_now(s, now_dt=now_dt))]
    if any(k in tx for k in ["accesible", "silla de ruedas", "wheelchair"]):
        if "ACCESIBILIDAD_SILLA_RUEDAS" in out.columns:
            out = out.loc[out["ACCESIBILIDAD_SILLA_RUEDAS"].str.upper().eq("SI")]
    if any(k in tx for k in ["reserva", "reservar", "booking", "book"]):
        if "RESERVA_POSIBLE" in out.columns:
            out = out.loc[out["RESERVA_POSIBLE"].str.upper().eq("SI")]

    return out if not out.empty else hits


def _sort_by_distance(df_in: pd.DataFrame, user_lat: Any, user_lon: Any) -> pd.DataFrame:
    """Ordena por distancia Haversine si hay coordenadas de usuario."""
    if df_in is None or df_in.empty:
        return df_in
    if user_lat is None or user_lon is None:
        return df_in
    dists: List[Optional[float]] = []
    for _, r in df_in.iterrows():
        d = haversine_km(user_lat, user_lon, r.get("LATITUD_TUI"), r.get("LONGITUD_TUI"))
        dists.append(d if d is not None else 1e9)
    out = df_in.copy()
    out["_DIST_KM"] = dists
    return out.sort_values(["_DIST_KM", "NOMBRE_TUI"], ascending=[True, True])


def _filter_by_radius_km(df_in: pd.DataFrame, user_lat: Any, user_lon: Any, radius_km: float = 1.5) -> pd.DataFrame:
    """Filtra a un radio en km desde user_lat/lon."""
    if df_in is None or df_in.empty or user_lat is None or user_lon is None:
        return df_in
    dists = []
    for _, r in df_in.iterrows():
        d = haversine_km(user_lat, user_lon, r.get("LATITUD_TUI"), r.get("LONGITUD_TUI"))
        dists.append(d if d is not None else 1e9)
    out = df_in.copy()
    out["_DIST_KM"] = dists
    return out.loc[out["_DIST_KM"] <= radius_km].copy()


def _boost_recent_by_conv(df_in: pd.DataFrame, conversation_id: str, boost: float = 0.6) -> pd.DataFrame:
    """Rerank suave: si el PLACE_ID estuvo recientemente en la misma conversación, sube un poco."""
    if df_in is None or df_in.empty or "PLACE_ID" not in df_in.columns:
        return df_in
    recent = set(_conv_recent_ids(conversation_id))
    out = df_in.copy()
    # base score: rating y reseñas
    if "_RATING" in out.columns:
        out["_R_REV"] = pd.to_numeric(out.get("TOTAL_VALORACIONES_TUI"), errors="coerce").fillna(0)
        base = (out["_RATING"].fillna(0) + (out["_R_REV"] / (out["_R_REV"].max() or 1 + 1e-9))*0.2)
    else:
        base = pd.Series(0, index=out.index)

    # proximidad si existe
    if "_DIST_KM" in out.columns:
        prox = (out["_DIST_KM"].max() - out["_DIST_KM"]) / (out["_DIST_KM"].max() or 1 + 1e-9)
    else:
        prox = 0

    recent_mask = out["PLACE_ID"].astype(str).isin(recent)
    out["__rerank"] = base + prox + (recent_mask.astype(float) * boost)
    out = out.sort_values(["__rerank", "NOMBRE_TUI"], ascending=[False, True])
    return out


# ======================
# Estado (plan/mapa) por conversación + helpers
# ======================
LAST_PLAN: Dict[str, Dict[str, Any]] = {}

# Estado por conversación
CONV_STATE: Dict[str, Dict[str, Any]] = {}
CONV_LOCK = threading.Lock()

def _conv_get(conv_id: str) -> Dict[str, Any]:
    if not conv_id:
        conv_id = ""
    with CONV_LOCK:
        st = CONV_STATE.get(conv_id)
        if not st:
            st = {
                "place_ids_recent": deque(maxlen=60),  # últimos ids vistos en esa conversación
                "last_map_ids": [],
                "last_catalog": [],
                "last_intent": None,
                "ts": int(time.time()),
            }
            CONV_STATE[conv_id] = st
        return st

def _conv_update_last_intent(conv_id: str, intent: str) -> None:
    st = _conv_get(conv_id)
    with CONV_LOCK:
        st["last_intent"] = intent
        st["ts"] = int(time.time())

def _ordered_unique(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq or []:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _conv_publish_map_ids(conv_id: str, place_ids: Iterable[str]) -> None:
    """Publica ids 'actuales' para el mapa de ESTA conversación y acumula en el buffer reciente."""
    st = _conv_get(conv_id)
    clean_ids = [str(x).strip() for x in (place_ids or []) if str(x).strip()]
    clean_ids = _ordered_unique(clean_ids)
    with CONV_LOCK:
        st["last_map_ids"] = list(clean_ids)
        for pid in clean_ids:
            st["place_ids_recent"].append(pid)
        st["ts"] = int(time.time())

def _conv_recent_ids(conv_id: str) -> List[str]:
    st = _conv_get(conv_id)
    with CONV_LOCK:
        return list(dict.fromkeys(list(st["place_ids_recent"])))


def _conv_set_catalog(conv_id: str, rows: List[Dict[str, Any]]) -> None:
    """Guarda el catálogo del último output (sirve para follow-ups)."""
    st = _conv_get(conv_id)
    items = []
    for i, r in enumerate(rows, start=1):
        nombre = r.get("nombre") or r.get("name") or ""
        pid = (r.get("place_id") or "").strip()
        lat = r.get("lat")
        lon = r.get("lon")
        items.append({
            "idx": i,
            "name": nombre,
            "place_id": pid,
            "lat": lat,
            "lon": lon,
            "key": pid or f"{norm_text(nombre)}|{lat or ''}|{lon or ''}",
        })
    with CONV_LOCK:
        st["last_catalog"] = items
        st["ts"] = int(time.time())

def _conv_get_catalog(conv_id: str) -> List[Dict[str, Any]]:
    st = _conv_get(conv_id)
    with CONV_LOCK:
        return list(st.get("last_catalog", []))

def _cleanup_conv_state(ttl_seconds: int = 2 * 24 * 3600):
    now_ts = int(time.time())
    with CONV_LOCK:
        expired = [cid for cid, st in CONV_STATE.items() if now_ts - st.get("ts", now_ts) > ttl_seconds]
        for cid in expired:
            CONV_STATE.pop(cid, None)


# === NUEVO: helpers de memoria conversacional para el LLM ===
def _clip(s: str, max_chars: int = 1400) -> str:
    s = (s or "").strip()
    return (s[: max_chars - 3] + "...") if len(s) > max_chars else s

def _compact_history(messages: List[Dict[str, str]], max_turns: int = 12, max_chars: int = 1800) -> str:
    """
    Devuelve una versión compacta (user/assistant) de los últimos turnos.
    Sin NLP costoso: solo limpia, etiqueta por rol y recorta.
    """
    hist = []
    for m in messages[-max_turns:]:
        role = m.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "U:" if role == "user" else "A:"
        hist.append(f"{prefix} {content}")
    return _clip("\n".join(hist), max_chars)

def _catalog_for_coref(conv_id: str) -> List[Dict[str, Any]]:
    """
    Último catálogo mostrado para resolver referencias tipo 'la segunda', 'el del parque', etc.
    Devuelve estructura ligera: idx, nombre, place_id, lat, lon.
    """
    items = _conv_get_catalog(conv_id)
    out = []
    for it in items[:24]:
        out.append({
            "idx": it.get("idx"),
            "nombre": it.get("name"),
            "place_id": it.get("place_id"),
            "lat": it.get("lat"),
            "lon": it.get("lon"),
        })
    return out


# ======================
# Scraping de horarios (opcional)
# ======================
TIME_RX = re.compile(r"\b(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})\b")
LABEL_RX = re.compile(r"(horario|hours|opening|apertura|abierto)", re.I)


def try_fetch_hours_from_web(row: pd.Series) -> Optional[str]:
    """Intenta extraer franjas horarias básicas de la web oficial (best-effort)."""
    url = (row.get("WEBSITE") or "").strip()
    if not url or url.lower() in {"nan", "none", "null"}:
        return None
    if not requests or not BeautifulSoup:
        return None
    try:
        r = requests.get(url, timeout=6)
        if r.status_code >= 400:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        matches = []
        if LABEL_RX.search(text):
            matches = TIME_RX.findall(text)
            if matches:
                spans = []
                for a, b in matches[:2]:
                    a = a.replace(".", ":")
                    b = b.replace(".", ":")
                    spans.append(f"{a}-{b}")
                return " / ".join(spans)
        else:
            matches = TIME_RX.findall(text)
            if matches:
                a, b = matches[0]
                return f"{a.replace('.',':')}-{b.replace('.',':')}"
    except Exception:
        return None
    return None


# ======================
# LLM (wrapper)
# ======================
def _llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 8000,
) -> str:
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    resp = client.chat.completions.create(
        model=mdl, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


# ======================
# Intención por turno (concurrencia)
# ======================
LAST_STATE: Dict[str, Dict[str, Any]] = {}

PLAN_SYNONYMS = {
    "plan", "itinerario", "ruta", "agenda", "programa", "planning", "cronograma",
    "schedule", "tour", "recorrido", "itinerary", "propuesta", "timeline",
}
INFO_SYNONYMS = {
    "info", "informacion", "información", "ideas", "recomendaciones", "sitios",
    "lugares", "opciones", "listado", "lista", "consejos", "mapa", "detalle", "detalles",
}
FOLLOWUP_MARKERS = {
    "respecto al ultimo", "respecto al último", "sobre lo anterior", "de lo anterior",
    "del ultimo", "del último", "lo de antes", "lo anterior", "siguiendo con", "en lo anterior",
}


def _detect_intent(messages: List[Dict[str, str]], conversation_id: str) -> str:
    """
    Intención SOLO para ESTE turno:
      - 'plan' si el mensaje actual contiene sinónimos de plan.
      - 'info' en caso contrario (incluye follow-ups).
    """
    user_texts = [m["content"] for m in messages if m.get("role") == "user"]
    latest = (user_texts[-1] if user_texts else "").lower()
    norm = norm_text(latest)

    has_plan_kw = any(f" {kw} " in f" {norm} " for kw in PLAN_SYNONYMS)
    has_info_kw = any(f" {kw} " in f" {norm} " for kw in INFO_SYNONYMS)
    is_followup = any(phrase in norm for phrase in FOLLOWUP_MARKERS)

    if has_plan_kw:
        intent = "plan"
    elif has_info_kw:
        intent = "info"
    elif is_followup:
        intent = "info"
    else:
        intent = "info"

    LAST_STATE[conversation_id or ""] = {"last_intent": intent, "ts": int(time.time())}
    return intent


def _target_count_for_intent(intent: str, bootstrap: bool) -> int:
    if bootstrap:
        return 6
    return 6 if (intent or "").lower() == "plan" else 6


def _coherence_score(hits: pd.DataFrame) -> float:
    """Devuelve 0..1 según el máximo __score normalizado."""
    if hits is None or hits.empty:
        return 0.0
    s = hits.get("__score")
    if s is None:
        return 0.0
    mx = float(pd.to_numeric(s, errors="coerce").fillna(0).max())
    return min(1.0, mx / 10.0)

def _parse_selected_idx(text: str) -> List[int]:
    """Extrae [índices] del bloque <pins>...<pins> al final de la respuesta del LLM."""
    try:
        m = re.search(r"<pins>\s*(\{.*?\}|\[.*?\])\s*</pins>", text, re.S)
        if not m:
            return []
        raw = m.group(1)
        data = json.loads(raw)
        if isinstance(data, dict):
            arr = data.get("selected_idx", [])
        else:
            arr = data
        out: List[int] = []
        for x in arr:
            try:
                out.append(int(x))
            except Exception:
                continue
        return [i for i in out if i >= 1]
    except Exception:
        return []

def _idx_to_place_ids(idxs: List[int], catalog: List[Dict[str, Any]]) -> List[str]:
    """Convierte índices 1-based del catálogo reciente en place_ids únicos preservando orden."""
    by_idx = {item.get("idx"): (item.get("place_id") or "").strip() for item in catalog}
    pids = [(by_idx.get(i) or "") for i in idxs]
    pids = [p for p in pids if p]
    seen: set[str] = set()
    uniq: List[str] = []
    for p in pids:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq

def _strip_pins_block(text: str) -> str:
    """Elimina el bloque <pins>...<pins> del texto para no mostrarlo al usuario."""
    return re.sub(r"\s*<pins>\s*(\{.*?\}|\[.*?\])\s*</pins>\s*$", "", text, flags=re.S)


# ======================
# Rutas
# ======================
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint principal del chat (plan/info concurrente + comprensión del historial)."""
    data = request.get_json(force=True) or {}

    # --- Mensajes e identificadores ---
    messages = data.get("messages")
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    conversation_id = (data.get("conversation_id") or data.get("conv_id") or "").strip()
    user_latest = [m["content"] for m in messages if m.get("role") == "user"][-1]

    # --- Hora de Madrid por defecto y 'now_dt' coherente ---
    start_time = data.get("start_time") or datetime.now(MAD_TZ).strftime("%H:%M")
    h, m = _parse_hhmm(start_time)
    now_dt = datetime.now(MAD_TZ).replace(hour=h, minute=m, second=0, microsecond=0)

    # --- Parámetros auxiliares ---
    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")
    prefer_open = bool(data.get("prefer_open"))
    bootstrap = bool(data.get("bootstrap"))

    # --- Intención actual + persistencia simple por conversación ---
    intent = _detect_intent(messages, conversation_id)
    _conv_update_last_intent(conversation_id, intent)

    # Autoactivar "abierto ahora" si piden plan "hoy/ahora" (sin romper lo que venga del front)
    if not prefer_open and intent == "plan":
        q_norm_for_open = norm_text(user_latest)
        if ("hoy" in q_norm_for_open) or ("ahora" in q_norm_for_open):
            prefer_open = True

    # --- 1) Candidatos del CSV ---
    hits = df.copy() if bootstrap else buscar_top(user_latest, max_resultados=60)

    if prefer_open:
        # pista textual para el filtro de horarios
        user_latest = (user_latest + " abiertos ahora").strip()
    hits = _aplica_filtros(hits, user_latest, now_dt=now_dt)

    # --- Proximidad / "cerca de mí" ---
    q_norm = norm_text(user_latest)
    wants_near = any(w in q_norm for w in ["cerca", "alrededor", "cercano", "cercanos", "cercanas"])
    if start_lat is not None and start_lon is not None:
        hits = _sort_by_distance(hits, start_lat, start_lon)
        if wants_near:
            hits = _filter_by_radius_km(hits, start_lat, start_lon, radius_km=1.5)

    # --- Rerank por memoria de conversación ---
    hits = _boost_recent_by_conv(hits, conversation_id, boost=0.6)

    # --- Deduplicado de hits ---
    if hits is not None and not hits.empty:
        if "PLACE_ID" in hits.columns:
            hits = hits.drop_duplicates(subset=["PLACE_ID"], keep="first")
        else:
            keys = [k for k in ["ID", "NOMBRE_TUI", "LATITUD_TUI", "LONGITUD_TUI"] if k in hits.columns]
            if keys:
                hits = hits.drop_duplicates(subset=keys, keep="first")

    # --- 2) Coherencia CSV y selección top-N ---
    coh = _coherence_score(hits)
    used_csv = (not hits.empty) and (coh >= 0.3)

    n_target = _target_count_for_intent(intent, bootstrap)  # ideal 6–6 para ahorrar tokens
    top_rows = hits.head(n_target) if used_csv else hits.head(0)

    # --- 3) Contexto CSV para el LLM ---
    csv_context = []
    _seen_keys = set()

    for _, r in top_rows.iterrows():
        horario = (r.get("HORARIO", "") or "").strip()
        if not horario:
            fetched = try_fetch_hours_from_web(r)
            if fetched:
                horario = fetched

        pid = (str(r.get("PLACE_ID")) if pd.notna(r.get("PLACE_ID")) else "") or ""
        fb_key = f"{norm_text(r.get('NOMBRE_TUI',''))}|{r.get('LATITUD_TUI')}|{r.get('LONGITUD_TUI')}"
        key = pid or fb_key
        if key in _seen_keys:
            continue
        _seen_keys.add(key)

        csv_context.append({
            "place_id": pid,
            "nombre": r.get("NOMBRE_TUI", ""),
            "tipo": r.get("TIPOS_TUI", ""),
            "direccion": r.get("DIRECCION", r.get("DIRECCION_TUI", "")),
            "telefono": r.get("TELEFONO", ""),
            "web": r.get("WEBSITE", "") or "",
            "mapa": r.get("URL", "") or "",
            "horario": horario or "Horarios no disponibles; verifica la web oficial del sitio.",
            "rating": (float(r.get("_RATING")) if pd.notna(r.get("_RATING")) else None),
            "reseñas": (int(r.get("TOTAL_VALORACIONES_TUI")) if str(r.get("TOTAL_VALORACIONES_TUI", "")).isdigit() else None),
            "lat": (float(r.get("LATITUD_TUI")) if pd.notna(r.get("LATITUD_TUI")) else None),
            "lon": (float(r.get("LONGITUD_TUI")) if pd.notna(r.get("LONGITUD_TUI")) else None),
        })

    # --- 4) Catálogo para coreferencias/concurrencia ---
    catalog_rows = [{"nombre": r.get("nombre",""), "place_id": r.get("place_id","") or "", "lat": r.get("lat"), "lon": r.get("lon")} for r in csv_context]
    _conv_set_catalog(conversation_id, catalog_rows)

    # --- 5) Prompt (system) + historial compacto ---
    user_name = current_user.username if current_user and current_user.is_authenticated else ""
    overview_intro = (
        f"Hola **{user_name}** 👋\n"
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

    historial_compacto = _compact_history(messages, max_turns=60, max_chars=12000)
    catalogo_coref = _catalog_for_coref(conversation_id)

    system = (
        "Eres un asistente turístico local de Madrid. Objetivo: responder con precisión y mantener **concurrencia de chat**.\n"
        "- Usa PRIMERO los lugares del CSV si son coherentes con la consulta; si no, responde con conocimiento general sin mencionar el CSV.\n"
        "- No inventes teléfonos ni precios exactos. Puedes dar rangos o recomendaciones prudentes.\n\n"
        "Intención por turno (reglas duras):\n"
        "1) Genera un **PLAN** solo si el mensaje ACTUAL contiene sinónimos de plan (plan, itinerario, ruta, agenda, programa, planning, cronograma, schedule, tour, recorrido, itinerary, propuesta, timeline).\n"
        "2) Si el usuario hace un **follow-up** y NO pide plan explícitamente, responde en **modo INFO** (concurrente) sin crear un plan nuevo.\n"
        "3) Si pide **info general**, responde en modo INFO.\n\n"
        "Comprensión del **contexto conversacional** (precisión):\n"
        "- Recibirás `historial_compacto` y `catalogo_reciente` (ítems con `idx`, `nombre`, `place_id`).\n"
        "- Si la consulta actual apunta a algo mencionado antes (por nombre, por `idx` del listado anterior, o por referencia ambigua tipo “el segundo”), **resuelve la referencia** usando `catalogo_reciente`.\n"
        "- Si hay ambigüedad entre varios ítems, elige el más probable; SOLO si la ambigüedad hace insegura la recomendación, formula **una única** pregunta breve y ofrece **la mejor respuesta provisional**.\n"
        "- Mantén continuidad: respeta preferencias del historial (horarios, presupuesto, zonas, accesibilidad, 'abierto ahora').\n\n"
        "Formato y nivel de detalle (consistencia):\n"
        "- **PLAN**: 5–7 paradas; intervalos HH:MM–HH:MM; orden lógico; trayectos sugeridos (a pie/metro/bus). Cada parada: nombre, breve descripción, dirección, horario (o aviso), web oficial y enlace a mapa; accesibilidad si está en CSV.\n"
        "- **INFO**: listado ≤10 con bullets y el mismo detalle.\n"
        "- Si preguntan CÓMO LLEGAR: rutas (metro/bus/a pie) y tiempos estimados.\n\n"
        "Reglas de datos:\n"
        "- Valida coherencia del CSV; si no encaja, responde útil sin mencionar el CSV.\n"
        "- Iconos cuando ayuden: ⏰ 🎯 🗺️ 🌐 📞 🦽 🍽️ 🚇 🚌 🚶 💡 ✨ 💶 🌦️.\n"
        "➡️ **Al FINAL** incluye un bloque JSON dentro de <pins>...</pins> con los índices del `contexto_csv` usado, p.ej. <pins>{\"selected_idx\":[1,2,3]}</pins>. Si no usaste CSV, devuelve <pins>{\"selected_idx\":[]}</pins>.\n"
    )

    # --- 6) Payload al modelo (con fecha/TZ) ---
    user_block = {
        "modo": "overview" if bootstrap else "normal",
        "intent": intent,
        "instrucciones_overview": overview_intro if bootstrap else None,
        "consulta_usuario": user_latest,
        "hora_inicio": start_time,
        "fecha_hoy": now_dt.strftime("%Y-%m-%d"),
        "tz": "Europe/Madrid",
        "coords_usuario": f"{start_lat},{start_lon}" if (start_lat is not None and start_lon is not None) else None,
        "historial_compacto": historial_compacto,
        "catalogo_reciente": catalogo_coref,
        "contexto_csv": csv_context,
    }

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_block, ensure_ascii=False)},
    ]
    reply = _llm(msgs, temperature=0.6, max_tokens=2000)

    # --- 7) Alineación mapa ↔️ chat en base a <pins> ---
    catalog_recent = _conv_get_catalog(conversation_id)
    selected_idx = _parse_selected_idx(reply)

    if used_csv:
        if not selected_idx:
            # Fallback: detecta numeraciones 1., 2., 3. si el modelo no devolvió <pins>
            found = re.findall(r"(?m)^\s*(\d{1,2})[)\.\-]\s", reply)
            try_idx = [int(x) for x in found if x.isdigit()]
            try_idx = [i for i in try_idx if 1 <= i <= len(catalog_recent)]
            selected_idx = try_idx

        place_ids_for_map = _idx_to_place_ids(selected_idx, catalog_recent) if selected_idx else []
        _conv_publish_map_ids(conversation_id, place_ids_for_map)

        LAST_PLAN[conversation_id] = {
            "place_ids": list(place_ids_for_map),
            "names": [
                it.get("nombre") or it.get("name") or ""
                for it in catalog_recent
                if (it.get("place_id") or "").strip() in place_ids_for_map
            ],
            "ts": int(time.time()),
        }
    else:
        # Sin CSV no publicamos pins aunque el LLM escriba generalidades
        _conv_publish_map_ids(conversation_id, [])

    # --- 8) Limpieza y persistencia ligera ---
    reply_clean = _strip_pins_block(reply)

    if current_user.is_authenticated:
        _save_conversation_snapshot(
            current_user.id,
            conversation_id,
            messages,
            reply_clean
        )

    return jsonify({
        "response": reply_clean,
        "place_ids": _conv_get(conversation_id).get("last_map_ids", []),
        "used_csv": used_csv
    })


@app.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    rows = (Conversation.query
            .filter_by(user_id=current_user.id)
            .order_by(Conversation.updated_at.desc())
            .all())
    out = []
    for r in rows:
        out.append({
            "conversation_id": r.conversation_id,
            "title": r.title,
            "updated_at": r.updated_at.isoformat(),
            "last_user_text": r.last_user_text,
        })
    return jsonify(out)

@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def get_conversation(conv_id):
    r = Conversation.query.filter_by(
        user_id=current_user.id, conversation_id=conv_id
    ).first()
    if not r:
        return jsonify({"error": "not_found"}), 404
    return jsonify({
        "conversation_id": r.conversation_id,
        "title": r.title,
        "messages": r.messages or [],
        "last_place_ids": r.last_place_ids or [],
        "updated_at": r.updated_at.isoformat()
    })

# ===== Vistas auxiliares =====
@app.route("/widget")
def widget():
    return render_template("chat_widget.html")


@app.route("/health")
def health():
    _cleanup_conv_state()
    return jsonify({"status": "ok", "rows": int(len(df))})


# --- API: estado de selección para mapa (por conversación) ---
@app.route("/api/current_place_ids", methods=["GET"])
def api_current_place_ids():
    """Devuelve place_ids actuales + nombres (para una conversación)."""
    conv_id = (request.args.get("conversation_id") or "").strip()
    st = _conv_get(conv_id)
    place_ids = st.get("last_map_ids", [])
    names: List[str] = []
    try:
        if place_ids and "PLACE_ID" in df.columns and "NOMBRE_TUI" in df.columns:
            names = (
                df.loc[df["PLACE_ID"].astype(str).isin(place_ids), "NOMBRE_TUI"]
                .astype(str).str.strip().tolist()
            )
    except Exception:
        names = []
    return jsonify({"place_ids": place_ids, "names": names, "ts": st.get("ts", 0)})


@app.route("/api/current_place_ids/clear", methods=["POST"])
def api_current_place_ids_clear():
    """Limpia place_ids actuales de una conversación."""
    conv_id = (request.args.get("conversation_id") or "").strip()
    _conv_publish_map_ids(conv_id, [])
    return jsonify({"ok": True, "cleared": True, "ts": _conv_get(conv_id).get("ts", 0)})


# --- API: puntos por bounding box para mapa ---
@app.route("/api/poi")
def api_poi():
    """
    Devuelve POIs dentro de un bbox. Modo compacto disponible:
      - ids_only=1   ó  fields=place_ids  -> devuelve solo place_ids únicos.
    Filtros opcionales: ids, place_ids (forzados desde todo el df).
    Extra opcional: conversation_id, user_lat/lon, near=1, radius_km, prefer_open.
    """
    try:
        south = float(request.args.get("south"))
        west = float(request.args.get("west"))
        north = float(request.args.get("north"))
        east = float(request.args.get("east"))
    except (TypeError, ValueError):
        return jsonify({"error": "parámetros bbox requeridos: south,west,north,east"}), 400

    zoom = int(request.args.get("zoom", 12))
    limit = int(request.args.get("limit", 1200 if zoom >= 13 else 700))
    ids_only = str(request.args.get("ids_only", "")).strip().lower() in {"1", "true", "yes"}
    fields = (request.args.get("fields") or "").strip().lower()
    want_place_ids_only = ids_only or fields == "place_ids"

    conversation_id = (request.args.get("conversation_id") or "").strip()
    user_lat = request.args.get("user_lat", type=float)
    user_lon = request.args.get("user_lon", type=float)
    prefer_open = str(request.args.get("prefer_open", "")).lower() in {"1","true","yes"}
    near = str(request.args.get("near", "")).lower() in {"1","true","yes"}
    radius_km = float(request.args.get("radius_km", 1.5))

    # Filtros forzados
    ids_raw = (request.args.get("ids") or "").strip()
    pids_raw = (request.args.get("place_ids") or "").strip()
    filter_ids = {x.strip() for x in ids_raw.split(",") if x.strip()}
    filter_pids = {x.strip() for x in pids_raw.split(",") if x.strip()}

    LAT, LON = "LATITUD_TUI", "LONGITUD_TUI"
    if LAT not in df.columns or LON not in df.columns:
        return jsonify([])

    # 1) BBOX
    bbox_df = df[df[LAT].between(south, north) & df[LON].between(west, east)].copy()
    

    # 2) Forzados desde todo el df
    forced = pd.DataFrame()
    if filter_ids and "ID" in df.columns:
        forced = pd.concat([forced, df[df["ID"].astype(str).isin(filter_ids)]], ignore_index=True)
    if filter_pids and "PLACE_ID" in df.columns:
        forced = pd.concat([forced, df[df["PLACE_ID"].astype(str).isin(filter_pids)]], ignore_index=True)

    if not forced.empty:
        dedup_keys = [k for k in ["PLACE_ID", "ID", LAT, LON] if k in forced.columns]
        forced = forced.drop_duplicates(subset=dedup_keys, keep="first")

    # 3) Resto ordenado por rating (desempate) + limit
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

    # Proximidad / abiertos ahora / rerank conversación
    if user_lat is not None and user_lon is not None:
        others = _sort_by_distance(others, user_lat, user_lon)
        if near:
            others = _filter_by_radius_km(others, user_lat, user_lon, radius_km=radius_km)

    if prefer_open:
        now_dt = datetime.now(MAD_TZ)
        others = _aplica_filtros(others, "abiertos ahora", now_dt=now_dt)

    others = _boost_recent_by_conv(others, conversation_id, boost=0.6)

    # 4) Resultado
    result = pd.concat([forced, others], ignore_index=True)
    dedup_keys = [k for k in ["PLACE_ID", "ID", "NOMBRE_TUI", LAT, LON] if k in result.columns]
    if dedup_keys:
        result = result.drop_duplicates(subset=dedup_keys, keep="first")

    def row_to_obj(r: pd.Series) -> Dict[str, Any]:
        def as_list(s: Any) -> List[str]:
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
            "total_reviews": (
                int(r["TOTAL_VALORACIONES_TUI"]) if pd.notna(r.get("TOTAL_VALORACIONES_TUI")) else None
            ),
        }

    objects = [row_to_obj(r) for _, r in result.iterrows()]

    # Modo compacto para el mapa
    if want_place_ids_only:
        pid_list: List[str] = []
        for o in objects:
            pid = (o.get("place_id") or "").strip()
            if pid:
                pid_list.append(pid)
        # Únicos preservando orden
        pid_list = list(dict.fromkeys(pid_list))
        return jsonify({"place_ids": pid_list})

    return jsonify(objects)


@app.route("/api/lookup_place_ids", methods=["GET"])
def api_lookup_place_ids():
    """Devuelve name y coords de una lista dada de place_ids."""
    ids_raw = request.args.get("place_ids", "") or ""
    ids = [x.strip() for x in ids_raw.split(",") if x and x.strip()]
    if not ids:
        return jsonify([])
    if "PLACE_ID" not in df.columns or "LATITUD_TUI" not in df.columns or "LONGITUD_TUI" not in df.columns:
        return jsonify([])

    sub = df.loc[
        df["PLACE_ID"].astype(str).isin(ids),
        ["PLACE_ID", "NOMBRE_TUI", "LATITUD_TUI", "LONGITUD_TUI"],
    ].copy()
    out: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        try:
            out.append(
                {
                    "place_id": str(r["PLACE_ID"]),
                    "name": (r.get("NOMBRE_TUI") or ""),
                    "lat": float(r["LATITUD_TUI"]),
                    "lon": float(r["LONGITUD_TUI"]),
                }
            )
        except Exception:
            continue
    return jsonify(out)


@app.route("/api/conversation_place_ids/<conversation_id>", methods=["GET"])
def api_conversation_place_ids(conversation_id: str):
    """Devuelve place_ids/nombres guardados para una conversación."""
    st = _conv_get(conversation_id)
    place_ids = st.get("last_map_ids", [])
    names: List[str] = []
    try:
        if place_ids and "PLACE_ID" in df.columns and "NOMBRE_TUI" in df.columns:
            names = (
                df.loc[df["PLACE_ID"].astype(str).isin(place_ids), "NOMBRE_TUI"]
                .astype(str).str.strip().tolist()
            )
    except Exception:
        names = []
    return jsonify({"place_ids": place_ids, "names": names, "ts": st.get("ts", 0)})



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Usuario o email ya existe")
            return render_template("register.html")
        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))
    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter((User.username == u) | (User.email == u)).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))
        flash("Credenciales no válidas")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ======================
# Favoritos API
# ======================
@app.route("/api/favorites", methods=["GET"])
@login_required
def get_favorites():
    """Devuelve ids propios + place_ids asociados del usuario actual."""
    favs = Favorite.query.filter_by(user_id=current_user.id).all()
    tui_ids = [f.tui_id for f in favs]
    place_ids: List[str] = []
    try:
        q = db.session.query(Tui.place_id).filter(Tui.id.in_(tui_ids)).all()
        place_ids = [p[0] for p in q if p and p[0]]
    except Exception:
        place_ids = []
    return jsonify({"tui_ids": tui_ids, "place_ids": place_ids})


@app.route("/api/favorites/toggle", methods=["POST"])
@login_required
def toggle_favorite():
    """Alterna favorito del usuario sobre un TUI localizado por varias claves."""
    data = request.get_json(force=True) or {}
    tui_id = data.get("tui_id")
    place_id = (data.get("place_id") or "").strip() or None
    name = (data.get("name") or "").strip() or None
    lat = data.get("lat")
    lon = data.get("lon")

    place = None
    if tui_id:
        try:
            place = Tui.query.get(int(tui_id))
        except Exception:
            place = None

    if not place and place_id:
        place = Tui.query.filter_by(place_id=place_id).first()

    if not place and name and lat is not None and lon is not None:
        try:
            lat = float(lat)
            lon = float(lon)
            place = (
                Tui.query.filter(Tui.NOMBRE_TUI == name)
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


# ======================
# Main
# ======================
# @app.route("/")
# @login_required
# def home():
#     return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
