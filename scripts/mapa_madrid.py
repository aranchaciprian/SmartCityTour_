# -*- coding: utf-8 -*-
import shelve, time, json, re, unicodedata
import pandas as pd, folium
from tqdm import tqdm
from geopy.geocoders import Nominatim
from ratelimit import limits, sleep_and_retry
import os
import re
from collections import Counter, defaultdict

# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube desde scripts/ al raíz
CSV_FILE = os.path.join(ROOT, "data", "BBDD_TUI.csv")
HTML_OUT  = os.path.join(ROOT, "static", "mapa", "mapa_madrid.html")
CACHE_DB = os.path.join(ROOT, "data", "geocode_cache.db")

CAT2EMOJI = {
    "TRANSPORTE":                        "🚖",
    "GASOLINERAS Y APARCAMIENTOS":       "⛽",
    "ALOJAMIENTOS":                      "🏨",
    "ESPACIOS DEPORTIVOS":               "⚽",
    "OCIO NOCTURNO":                     "🍸",
    "ADMINISTRACIÓN PÚBLICA Y SEGURIDAD":"🏛️",
    "COMIDA Y BEBIDA":                   "🍽️",
    "ESPACIOS RELIGIOSOS":               "🛐",
    "TIENDAS DE COMIDA":                 "🛒",
    "TIENDAS":                           "🛍️",
    "MODA Y BELLEZA":                    "💄",
    "EDUCACIÓN":                         "🎓",
    "SALUD Y MEDICINA":                  "🏥",
    "CULTURA, ARTE Y NATURALEZA":        "🎭",
    "OCIO":                              "🎡",
    "EVENTOS Y CELEBRACIONES":           "🎉",
    "SERVICIOS":                         "🛠️",
}

grupos = {
    "TRANSPORTE": [
        "Estaciones De Autobús", "Estaciones De Metro", "Estaciones De Tren",
        "Estación De Transporte", "Alquiler De Coches"
    ],
    "GASOLINERAS Y APARCAMIENTOS": [
        "Aparcamientos", "Gasolineras", "Estación De Carga De Vehículos Eléctricos", "Lavadero De Coches",
        "Aparcamiento Disuasorio", "Área De Servicio Para Camiones"
    ],
    "ALOJAMIENTOS": [
        "Alojamiento", "B&B (Cama Y Desayuno)", "Cabañas", "Campings",
        "Casa De Huéspedes", "Edificio De Apartamentos",
        "Habitación De Huéspedes Privada", "Hostal", "Hotel De Larga Estancia",
        "Hotel Resort", "Hoteles", "Posada", "Área Para Autocaravanas", "Moteles",
        "Complejo De Apartamentos", "Complejo De Condominios", "Parque De Casas Móviles"
    ],
    "ESPACIOS DEPORTIVOS": [
        "Campos De Golf", "Centro De Fitness", "Club Deportivo", "Coaching Deportivo",
        "Complejo Deportivo", "Estudio De Yoga", "Gimnasios", "Instalación Deportiva",
        "Piscinas", "Saunas", "Zona De Senderismo",
        "Campo Deportivo", "Centro De Deportes De Aventura", "Pista De Hielo",
        "Parque De Ciclismo", "Estación De Esquí", "Parque De Skate", "Estanque De Pesca"
    ],
    "OCIO NOCTURNO": [
        "Discotecas", "Pub"
    ],
    "ADMINISTRACIÓN PÚBLICA Y SEGURIDAD": [
        "Ayuntamientos", "Oficina De Gobierno Local", "Oficina Gubernamental", "Policía", "Marina", "Embajada", "Subunidad"
    ],
    "COMIDA Y BEBIDA": [
        "Asador", "Bar De Vinos", "Bar Y Parrilla", "Bares", "Bocatería", "Cafetería Americana",
        "Cafés", "Café Para Perros", "Chocolatería", "Comida", "Comida Para Llevar",
        "Hamburguesería", "Heladerías", "Reparto De Comida", "Restaurante Africano",
        "Restaurante Asiático", "Restaurante Brasileño", "Restaurante Bufé", "Restaurante Chino",
        "Restaurante Coreano", "Restaurante De Alta Cocina", "Restaurante De Barbacoa",
        "Restaurante De Brunch", "Restaurante De Comida Rápida", "Restaurante De Desayunos",
        "Restaurante De Oriente Medio", "Restaurante De Ramen", "Restaurante De Sushi",
        "Restaurante Español", "Restaurante Estadounidense", "Restaurante Francés",
        "Restaurante Griego", "Restaurante Indio", "Restaurante Italiano", "Restaurante Japonés",
        "Restaurante Libanés", "Restaurante Mediterráneo", "Restaurante Mexicano",
        "Restaurante Tailandés", "Restaurante Turco", "Restaurante Vegano",
        "Restaurante Vegetariano", "Restaurante Vietnamita", "Restaurantes",
        "Panaderías", "Pastelerías", "Patio De Comidas", "Pizzería", "Marisquerías", "Salón De Té",
        "Cibercafé", "Restaurante De Postres"
    ],
    "ESPACIOS RELIGIOSOS": [
        "Iglesias", "Lugar De Culto", "Mezquitas", "Cementerio", "Templo Hindú"
    ],
    "TIENDAS DE COMIDA": [
        "Carnicerías", "Charcuterías", "Mercados", "Mayorista", "Supermercado Asiático", "Supermercados"
    ],
    "TIENDAS": [
        "Tienda De Alimentación", "Tienda De Artículos Para El Hogar", "Tienda De Açaí",
        "Tienda De Bagels", "Tienda De Bicicletas", "Tienda De Bricolaje",
        "Tienda De Comestibles", "Tienda De Conveniencia", "Tienda De Deportes",
        "Tienda De Descuentos", "Tienda De Donuts", "Tienda De Dulces", "Tienda De Electrónica",
        "Tienda De Licores", "Tienda De Mascotas", "Tienda De Muebles", "Tienda De Móviles",
        "Tienda De Regalos", "Tienda De Repuestos", "Tienda De Ropa", "Tienda De Zumos",
        "Tiendas", "Zapaterías", "Gran Almacén", "Floristerías", "Confitería",
        "Ferretería", "Librerías", "Joyerías", "Centros Comerciales", "Tienda Tipo Almacén"
    ],
    "MODA Y BELLEZA": [
        "Barbería", "Cuidado Del Cabello", "Esteticista", "Estudio De Tatuajes Y Piercings",
        "Peluquerías", "Salón De Belleza", "Salón De Uñas", "Spas", "Sastre", "Maquillador", "Cuidado De Pies", "Centro De Bronceado"
    ],
    "EDUCACIÓN": [
        "Bibliotecas", "Colegios", "Escuela Primaria", "Instituto", "Universidades", "Preescolar"
    ],
    "SALUD Y MEDICINA": [
        "Salud", "Clínica Dental", "Clínica Dermatológica", "Dentistas", "Droguerías",
        "Farmacias", "Hospitales", "Fisioterapeuta", "Masajes", "Laboratorio Médico",
        "Médico", "Quiropráctico", "Centro De Bienestar", "Veterinaria"
    ],
    "CULTURA, ARTE Y NATURALEZA": [
        "Centro Cultural", "Lugar De Interés", "Escultura", "Monumento",
        "Museos", "Plaza", "Sitio Histórico", "Teatro De Artes Escénicas", 'Lugar Emblemático', 
        "Galería De Arte", "Estudio De Arte", "Acuario", "Jardín", "Jardín Botánico", "Zona De Picnic",
        "Granjas", "Zoológicos", "Parque Nacional", "Lugar Histórico", "Centro De Información Turística", "Mirador", "Planetario"
    ],
    "OCIO": [
        "Cine", "Club De Comedia", "Centro De Ocio",
        "Boleras", "Parque Infantil",
        "Parques", "Parques De Atracciones", "Salón Recreativo",
        "Atracción Turística", "Campamento Infantil", "Casino", "Karaoke"
    ],
    "EVENTOS Y CELEBRACIONES": [
        "Sala De Baile", "Sala De Conciertos", "Sala De Eventos", "Salón De Bodas", "Recinto",
        "Servicio De Catering", "Auditorio", "Salón De Banquetes", "Estadio", "Pabellón", "Establecimiento", "Centro De Convenciones", "Centro Comunitario"
    ],
    "SERVICIOS": [
        "Cerrajero", "Electricista", "Fontanero", "Empresa De Mudanzas", "Empresa De Pintura",
        "Inmobiliarias", "Agencia De Cuidado Infantil", "Agencia De Excursiones", "Agencia De Viajes",
        "Almacenaje", "Aseguradora", "Consultoras", "Contratista General", "Mensajería",
        "Oficina Corporativa", "Operador De Telecomunicaciones", "Organizador De Campamentos De Verano",
        "Servicios Financieros", "Tanatorios", "Taller Mecánico", "Concesionario",
        "Abogado", "Alquiler De Películas", "Aseos Públicos", "Banco", "Cajero Automático", "Contabilidad",
        "Lavandería", "Contratista De Techos", "Oficina De Correos", "Dirección"
    ]
}

# Categoría de respaldo
if "OTROS" not in grupos:
    grupos["OTROS"] = []
if "OTROS" not in CAT2EMOJI:
    CAT2EMOJI["OTROS"] = "📍"

# ------------------------------------------------------------------
# CARGA CSV
df = pd.read_csv(CSV_FILE, sep=",", encoding="utf-8-sig", dtype=str, on_bad_lines="skip")

#----------------------------------------------------------------------------------
# ID único y PLACE_ID secuencial si falta
df.reset_index(inplace=True)
df.rename(columns={"index": "ID"}, inplace=True)
if "PLACE_ID" not in df.columns:
    df["PLACE_ID"] = None
df["PLACE_ID"] = df["PLACE_ID"].where(
    df["PLACE_ID"].notna() & (df["PLACE_ID"].astype(str).str.strip() != ""),
    (df["ID"].astype(int) + 1).astype(str)
)

def safe_get_series(col, default=""):
    if col in df.columns:
        s = df[col]
        return s if default is None else s.fillna(default)
    return pd.Series([default] * len(df), index=df.index)

df["DESCRIPCION_TUI"] = safe_get_series("DESCRIPCION_TUI", "").astype(str)
df["CATEGORIA_TUI"]   = safe_get_series("CATEGORIA_TUI", "").astype(str)
df["TIPOS_TUI"]       = safe_get_series("TIPOS_TUI", "").astype(str)

def safe_get(val, default=""):
    return default if (pd.isna(val) or (isinstance(val, str) and not val.strip())) else val

# Mostrar NaN/vacíos como "nan"
def nan_or_str(v):
    return "nan" if (pd.isna(v) or (isinstance(v, str) and not v.strip())) else str(v).strip()

# ------------------------------------------------------------------
# LAT/LON (arreglo coma decimal + extracción robusta)
lat_raw = safe_get_series("LATITUD_TUI", "").str.replace(",", ".", regex=False)
lon_raw = safe_get_series("LONGITUD_TUI", "").str.replace(",", ".", regex=False)
lat_num = pd.to_numeric(lat_raw.str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors="coerce")
lon_num = pd.to_numeric(lon_raw.str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors="coerce")
df["LAT"] = lat_num
df["LON"] = lon_num

# ------------------------------------------------------------------
# RATING_TUI y TOTAL_VALORACIONES_TUI (conversión a numérico)
rating_raw = safe_get_series("RATING_TUI", "")
rating_num = pd.to_numeric(
    rating_raw.str.replace(",", ".", regex=False).str.extract(r'(-?\d+(?:\.\d+)?)')[0],
    errors="coerce"
)
df["RATING"] = rating_num

reviews_raw = safe_get_series("TOTAL_VALORACIONES_TUI", "")
reviews_num = pd.to_numeric(reviews_raw.str.extract(r'(\d+)')[0], errors="coerce")
df["TOTAL_REVIEWS"] = reviews_num

# ------------------------------------------------------------------
# Geocoder (sólo si faltan coords)
geolocator = Nominatim(user_agent="mi_mapa_geocoder", timeout=10)

@sleep_and_retry
@limits(calls=1, period=1)
def geocode(addr: str):
    try:
        loc = geolocator.geocode(addr)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return float("nan"), float("nan")

cache = shelve.open(CACHE_DB)
for i, row in tqdm(df.iterrows(), total=len(df), desc="Geocodificando faltantes"):
    if pd.notna(row["LAT"]) and pd.notna(row["LON"]):
        continue
    addr = safe_get(row.get("DIRECCION_TUI", "")) or safe_get(row.get("DIRECCION", ""))
    if not addr:
        continue
    if addr in cache:
        lat, lon = cache[addr]
    else:
        lat, lon = geocode(addr)
        cache[addr] = (lat, lon)
        time.sleep(1.1)
    df.at[i, "LAT"] = lat
    df.at[i, "LON"] = lon
cache.close()

df = df.dropna(subset=["LAT", "LON"])
print("Filas válidas →", len(df))

# ------------------------------------------------------------------
# Normalización + utilidades
def norm(s: str) -> str:
    s = str(s or "").strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def split_normalize(cell: str):
    """Separa por , | / ; · normaliza para comparar · deduplica manteniendo orden."""
    parts = re.split(r"[,\|/;]+", str(cell or ""))
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        np = norm(p)
        if np not in seen:
            seen.add(np)
            out.append(p)
    return out

# Mapeos canónicos
norm_cat2canon = {norm(c): c for c in grupos.keys()}
tipo2cat = {}
valid_tipos_norm = set()
for cat, lista in grupos.items():
    for t in lista:
        nt = norm(t)
        valid_tipos_norm.add(nt)
        tipo2cat.setdefault(nt, set()).add(cat)

# ------------------------------------------------------------------
# Construcción de records con LÓGICA DE UNIÓN (OR)
records = []
present_types_by_cat = {cat: set() for cat in grupos}
cat_needing_empty = set()

for _, r in df.iterrows():
    # TIPOS reconocidos (canónicos)
    tipos_src = split_normalize(r.get("TIPOS_TUI", ""))
    tipos_valid = []
    for t in tipos_src:
        nt = norm(t)
        if nt in valid_tipos_norm:
            for cat, lista in grupos.items():
                for can in lista:
                    if norm(can) == nt:
                        tipos_valid.append(can)
                        break
                else:
                    continue
                break

    # CATEGORÍAS desde la columna (si hay)
    cats_src = split_normalize(r.get("CATEGORIA_TUI", ""))
    cats_from_col = []
    for c in cats_src:
        nc = norm(c)
        if nc in norm_cat2canon:
            cats_from_col.append(norm_cat2canon[nc])

    # CATEGORÍAS derivadas desde TIPOS
    cats_from_types = []
    for t in tipos_valid:
        nt = norm(t)
        for c in tipo2cat.get(nt, []):
            if c not in cats_from_types:
                cats_from_types.append(c)

    # Unión OR (y fallback a OTROS)
    categories = []
    for c in cats_from_col + cats_from_types:
        if c not in categories:
            categories.append(c)
    if not categories:
        categories = ["OTROS"]

    subcats_display = list(dict.fromkeys(tipos_valid))

    cats_without_subs = set()
    for cat in categories:
        subs_in_cat = [s for s in subcats_display if s in grupos.get(cat, [])]
        if subs_in_cat:
            present_types_by_cat[cat].update(subs_in_cat)
        else:
            cats_without_subs.add(cat)
    if cats_without_subs:
        cat_needing_empty.update(cats_without_subs)

    val_acces = nan_or_str(r.get("ACCESIBILIDAD_SILLA_RUEDAS"))
    val_precio = nan_or_str(r.get("PRECIO"))
    val_estado = nan_or_str(r.get("ESTADO_NEGOCIO"))
    val_reserva = nan_or_str(r.get("RESERVA_POSIBLE"))

    # rating y total_reviews numéricos (None si NaN)
    rating_val = r.get("RATING")
    rating_out = float(rating_val) if pd.notna(rating_val) else None
    reviews_val = r.get("TOTAL_REVIEWS")
    reviews_out = int(reviews_val) if pd.notna(reviews_val) else None

    records.append({
        "id": int(r["ID"]),
        "place_id": safe_get(r.get("PLACE_ID", str(int(r["ID"]) + 1))),
        "name": safe_get(r.get("NOMBRE_TUI", "") or r.get("NOMBRE", "")),
        "description": safe_get(r.get("DESCRIPCION_TUI", "")),
        "lat": float(r["LAT"]),
        "lon": float(r["LON"]),
        "address": safe_get(r.get("DIRECCION_TUI", "") or r.get("DIRECCION", ""), f"{float(r['LAT']):.5f}, {float(r['LON']):.5f}"),
        "categories": categories,           # se usa para agrupar casillas
        "subcategories": subcats_display,   # TIPOS_TUI LIMPIOS → filtrado real por tipos
        "email": safe_get(r.get("EMAIL", "")),
        "phone": safe_get(r.get("TELEFONO", "")),
        "website":   safe_get(r.get("CONTENT_URL", "") or r.get("WEBSITE", "")),
        "gmaps_url": safe_get(r.get("URL", "")),
        "horario": safe_get(r.get("HORARIO", "")),
        "precio": val_precio,
        "estado_negocio": val_estado,
        "reserva_posible": val_reserva,
        "accesibilidad_silla_ruedas": val_acces,
        "rating": rating_out,
        "total_reviews": reviews_out,
    })

# cat2subs = subtipos realmente presentes por categoría (+ "(sin subcategoría)" donde haga falta)
cat2subs = {}
for cat in grupos:
    lst = sorted(present_types_by_cat.get(cat, set()), key=lambda s: s.casefold())
    if cat in cat_needing_empty and "(sin subcategoría)" not in lst:
        lst.append("(sin subcategoría)")
    if cat == "OTROS" and "(sin subcategoría)" not in lst:
        lst.append("(sin subcategoría)")
    cat2subs[cat] = lst

print("Registros totales:", len(df))
print("Con categorías por registro:", sum(1 for r in records if r["categories"]))
print("Con subtipos válidos:",      sum(1 for r in records if r["subcategories"]))

# ------------------------------------------------------------------
# Mapa
m = folium.Map(location=[40.4168, -3.7038], zoom_start=11, tiles="CartoDB positron")

def safe_json_dump_text(obj):
    txt = json.dumps(obj, ensure_ascii=False)
    return txt.replace("</script>", "<\\/script>")

def unique_values(field):
    return sorted({str(r[field]) for r in records if r.get(field) is not None})

unique_filters = {
    "acc_silla": unique_values("accesibilidad_silla_ruedas"),
    "precio":    unique_values("precio"),
    "estado":    unique_values("estado_negocio"),
    "reserva":   unique_values("reserva_posible"),
}

# 2.a — NO embebemos el mega-JSON de 30k registros
data_json           = "[]"
cat2subs_json       = safe_json_dump_text(cat2subs)
emoji_json          = safe_json_dump_text(CAT2EMOJI)
unique_filters_json = safe_json_dump_text(unique_filters)

# --- JS embebido ---
js = """
<!-- CARGA EL JS GLOBAL DE FAVORITOS (debe existir en /static/favorites/favorites.js) -->
<script src="/static/favorites/favorites.js"></script>

<style>
#filter-panel { position:relative; width:300px; max-height:85vh; font-family:sans-serif; z-index:5000; }
#filter-panel .panel { background:white; border-radius:8px; box-shadow:0 6px 20px rgba(0,0,0,0.25); padding:8px 12px 6px; display:flex; flex-direction:column; gap:6px; max-height:85vh; overflow:hidden; }
#filter-header { display:flex; justify-content:space-between; align-items:center; }
#filter-header .title-wrapper { display:flex; align-items:center; gap:6px; }
#filter-header h3 { margin:0; font-size:1.2em; font-weight:600; }
#panel-toggle { cursor:pointer; font-size:1.2em; user-select:none; }
#filter-body { display:flex; flex-direction:column; gap:6px; overflow:hidden; }
#filter-panel .buttons { display:flex; gap:6px; margin-top:2px; }
#filter-panel button { flex:1; padding:6px 10px; cursor:pointer; border:1px solid #d0d0d0; background:#f0f0f5; border-radius:5px; font-size:0.9em; transition:background .15s ease; }
#filter-panel button:hover { background:#e2e2ea; }
.category-block { border-top:1px solid #eee; padding-top:6px; margin-top:6px; }
.category-header { display:flex; align-items:center; cursor:pointer; gap:8px; user-select:none; padding:6px 8px; border-radius:6px; background:#fafafa; }
.category-header:hover { background:#f2f2f9; }
.category-header .title { flex:1; display:flex; align-items:center; gap:6px; }
.subcat-list { margin-left:4px; margin-top:4px; display:none; flex-direction:column; gap:6px; max-height:300px; overflow:auto; padding-left:10px; }
.toggle-arrow { width:16px; display:inline-block; transform:rotate(0deg); transition:transform .2s ease; font-weight:bold; }
.category-header.expanded .toggle-arrow { transform:rotate(90deg); }
.checkbox-wrapper { display:flex; align-items:center; gap:6px; padding:2px 0; }
.small-checkbox { width:16px; height:16px; }
.fav-button { 
  cursor:pointer; padding:4px 8px; border:1px solid #d1d5db;
  border-radius:6px; background:#f9fafb; color:#111;
  transition: background .15s ease, border-color .15s ease;
}
.fav-button.faved { 
  background:#e5e7eb;
  border-color:#9ca3af;
}
.download-icon { cursor:pointer; font-size:18px; user-select:none; margin-left:auto; padding:2px 6px; border-radius:4px; transition:background .15s ease; }
.download-icon:hover { background:#e0e0ea; }
.leaflet-div-icon.emoji-pin{ background: transparent; border: none; font-size: 24px; line-height: 1; }
.leaflet-div-icon.user-pin { background: transparent; border: none; font-size: 28px; line-height: 1; }

/* Botón “Ubicarme” */
.leaflet-bar .locate-btn{
  width:34px; height:34px; line-height:34px; text-align:center;
  font-size:20px; text-decoration:none; display:block;
}
.leaflet-bar .locate-btn.loading { opacity:.6; pointer-events:none; }

</style>

<script id="data-json" type="application/json">""" + data_json + """</script>
<script id="cat2subs-json" type="application/json">""" + cat2subs_json + """</script>
<script id="emoji-json" type="application/json">""" + emoji_json + """</script>
<script id="filters-json" type="application/json">""" + unique_filters_json + """</script>

<script>
/* ======= Fallback por si no está cargado static/favorites/favorites.js ======= */
if (typeof window.toggleFavoriteById !== "function") {
  window.toggleFavoriteById = async function(arg){
    // arg puede ser Number (tui_id) o un objeto {tui_id, place_id?, name?, lat?, lon?}
    let payload = {};
    if (typeof arg === 'number' || typeof arg === 'string') payload = { tui_id: arg };
    else if (arg && typeof arg === 'object') payload = arg;
    const res = await fetch("/api/favorites/toggle", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      credentials: "same-origin",
      redirect: "manual",
      body: JSON.stringify(payload)
    });
    if (res.status === 401 || res.redirected || res.type === "opaqueredirect") {
      return { ok:false, error:"AUTH_REQUIRED" };
    }
    try { 
      return await res.json(); 
    } catch { 
      return { ok:false, error:"BAD_RESPONSE" }; 
    }
  }
}

if (typeof window.loadMyFavorites !== "function") {
  window.loadMyFavorites = async function(){
    const res = await fetch("/api/favorites", {
      credentials: "same-origin",
      redirect: "manual"
    });
    if (res.status === 401 || res.redirected || res.type === "opaqueredirect") return { tui_ids: [], place_ids: [] };
    const data = await res.json().catch(()=>({}));
    return { tui_ids: data.tui_ids || [], place_ids: data.place_ids || [] };
  }
}

/* ===================================================================== */

function getFoliumMap() {
  if (window._folium_map) return window._folium_map;
  for (const k in window) { try { if (window[k] instanceof L.Map) { window._folium_map = window[k]; return window[k]; } } catch {}
  }
  return null;
}
function withMap(cb, tries=0) {
  const m = getFoliumMap();
  if (m) cb(m); else if (tries < 60) setTimeout(()=>withMap(cb, tries+1), 100);
}

const rawData   = JSON.parse(document.getElementById('data-json').textContent   || "[]");
const cat2subs  = JSON.parse(document.getElementById('cat2subs-json').textContent || "{}");
const cat2emoji = JSON.parse(document.getElementById('emoji-json').textContent  || "{}");
const filtersUnique = JSON.parse(document.getElementById('filters-json').textContent || "{}");

/* ===== Favoritos del servidor (tui_id + place_id) ===== */
let serverFavoritesTuiIds = new Set();     // tui_ids (Number, DB)
let serverFavoritesPlaceIds = new Set();   // place_ids (String)
let favoritesLayer = L.layerGroup();       // capa visual para togglear desde el panel

const allMarkers = [];
const recById = new Map();               // para exportar por tui_id
const scheduleCache = new Map();

/* === Normalizador JS equivalente al de Python === */
function normJS(s){
  s = (s || "").toString().trim();
  s = s.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
  s = s.replace(/\\s+/g, " ").trim();
  return s.toLowerCase();
}

/* --- helpers de horario --- */
const DAY_INDEX = { "monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6,
  "lunes":0,"martes":1,"miércoles":2,"miercoles":2,"jueves":3,"viernes":4,"sábado":5,"sabado":5,"domingo":6 };
function normalizeSpaces(s){ return (s||"").replace(/[\\u2000-\\u200A\\u202F\\u205F\\u00A0]/g," ").trim(); }
function parseTimeToken(tok){ tok = normalizeSpaces(tok).replace(/\\./g,"").toUpperCase();
  const m = tok.match(/^\\s*(\\d{1,2})(?::(\\d{2}))?\\s*(AM|PM)?\\s*$/i);
  if(!m) return {hh:null, mm:null, mer:null, valid:false};
  let hh = parseInt(m[1],10), mm = m[2]?parseInt(m[2],10):0, mer = m[3]?m[3].toUpperCase():null;
  if(hh>24 || mm>59) return {hh:null, mm:null, mer:null, valid:false};
  return {hh, mm, mer, valid:true};
}
function toMinutes(hh,mm,mer){ if(mer==="AM"){ if(hh===12) hh=0; } else if(mer==="PM"){ if(hh!==12) hh+=12; } return hh*60+mm; }
function parseSchedule(horario){
  const sched = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]};
  if(!horario) return sched;
  const normalized = normalizeSpaces(horario).replace(/[—–−]/g,"-");
  const parts = normalized.split("|");
  for(const seg0 of parts){
    const seg = seg0.trim();
    const idxColon = seg.indexOf(":"); if(idxColon===-1) continue;
    const dayRaw = seg.slice(0, idxColon).trim().toLowerCase();
    const dayIdx = DAY_INDEX[dayRaw]; if(dayIdx===undefined) continue;
    let rest = seg.slice(idxColon+1).trim();
    if(/\\b(closed|cerrado)\\b/i.test(rest)){ sched[dayIdx]=[]; continue; }
    if(/(open\\s*24\\s*hours|abierto\\s*24\\s*horas|\\b24\\s*h\\b|\\b24\\s*horas\\b)/i.test(rest)){ sched[dayIdx]=[[0,1440]]; continue; }
    const intervals = rest.split(",").map(s=>s.trim()).filter(Boolean);
    for(const it of intervals){
      const m = it.split(/\\s*-\\s*/); if(m.length!==2) continue;
      const [a,b] = m; const t1=parseTimeToken(a), t2=parseTimeToken(b);
      if(!t1.valid || !t2.valid) continue;
      if(t1.mer==null && t2.mer!=null) t1.mer=t2.mer; if(t2.mer==null && t1.mer!=null) t2.mer=t1.mer;
      const m1=toMinutes(t1.hh,t1.mm,t1.mer), m2=toMinutes(t2.hh,t2.mm,t2.mer);
      if(Number.isNaN(m1)||Number.isNaN(m2)||m1===m2) continue;
      if(m1<m2) sched[dayIdx].push([m1,m2]); else { sched[dayIdx].push([m1,1440]); sched[(dayIdx+1)%7].push([0,m2]); }
    }
  }
  for(const d of Object.keys(sched)){
    const arr = sched[d].sort((x,y)=>x[0]-y[0]);
    const merged=[]; for(const [s,e] of arr){ if(!merged.length||s>merged[merged.length-1][1]) merged.push([s,e]); else merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],e); }
    sched[d]=merged;
  }
  return sched;
}
function isOpenAtSchedule(sched, dow, minute){ return (sched[dow]||[]).some(([s,e])=>minute>=s && minute<e); }

/* === Estado de filtro proveniente del CHATBOT === */
let chatFilter = {
  active: false,
  placeIds: new Set(),
  names: [],
};

let strictChatMode = false;

/* Extraer place_id de gmaps_url como respaldo */
function extractPlaceIdFromUrl(url){
  if(!url) return null;
  const m = url.match(/(?:[?&])(place_id|placeid|query_place_id)=([^&#]+)/i);
  return m ? decodeURIComponent(m[2]) : null;
}

/* ==== Utilidades favoritos (capa + UI) por tui_id o place_id ==== */
function isFavoritedByIds(tuiId, placeId){
  if (placeId && serverFavoritesPlaceIds.has(String(placeId))) return true;
  if (tuiId != null && serverFavoritesTuiIds.has(Number(tuiId))) return true;
  return false;
}
function refreshFavoritesLayer(){
  favoritesLayer.clearLayers();
  for (const mk of allMarkers){
    const pid = mk && mk._rec && mk._rec.place_id ? mk._rec.place_id : null;
    if (isFavoritedByIds(mk._tuiId, pid)) { favoritesLayer.addLayer(mk); }
  }
}
function setFavButtonState(btn, isFav){
  if (!btn) return;
  btn.classList.toggle("faved", !!isFav);
  btn.textContent = isFav ? "❤ Quitar favorito" : "❤ Favorito";
}

window.__toggleFavoriteFromMap = async function(tuiId){
  try{
    const rec = recById.get(Number(tuiId));
    const payload = { tui_id: tuiId };
    if (rec && rec.place_id) payload.place_id = rec.place_id;
    if (rec && rec.name) payload.name = rec.name;
    if (rec && typeof rec.lat === 'number' && typeof rec.lon === 'number') { payload.lat = rec.lat; payload.lon = rec.lon; }
    const data = await toggleFavoriteById(payload);
    if (!data || !data.ok) {
      if (data && data.error === "AUTH_REQUIRED") {
        alert("Inicia sesión para guardar favoritos (menú: /login).");
      } else {
        alert("Error al guardar favorito");
      }
      return;
    }
    const pid = (data.place_id || (rec && rec.place_id) || "").toString();
    const tid = Number(data.tui_id || tuiId);
    if (data.favorite) {
      if (pid) serverFavoritesPlaceIds.add(pid);
      if (!Number.isNaN(tid)) serverFavoritesTuiIds.add(tid);
    } else {
      if (pid) serverFavoritesPlaceIds.delete(pid);
      if (!Number.isNaN(tid)) serverFavoritesTuiIds.delete(tid);
    }

    // actualizar capa y botón abierto
    refreshFavoritesLayer();
    const openPop = document.querySelector(".leaflet-popup");
    if (openPop){
      const btn = openPop.querySelector(`.fav-button[data-tui-id="${tuiId}"]`);
      setFavButtonState(btn, data.favorite);
    }
  } catch(e){
    alert("Error de red al guardar favorito");
  }
};

async function initFavorites(){
  try {
    const favs = await loadMyFavorites();
    const tuiIds = favs && favs.tui_ids ? favs.tui_ids : [];
    const placeIds = favs && favs.place_ids ? favs.place_ids : [];
    serverFavoritesTuiIds = new Set((tuiIds || []).map(Number));
    serverFavoritesPlaceIds = new Set((placeIds || []).map(String));
    refreshFavoritesLayer();
  } catch(e){
    serverFavoritesTuiIds = new Set();
    serverFavoritesPlaceIds = new Set();
  }
}

withMap((map)=>{
  // Panel de filtros
  const FilterControl = L.Control.extend({
    onAdd: function(){
      const container = L.DomUtil.create('div'); container.id='filter-panel'; container.className='leaflet-bar';
      container.innerHTML = `
        <div class="panel">
          <div id="filter-header">
            <div class="title-wrapper"><h3>Filtros</h3></div>
            <div id="panel-toggle">&#9660;</div>
          </div>
          <div id="filter-body">
            <div class="buttons">
              <button id="show-all">Mostrar todo</button>
              <button id="hide-all">Ocultar todo</button>
            </div>
            <div id="categories-container" style="overflow:auto; flex:1; margin-top:4px; padding-right:4px;"></div>
          </div>
        </div>`;
      L.DomEvent.disableClickPropagation(container);
      L.DomEvent.disableScrollPropagation(container);
      return container;
    },
    options: { position:'topright' }
  });
  map.addControl(new FilterControl());

  // =========================
  // 2.b — CARGADOR INCREMENTAL
  // =========================
  const loaded = new Set();
  const liveLayer = L.layerGroup().addTo(map);

  function createMarker(rec){
    const mains = (rec.categories || []);
    const subs  = (rec.subcategories || []);

    // 👉 Set con TODOS los tipos del registro (normalizados) para filtrado por TIPOS_TUI
    const typesNormSet = new Set((subs || []).map(normJS));

    // 👉 Mapa por categoría a subtipos propios del registro; si no hay, "(sin subcategoría)"
    const catToSubs = {};
    const catsEmpty = new Set();
    for (const m of mains) {
      const globalSubsOfCat = cat2subs[m] || [];
      const recSubsInCat = (subs || []).filter(s => globalSubsOfCat.includes(s));
      if (recSubsInCat.length) {
        catToSubs[m] = recSubsInCat;
      } else {
        catToSubs[m] = ["(sin subcategoría)"];
        catsEmpty.add(m);
      }
    }

    const emoji = cat2emoji[mains[0]] || "📍";

    const websiteHTML = rec.website ? `<a href="${rec.website}" target="_blank" rel="noopener noreferrer">Abrir sitio</a>` : '—';
    const gmapsHTML   = rec.gmaps_url ? `<a href="${rec.gmaps_url}" target="_blank" rel="noopener noreferrer">Ver en Google Maps</a>` : '—';
    const ratingDisp  = (rec.rating!=null && !Number.isNaN(rec.rating)) ? rec.rating : '—';
    const reviewsDisp = (rec.total_reviews!=null && !Number.isNaN(rec.total_reviews)) ? rec.total_reviews : '—';

    // 👉 ID de TUI para favoritos
    const tuiId = rec.id;

    // (Mantenemos placeId sólo como soporte del filtro del chat existente)
    const placeId = (rec.place_id || extractPlaceIdFromUrl(rec.gmaps_url) || "").trim();

    const popupContent = `
      <div style="font-size:14px; max-width:850px; max-height:700px; overflow:auto">
        <b>${rec.name}</b><br>
        <small>${rec.address||""}</small><br><br>
        ${rec.description||""}<br><br>
        <strong>Categorías:</strong> ${mains.join(", ")}<br>
        <strong>Tipos:</strong> ${subs.length? subs.join(", ") : "(sin subcategoría)"}<br>
        <strong>Sitio web:</strong> ${websiteHTML}<br>
        <strong>Google Maps:</strong> ${gmapsHTML}<br>
        <strong>Teléfono:</strong> ${rec.phone || '—'}<br>
        <strong>Precio:</strong> ${rec.precio || '—'}<br>
        <strong>Valorado:</strong> ${ratingDisp}<br>
        <strong>Número de reseñas:</strong> ${reviewsDisp}<br>
        <strong>Estado de negocio:</strong> ${rec.estado_negocio || '—'}<br>
        <strong>Reserva posible:</strong> ${rec.reserva_posible || '—'}<br>
        <strong>Accesibilidad Silla de ruedas:</strong> ${rec.accesibilidad_silla_ruedas || '—'}<br>
        <strong>Horario:</strong> ${rec.horario || '—'}<br>
        <div style="margin-top:10px">
          <button class="fav-button" data-tui-id="${tuiId}" data-rec-id="${rec.id}"
            onclick="__toggleFavoriteFromMap(${tuiId})">❤ Favorito</button>
        </div>
      </div>`;

    const icon = L.divIcon({ className: "leaflet-div-icon emoji-pin", html: emoji, iconSize: [24,24], iconAnchor: [12,12] });
    const marker = L.marker([rec.lat, rec.lon], { icon }).bindPopup(popupContent, {maxWidth:900,maxHeight:800});

    marker._recId = rec.id;
    marker._rec   = rec;
    marker._tuiId = tuiId;   // <- clave para favoritos
    marker._placeId = placeId; // <- se conserva por compatibilidad con filtro del chat

    // ➕ campos nuevos para el filtrado por TIPOS_TUI
    marker._typesNorm = typesNormSet; // Set<string> normalizados de TIPOS_TUI del registro
    marker._catToSubs = catToSubs;    // { mainCat: ["sub1", ...] | ["(sin subcategoría)"], ... }
    marker._catsEmpty = catsEmpty;    // Set<string> categorías del registro sin subtipos

    marker._currentEmoji = emoji;
    marker._updateEmoji = function(){
      const e = cat2emoji[mains[0]] || "📍"; // mantenemos emoji de 1ª categoría
      if (e !== marker._currentEmoji){
        marker._currentEmoji = e;
        marker.setIcon(L.divIcon({ className:"leaflet-div-icon emoji-pin", html:e, iconSize:[24,24], iconAnchor:[12,12] }));
      }
    };

    marker.addTo(liveLayer);
    allMarkers.push(marker);
    recById.set(rec.id, rec);

    if (isFavoritedByIds(tuiId, rec.place_id || extractPlaceIdFromUrl(rec.gmaps_url))) { favoritesLayer.addLayer(marker); }
  }

  async function fetchAndRender(){

    if (strictChatMode) {
    applyFilters();
    return;
    }

    const b = map.getBounds();
    const url = `/api/poi?south=${b.getSouth()}&west=${b.getWest()}&north=${b.getNorth()}&east=${b.getEast()}&zoom=${map.getZoom()}`;
    const res = await fetch(url);
    const data = await res.json();
    for (const rec of data){
      if (!loaded.has(rec.id)){
        loaded.add(rec.id);
        createMarker(rec);
      }
    }
    applyFilters();
    // asegurar que los favoritos del servidor quedan marcados
    refreshFavoritesLayer();
  }

    async function fetchStrictByPlaceIds(ids){
  const list = (ids || []).filter(Boolean);
  if (!list.length) return;

  // Construye query sin romper las comas
  const qs = list.map(encodeURIComponent).join(",");
  const url = `/api/poi?strict=1&place_ids=${qs}`;

  const res = await fetch(url);
  const data = await res.json();

  // Crea marcadores solo para estos registros (si no existían ya)
  for (const rec of data){
    if (!loaded.has(rec.id)) {
      loaded.add(rec.id);
      createMarker(rec);
    }
  }
  applyFilters();
}

  function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }
  map.on('moveend', debounce(()=>{ if (!strictChatMode) fetchAndRender(); }, 250));

  // Cargar favoritos del servidor ANTES de pintar
  initFavorites().then(()=> fetchAndRender());

  // Exportación CSV de favoritos (usa place_id preferentemente)
  window.__exportFavorites = function(){
    const hasByPid = serverFavoritesPlaceIds && serverFavoritesPlaceIds.size > 0;
    const hasByTid = serverFavoritesTuiIds && serverFavoritesTuiIds.size > 0;
    if (!hasByPid && !hasByTid) return alert("No has marcado favoritos todavía.");
    const headers = ["id","name","address","lat","lon","categories","subcategories","email","phone","website","gmaps_url","horario","description","rating","total_reviews"];
    const rows = [];
    const shouldPick = (rec) => {
      const pid = rec && rec.place_id ? String(rec.place_id) : null;
      if (pid && serverFavoritesPlaceIds.has(pid)) return true;
      if (serverFavoritesTuiIds.has(Number(rec.id))) return true;
      return false;
    };
    for (const [rid, rec] of recById.entries()){
      if (!shouldPick(rec)) continue;
      const row = headers.map(h=>{
        let v = rec[h];
        if (h==="categories") v=(rec.categories||[]).join("|");
        if (h==="subcategories") v=(rec.subcategories||[]).join("|");
        return `"` + String(v ?? "").replace(/"/g,'""') + `"`;
      }).join(";");
      rows.push(row);
    }
    const csv = [headers.join(";"), ...rows].join("\\r\\n");
    const blob = new Blob([csv], {type:"text/csv;charset=utf-8"});
    const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="favoritos.csv"; a.click();
  }

  // Al abrir un popup, ajustar el estado visual del botón según favoritos
  map.on('popupopen', (e)=>{
    const pop = e.popup && e.popup.getElement(); if(!pop) return;
    const btn = pop.querySelector('.fav-button');
    if (btn){
      const tid = Number(btn.getAttribute('data-tui-id'));
      let pid = null;
      const mk = allMarkers.find(m => m._tuiId === tid);
      if (mk && mk._rec && mk._rec.place_id) pid = mk._rec.place_id;
      setFavButtonState(btn, isFavoritedByIds(tid, pid));
    }
  });

  const container = document.getElementById("categories-container");

  // ================= CHATBOT =================
    const chatBlock = document.createElement("div"); chatBlock.className = "category-block";
    const chatHeader = document.createElement("div"); chatHeader.className = "category-header";
    const chatTitleW = document.createElement("div"); chatTitleW.className = "title";
    
    // ✅ checkbox maestro (mismo patrón que favoritos/categorías)
    const chatMainChk = document.createElement("input");
    chatMainChk.type = "checkbox";
    chatMainChk.className = "small-checkbox";
    chatMainChk.checked = true;
    chatMainChk.id = "chat-main-chk";
    
    const chatEmoji = document.createElement("span"); chatEmoji.textContent = "🤖";
    const chatName  = document.createElement("span"); chatName.textContent  = "CHATBOT";
    
    chatTitleW.appendChild(chatMainChk);
    chatTitleW.appendChild(chatEmoji);
    chatTitleW.appendChild(chatName);
    
    const chatArrow = document.createElement("div"); chatArrow.className = "toggle-arrow"; chatArrow.innerHTML = "&#9654;";
    chatHeader.appendChild(chatTitleW); chatHeader.appendChild(chatArrow);
    chatBlock.appendChild(chatHeader);
    
    const chatWrapper = document.createElement("div"); chatWrapper.className="subcat-list"; chatWrapper.style.paddingLeft="12px";
    const chatList = document.createElement("div"); chatList.id = "chatbot-names"; chatList.style.display="flex"; chatList.style.flexDirection="column"; chatList.style.gap="4px";
    
    // 👉 sin botón “Limpiar filtro”; solo un hint
    const chatHint = document.createElement("div"); chatHint.style.fontSize="12px"; chatHint.style.color="#666";
    chatHint.textContent = "El chat actualiza automáticamente estos resultados.";
    
    chatWrapper.appendChild(chatList);
    chatWrapper.appendChild(chatHint);
    chatBlock.appendChild(chatWrapper);
    container.appendChild(chatBlock);
    
    // toggle
    chatHeader.addEventListener("click", (e)=>{
      e.stopPropagation();
      const expanded = chatHeader.classList.toggle("expanded");
      chatWrapper.style.display = expanded ? "block" : "none";
      chatArrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
    });
    chatHeader.classList.add("expanded"); chatWrapper.style.display="block"; chatArrow.innerHTML="&#9660;";
    
    // estado del filtro proveniente del chat si no existe aún
    window.chatFilter = window.chatFilter || { active:false, placeIds:new Set(), names:[] };
    
    function renderChatPanel(){
      chatList.innerHTML = "";
      const names = chatFilter.names || [];
      if (!names.length){
        const empty = document.createElement("div");
        empty.style.color = "#888";
        empty.textContent = "— sin datos del chat —";
        chatList.appendChild(empty);
        return;
      }
    
      names.forEach(name => {
        const btn = document.createElement("button");
        btn.textContent = name;
        btn.style.textAlign = "left";
        btn.style.padding = "6px 8px";
        btn.style.border = "1px solid #d0d0d0";
        btn.style.borderRadius = "6px";
        btn.style.background = "#f9f9fb";
        btn.style.cursor = "pointer";
        btn.style.fontSize = "0.9em";
        btn.style.transition = "background .15s ease";
        btn.addEventListener("mouseenter",()=>btn.style.background="#f0f2ff");
        btn.addEventListener("mouseleave",()=>btn.style.background="#f9f9fb");
    
        btn.addEventListener("click", () => {
          const target = (name || "").toLowerCase();
          let candidate = null;
    
          // 1) nombre EXACTO
          for (const mk of allMarkers) {
            const mkName = (mk._rec?.name || "").toLowerCase();
            if (mkName === target) { candidate = mk; break; }
          }
          // 2) si no, nombre "contiene"
          if (!candidate) {
            for (const mk of allMarkers) {
              const mkName = (mk._rec?.name || "").toLowerCase();
              if (mkName.includes(target)) { candidate = mk; break; }
            }
          }
    
          if (candidate){
            map.setView(candidate.getLatLng(), 17);
            candidate.openPopup();
            const prevEmoji = candidate._currentEmoji || "📍";
            candidate.setIcon(L.divIcon({ className:"leaflet-div-icon emoji-pin", html:"✨", iconSize:[24,24], iconAnchor:[12,12] }));
            setTimeout(()=>{ candidate.setIcon(L.divIcon({ className:"leaflet-div-icon emoji-pin", html:prevEmoji, iconSize:[24,24], iconAnchor:[12,12] })); }, 850);
          } else {
            const matches = [];
            for (const mk of allMarkers) {
              const mkName = (mk._rec?.name || "").toLowerCase();
              if (mkName.includes(target)) matches.push(mk.getLatLng());
            }
            if (matches.length){
              try { map.fitBounds(L.latLngBounds(matches), { padding:[50,50] }); } catch {}
            }
          }
        });
    
        chatList.appendChild(btn);
      });
    }
    
    // checkbox maestro del chat → re-aplica filtros
    chatMainChk.addEventListener("change", applyFilters);
    
    renderChatPanel();

  // FAVORITOS (panel)
  const favBlock = document.createElement("div"); favBlock.className="category-block";
  const favHeader = document.createElement("div"); favHeader.className="category-header";
  const favTitleW = document.createElement("div"); favTitleW.className="title";
  const favChk = document.createElement("input"); favChk.type="checkbox"; favChk.className="small-checkbox"; favChk.checked=true; favChk.id="fav-main-chk";
  const favEmoji = document.createElement("span"); favEmoji.textContent="❤";
  const favName = document.createElement("span"); favName.textContent="FAVORITOS";
  const downloadIcon = document.createElement("span"); downloadIcon.className="download-icon"; downloadIcon.textContent="📥"; downloadIcon.title="Descargar favoritos como CSV";
  favTitleW.appendChild(favChk); favTitleW.appendChild(favEmoji); favTitleW.appendChild(favName); favTitleW.appendChild(downloadIcon);
  favHeader.appendChild(favTitleW); favBlock.appendChild(favHeader); container.appendChild(favBlock);
  favChk.addEventListener('change', ()=>{ if(favChk.checked){ if(!map.hasLayer(favoritesLayer)) favoritesLayer.addTo(map); } else { if(map.hasLayer(favoritesLayer)) map.removeLayer(favoritesLayer); } });
  downloadIcon.addEventListener('click', (e)=>{ e.stopPropagation(); window.__exportFavorites(); });

  // CATEGORÍAS
  const catMainBlock = document.createElement("div"); catMainBlock.className="category-block";
  const catMainHeader = document.createElement("div"); catMainHeader.className="category-header";
  const catMainTitleW = document.createElement("div"); catMainTitleW.className="title";
  const catMainChk = document.createElement("input"); catMainChk.type="checkbox"; catMainChk.className="small-checkbox"; catMainChk.checked=true; catMainChk.id="cat-main-chk";
  const catMainEmoji = document.createElement("span"); catMainEmoji.textContent="📁";
  const catMainName = document.createElement("span"); catMainName.textContent="CATEGORÍAS";
  catMainTitleW.appendChild(catMainChk); catMainTitleW.appendChild(catMainEmoji); catMainTitleW.appendChild(catMainName);
  const catMainArrow = document.createElement("div"); catMainArrow.className="toggle-arrow"; catMainArrow.innerHTML="&#9654;";
  catMainHeader.appendChild(catMainTitleW); catMainHeader.appendChild(catMainArrow);
  catMainBlock.appendChild(catMainHeader);
  const categoriesWrapper = document.createElement("div"); categoriesWrapper.className="subcat-list"; categoriesWrapper.style.paddingLeft="20px";

  for(const main of Object.keys(cat2subs)) {
    const block = document.createElement("div"); block.style.marginTop="8px";
    const header = document.createElement("div"); header.className="category-header"; header.style.padding="4px 6px";
    const titleWrapper = document.createElement("div"); titleWrapper.className="title";
    const chkCat = document.createElement("input"); chkCat.type="checkbox"; chkCat.className="small-checkbox"; chkCat.setAttribute("data-main", main); chkCat.checked=true;
    const emojiSpan = document.createElement("span"); emojiSpan.textContent = cat2emoji[main] || "📍";
    const nameSpan = document.createElement("span"); nameSpan.textContent = main;
    titleWrapper.appendChild(chkCat); titleWrapper.appendChild(emojiSpan); titleWrapper.appendChild(nameSpan);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleWrapper); header.appendChild(arrow); block.appendChild(header);
    const sublist = document.createElement("div"); sublist.className="subcat-list"; sublist.style.paddingLeft="20px";

    for (const sub of (cat2subs[main]||[])) {
      const row = document.createElement("div"); row.className="checkbox-wrapper";
      const chkSub = document.createElement("input"); chkSub.type="checkbox"; chkSub.checked=true; chkSub.setAttribute("data-main", main); chkSub.setAttribute("data-sub", sub);
      const lbl = document.createElement("span"); lbl.textContent=sub;
      row.appendChild(chkSub); row.appendChild(lbl); sublist.appendChild(row);
      chkSub.addEventListener("change", applyFilters);
    }
    block.appendChild(sublist); categoriesWrapper.appendChild(block);
    header.addEventListener("click", (e)=>{ e.stopPropagation(); const expanded=header.classList.toggle("expanded"); sublist.style.display = expanded ? "flex" : "none"; arrow.innerHTML = expanded ? "&#9660;" : "&#9654;"; });
    chkCat.addEventListener("change", ()=>{ const on=chkCat.checked; sublist.querySelectorAll("input[type=checkbox]").forEach(si=>{ si.checked=on; }); applyFilters(); });
  }
  catMainBlock.appendChild(categoriesWrapper); container.appendChild(catMainBlock);

  // Maestro de CATEGORÍAS: marca/desmarca TODAS las subcasillas del bloque
  catMainChk.addEventListener("change", ()=>{
    const on = catMainChk.checked;
    categoriesWrapper.querySelectorAll("input[type=checkbox]").forEach(chk => { chk.checked = on; });
    applyFilters();
  });

  catMainHeader.classList.add("expanded"); categoriesWrapper.style.display="block"; catMainArrow.innerHTML="&#9660;";
  catMainHeader.addEventListener("click", ()=>{ const expanded=catMainHeader.classList.toggle("expanded"); categoriesWrapper.style.display = expanded ? "block" : "none"; catMainArrow.innerHTML="&#9660;"; if(!expanded) catMainArrow.innerHTML="&#9654;"; });

  // HORARIO
  const hourBlock = document.createElement("div"); hourBlock.className="category-block";
  const hourHeader = document.createElement("div"); hourHeader.className="category-header";
  const hourTitleW = document.createElement("div"); hourTitleW.className="title";
  const hourEmoji = document.createElement("span"); hourEmoji.textContent="⏰";
  const hourName = document.createElement("span"); hourName.textContent="HORARIO";
  hourTitleW.appendChild(hourEmoji); hourTitleW.appendChild(hourName);
  const hourArrow = document.createElement("div"); hourArrow.className="toggle-arrow"; hourArrow.innerHTML="&#9654;";
  hourHeader.appendChild(hourTitleW); hourHeader.appendChild(hourArrow);
  hourBlock.appendChild(hourHeader);

  const hoursWrapper = document.createElement("div"); hoursWrapper.className="subcat-list"; hoursWrapper.style.paddingLeft="12px"; hoursWrapper.id="hours-wrapper";

  const dayRow = document.createElement("div"); dayRow.className="checkbox-wrapper"; dayRow.style.alignItems="center";
  const dayLbl = document.createElement("label"); dayLbl.textContent = "Día:"; dayLbl.style.minWidth = "40px";
  const daySelect = document.createElement("select"); daySelect.id="day-select"; daySelect.style.flex="1";
  ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"].forEach((d,i)=>{ const opt=document.createElement("option"); opt.value=String(i); opt.textContent=d; daySelect.appendChild(opt); });
  dayRow.appendChild(dayLbl); dayRow.appendChild(daySelect);

  const timeRow = document.createElement("div"); timeRow.className="checkbox-wrapper"; timeRow.style.alignItems="center";
  const timeLbl = document.createElement("label"); timeLbl.textContent = "Hora:"; timeLbl.style.minWidth = "40px";
  const timeInput = document.createElement("input"); timeInput.type="time"; timeInput.id="time-input"; timeInput.step=60;
  timeRow.appendChild(timeLbl); timeRow.appendChild(timeInput);

  const actionsRow = document.createElement("div"); actionsRow.style.display="flex"; actionsRow.style.gap="6px"; actionsRow.style.marginTop="6px";
  const applyBtn = document.createElement("button"); applyBtn.textContent="Aplicar horario";
  const clearBtn = document.createElement("button"); clearBtn.textContent="Quitar filtro";
  actionsRow.appendChild(applyBtn); actionsRow.appendChild(clearBtn);

  const hint = document.createElement("div"); hint.style.fontSize="12px"; hint.style.color="#666"; hint.textContent = "Elige día y hora y pulsa “Aplicar horario”.";
  hoursWrapper.appendChild(dayRow); hoursWrapper.appendChild(timeRow); hoursWrapper.appendChild(actionsRow); hoursWrapper.appendChild(hint);

  hourBlock.appendChild(hoursWrapper);
  container.appendChild(hourBlock);

  hourHeader.addEventListener("click", (e)=>{
    e.stopPropagation();
    const expanded=hourHeader.classList.toggle("expanded");
    hoursWrapper.style.display = expanded ? "block" : "none";
    hourArrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
  });

  (function setDefaultDayTime(){
    const now=new Date();
    daySelect.value=String((now.getDay()+6)%7);
    const hh=String(now.getHours()).padStart(2,"0");
    const mm=String(now.getMinutes()).padStart(2,"0");
    timeInput.value = `${hh}:${mm}`;
  })();

  let hourFilterActive = false;

  applyBtn.addEventListener("click", ()=>{
    if(!timeInput.value) { alert("Selecciona una hora."); return; }
    hourFilterActive = true;
    applyFilters();
  });
  clearBtn.addEventListener("click", ()=>{
    hourFilterActive = false;
    applyFilters();
  });

  // === Helpers filtros simples (OR interno, AND externo) ===
  function createSimpleFilterBlock(opts){
    const block  = document.createElement("div"); block.className="category-block";
    const header = document.createElement("div"); header.className="category-header";
    const titleW = document.createElement("div"); titleW.className="title";
    const em = document.createElement("span"); em.textContent = opts.emoji || "🗂️";
    const nm = document.createElement("span"); nm.textContent = opts.title;
    titleW.appendChild(em); titleW.appendChild(nm);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleW); header.appendChild(arrow);
    block.appendChild(header);

    const wrapper = document.createElement("div"); wrapper.className="subcat-list"; wrapper.style.paddingLeft="20px";

    (opts.values||[]).forEach(v=>{
      const row = document.createElement("div"); row.className="checkbox-wrapper";
      const chk = document.createElement("input"); chk.type="checkbox"; chk.className="small-checkbox";
      chk.checked = false;
      chk.setAttribute("data-filter", opts.id);
      chk.setAttribute("data-value", String(v));
      const lbl = document.createElement("span"); lbl.textContent = String(v);
      row.appendChild(chk); row.appendChild(lbl);
      wrapper.appendChild(row);
      chk.addEventListener("change", applyFilters);
    });

    block.appendChild(wrapper);
    container.appendChild(block);

    header.addEventListener("click", (e)=>{
      e.stopPropagation();
      const expanded=header.classList.toggle("expanded");
      wrapper.style.display = expanded ? "flex" : "none";
      arrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
    });

    header.classList.add("expanded");
    wrapper.style.display="flex";
    arrow.innerHTML="&#9660;";
  }

  function getActiveValues(filterId){
    const list = Array.from(document.querySelectorAll(`input[data-filter="${filterId}"][type="checkbox"]`));
    return new Set(list.filter(x=>x.checked).map(x=>x.getAttribute("data-value")));
  }

  createSimpleFilterBlock({
    id: "acc_silla",
    emoji: "♿",
    title: "ACCESIBILIDAD SILLA DE RUEDAS",
    values: filtersUnique["acc_silla"] || []
  });
  createSimpleFilterBlock({
    id: "precio",
    emoji: "💶",
    title: "PRECIO",
    values: filtersUnique["precio"] || []
  });

  // ========= UMBRALES: VALORACIÓN y RESEÑAS =========
  let ratingFilterActive = false;
  let reviewsFilterActive = false;

  function createThresholdFilterBlock(opts){
    const block  = document.createElement("div"); block.className="category-block";
    const header = document.createElement("div"); header.className="category-header";
    const titleW = document.createElement("div"); titleW.className="title";
    const em = document.createElement("span"); em.textContent = opts.emoji || "⭐";
    const nm = document.createElement("span"); nm.textContent = opts.title;
    titleW.appendChild(em); titleW.appendChild(nm);
    const arrow = document.createElement("div"); arrow.className="toggle-arrow"; arrow.innerHTML="&#9654;";
    header.appendChild(titleW); header.appendChild(arrow);
    block.appendChild(header);

    const wrapper = document.createElement("div"); wrapper.className="subcat-list"; wrapper.style.paddingLeft="12px";

    const row = document.createElement("div"); row.className="checkbox-wrapper"; row.style.alignItems="center";
    const label = document.createElement("label"); label.textContent = opts.label + ":"; label.style.minWidth="150px";
    const input = document.createElement("input"); input.type="number"; input.id = opts.inputId;
    if (opts.min !== undefined) input.min = String(opts.min);
    if (opts.max !== undefined) input.max = String(opts.max);
    input.step = String(opts.step ?? 1);
    input.style.flex="1";
    if (opts.placeholder) input.placeholder = opts.placeholder;

    row.appendChild(label); row.appendChild(input);

    const actionsRow = document.createElement("div"); actionsRow.style.display="flex"; actionsRow.style.gap="6px"; actionsRow.style.marginTop="6px";
    const applyBtn2 = document.createElement("button"); applyBtn2.textContent="Aplicar " + opts.shortTitle;
    const clearBtn2 = document.createElement("button"); clearBtn2.textContent="Quitar filtro";
    actionsRow.appendChild(applyBtn2); actionsRow.appendChild(clearBtn2);

    const hint = document.createElement("div"); hint.style.fontSize="12px"; hint.style.color="#666"; hint.textContent = opts.hint;

    wrapper.appendChild(row); wrapper.appendChild(actionsRow); wrapper.appendChild(hint);
    block.appendChild(wrapper);
    container.appendChild(block);

    header.addEventListener("click", (e)=>{
      e.stopPropagation();
      const expanded=header.classList.toggle("expanded");
      wrapper.style.display = expanded ? "block" : "none";
      arrow.innerHTML = expanded ? "&#9660;" : "&#9654;";
    });

    header.classList.add("expanded"); wrapper.style.display="block"; arrow.innerHTML="&#9660;";

    applyBtn2.addEventListener("click", ()=>{
      const val = input.value.trim();
      if (val === "") { alert("Introduce un valor numérico."); return; }
      const num = parseFloat(val);
      if (Number.isNaN(num)) { alert("Valor no válido."); return; }
      if (opts.min !== undefined && num < opts.min) { alert("El valor debe ser ≥ " + opts.min + "."); return; }
      if (opts.max !== undefined && num > opts.max) { alert("El valor debe ser ≤ " + opts.max + "."); return; }
      if (opts.type === "rating") ratingFilterActive = true;
      if (opts.type === "reviews") reviewsFilterActive = true;
      applyFilters();
    });
    clearBtn2.addEventListener("click", ()=>{
      if (opts.type === "rating") ratingFilterActive = false;
      if (opts.type === "reviews") reviewsFilterActive = false;
      applyFilters();
    });
  }

  createThresholdFilterBlock({
    type: "rating",
    emoji: "⭐",
    title: "VALORACIÓN",
    shortTitle: "valoración",
    label: "Mínimo (0–5)",
    inputId: "rating-input",
    min: 0, max: 5, step: 0.1,
    placeholder: "p.ej. 2.3",
    hint: "Muestra lugares con valoración mayor o igual al valor."
  });

  createThresholdFilterBlock({
    type: "reviews",
    emoji: "🧮",
    title: "NÚMERO DE RESEÑAS",
    shortTitle: "n.º reseñas",
    label: "Mínimo (≥ 0)",
    inputId: "reviews-input",
    min: 0, step: 1,
    placeholder: "p.ej. 23",
    hint: "Muestra lugares con un número de reseñas mayor o igual al valor."
  });
  // ========= FIN UMBRALES =========

  createSimpleFilterBlock({
    id: "estado",
    emoji: "🏷️",
    title: "ESTADO DE NEGOCIO",
    values: filtersUnique["estado"] || []
  });
  createSimpleFilterBlock({
    id: "reserva",
    emoji: "📅",
    title: "RESERVA POSIBLE",
    values: filtersUnique["reserva"] || []
  });

  document.getElementById("show-all").addEventListener("click", ()=>{
    document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=true; });
    ratingFilterActive = false; reviewsFilterActive = false;
    applyFilters();
  });
  document.getElementById("hide-all").addEventListener("click", ()=>{
    document.querySelectorAll("#categories-container input[type=checkbox]").forEach(chk=>{ chk.checked=false; });
    applyFilters();
  });

  function setMarkerVisible(marker, visible){
    const el = marker._icon; if(el) el.style.display = visible ? "" : "none";
    const sh = marker._shadow; if(sh) sh.style.display = visible ? "" : "none";
  }

  // ✅ Filtrado por TIPOS_TUI (OR entre tipos marcados). Las categorías agrupan casillas.
  function markerMatchesCategoryFilters(marker){
    // 1) Recoger TODOS los subtipos marcados (excepto "(sin subcategoría)")
    const checkedTypeLabels = Array
      .from(document.querySelectorAll(`#categories-container input[type=checkbox][data-sub]`))
      .filter(el => el.checked && el.getAttribute("data-sub") !== "(sin subcategoría)")
      .map(el => el.getAttribute("data-sub"));

    const anyTypeChecked = checkedTypeLabels.length > 0;

    // 2) Conjunto normalizado de subtipos seleccionados
    const checkedTypesNorm = new Set(checkedTypeLabels.map(normJS));

    // 3) Si hay subtipos marcados → OR por tipos normalizados
    if (anyTypeChecked){
      for (const t of marker._typesNorm){
        if (checkedTypesNorm.has(t)) return true;
      }
    }

    // 4) Soporte de "(sin subcategoría)": si está marcada bajo alguna categoría,
    //    mostramos registros que pertenezcan a esa categoría pero no tengan subtipos ahí.
    const checkedEmptyCats = Array
      .from(document.querySelectorAll(`#categories-container input[type=checkbox][data-sub="(sin subcategoría)"]`))
      .filter(el => el.checked)
      .map(el => el.getAttribute("data-main"));

    if (checkedEmptyCats.length){
      for (const c of checkedEmptyCats){
        if (marker._catsEmpty && marker._catsEmpty.has(c)) return true;
      }
    }

    // 5) Si no hay nada marcado en CATEGORÍAS (ni subtipos ni "(sin subcategoría)"), no muestra
    return false;
  }

  // === LÓGICA DE FILTRADO (AND entre bloques de filtros) ===
function applyFilters(){
  const favOn = document.getElementById('fav-main-chk').checked;
  if (favOn){ if(!map.hasLayer(favoritesLayer)) favoritesLayer.addTo(map); }
  else { if(map.hasLayer(favoritesLayer)) map.removeLayer(favoritesLayer); }

  // horario activado por botón
  const daySel   = document.getElementById("day-select")?.value ?? "";
  const timeSel  = document.getElementById("time-input")?.value ?? "";
  const timeFilterOn = (typeof hourFilterActive !== "undefined") && hourFilterActive && daySel!=="" && timeSel!=="";
  let dow=null, minute=null;
  if(timeFilterOn){ dow=parseInt(daySel,10); const [HH,MM]=timeSel.split(":").map(x=>parseInt(x,10)); minute=HH*60+MM; }

  // filtros simples
  const accSet     = getActiveValues("acc_silla");
  const precioSet  = getActiveValues("precio");
  const estadoSet  = getActiveValues("estado");
  const reservaSet = getActiveValues("reserva");

  const accActive     = accSet.size > 0;
  const precioActive  = precioSet.size > 0;
  const estadoActive  = estadoSet.size > 0;
  const reservaActive = reservaSet.size > 0;

  // filtros de umbral
  const ratingInput = document.getElementById("rating-input");
  const reviewsInput = document.getElementById("reviews-input");
  const ratingFilterOn = ratingFilterActive && ratingInput && ratingInput.value.trim() !== "";
  const reviewsFilterOn = reviewsFilterActive && reviewsInput && reviewsInput.value.trim() !== "";
  const ratingThreshold = ratingFilterOn ? parseFloat(ratingInput.value) : null;
  const reviewsThreshold = reviewsFilterOn ? parseFloat(reviewsInput.value) : null;

  // === Filtro CHATBOT (prioritario) ===
  const chatOn = document.getElementById('chat-main-chk')?.checked;
  const hasChatPins = chatFilter && chatFilter.placeIds && chatFilter.placeIds.size > 0;
  const hasChatNames = chatFilter && Array.isArray(chatFilter.names) && chatFilter.names.length > 0;

  for (const marker of allMarkers){
    // 0) CHAT
    let passesChat = true;
    if (chatOn && (hasChatPins || hasChatNames)){
      passesChat = false;

      // a) por place_id (estricto)
      const pid = (marker._placeId || "").trim();
      if (pid && hasChatPins && chatFilter.placeIds.has(pid)){
        passesChat = true;
      }

      // b) por nombre (fallback)
      if (!passesChat && hasChatNames){
        const nm = (marker._rec?.name || "").toLowerCase();
        for (const q of chatFilter.names){
          if (nm.includes(String(q).toLowerCase())) { passesChat = true; break; }
        }
      }
    }

    if (!passesChat){
      setMarkerVisible(marker, false);
      continue;
    }

    // 1) Categorías / Tipos
    let visible = markerMatchesCategoryFilters(marker);

    // 2) Horario
    if (visible && timeFilterOn){
      let sched = scheduleCache.get(marker._recId);
      if(!sched){ sched = parseSchedule(marker._rec.horario || ""); scheduleCache.set(marker._recId, sched); }
      visible = isOpenAtSchedule(sched, dow, minute);
    }

    // 3) Filtros simples
    if (visible && accActive){
      const v = String(marker._rec.accesibilidad_silla_ruedas);
      visible = accSet.has(v);
    }
    if (visible && precioActive){
      const v = String(marker._rec.precio);
      visible = precioSet.has(v);
    }
    if (visible && estadoActive){
      const v = String(marker._rec.estado_negocio);
      visible = estadoSet.has(v);
    }
    if (visible && reservaActive){
      const v = String(marker._rec.reserva_posible);
      visible = reservaSet.has(v);
    }

    // 4) Umbrales
    if (visible && ratingFilterOn){
      const v = Number(marker._rec.rating);
      visible = !Number.isNaN(v) && v >= ratingThreshold;
    }
    if (visible && reviewsFilterOn){
      const v = Number(marker._rec.total_reviews);
      visible = !Number.isNaN(v) && v >= reviewsThreshold;
    }

    setMarkerVisible(marker, visible);
    marker._updateEmoji();
  }
}

  const panelToggle = document.getElementById("panel-toggle");
  const filterBody  = document.getElementById("filter-body");
  let collapsed = false;
  panelToggle.addEventListener("click", ()=>{ collapsed = !collapsed; filterBody.style.display = collapsed ? "none" : "flex"; panelToggle.innerHTML = collapsed ? "&#9654;" : "&#9660;"; });

  applyFilters();




// === Geolocalización robusta + Botón “Ubicarme” (🧍) ===
// Colocar dentro de withMap((map)=>{ ... })

// Capa para el pin y círculo de usuario
const userLayer = L.layerGroup().addTo(map);

// Mejor fix actual y control de centrado
let bestFix = null;        // {lat, lon, accuracy, ts, source}
let hasCentered = false;   // evita re-centrados molestos

// Red de seguridad (bbox aprox Madrid, ajusta si quieres)
const MADRID_BBOX = { south: 40.1, west: -3.9, north: 40.6, east: -3.4 };

// Reglas de frescura/precisión
const MAX_AGE_MS = 5 * 60 * 1000;   // 5 min
const MAX_BAD_ACCURACY = 5000;      // >5km suele ser IP
const GOOD_ACCURACY = 150;          // centrado “automático” si ≤150 m

function inMadridBbox(lat, lon){
  return lat >= MADRID_BBOX.south && lat <= MADRID_BBOX.north &&
         lon >= MADRID_BBOX.west  && lon <= MADRID_BBOX.east;
}

function putUserPin(lat, lon, accuracy){
  userLayer.clearLayers();
  const icon = L.divIcon({
    className: "leaflet-div-icon user-pin",
    html: "📌",
    iconSize: [28, 28],
    iconAnchor: [14, 28]
  });
  L.marker([lat, lon], { icon, zIndexOffset: 1000 })
    .addTo(userLayer)
    .bindPopup("Estás aquí");

  if (Number.isFinite(accuracy) && accuracy > 0) {
    L.circle([lat, lon], { radius: accuracy }).addTo(userLayer);
  }
}

function maybeCenter(lat, lon, accuracy){
  if (hasCentered) return;
  if (!inMadridBbox(lat, lon)) return; // evita “saltos” a IP lejana
  if (accuracy == null || accuracy <= GOOD_ACCURACY) {
    try { map.setView([lat, lon], Math.max(map.getZoom() || 0, 14)); hasCentered = true; } catch {}
  }
}

function isFresh(ts){ return (Date.now() - ts) <= MAX_AGE_MS; }

function isAcceptableFix(fix){
  const okAcc = (fix.accuracy == null) || (fix.accuracy <= MAX_BAD_ACCURACY);
  const okAge = isFresh(fix.ts);
  const okBox = inMadridBbox(fix.lat, fix.lon);
  return okAcc && okAge && okBox;
}

function considerFix(fix){
  if (!isAcceptableFix(fix)) return false;

  const beats = () => {
    if (!bestFix) return true;
    const a = fix.accuracy ?? 99999;
    const b = bestFix.accuracy ?? 99999;
    // mejor si: precisión claramente mejor, o similar pero más reciente
    return (a < b - 50) || (Math.abs(a - b) <= 50 && fix.ts > bestFix.ts);
  };

  if (beats()) {
    bestFix = fix;
    putUserPin(fix.lat, fix.lon, fix.accuracy);
    maybeCenter(fix.lat, fix.lon, fix.accuracy);
    try { localStorage.setItem("client_coords", JSON.stringify(fix)); } catch {}
    return true;
  }
  return false;
}

// === 1) Primero: geolocalización del navegador (mejor calidad)
if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(
    pos => {
      const { latitude, longitude, accuracy } = pos.coords;
      considerFix({ lat: latitude, lon: longitude, accuracy, ts: Date.now(), source: "browser" });
    },
    () => { /* silencio; seguiremos con otras fuentes */ },
    { enableHighAccuracy: true, maximumAge: 0, timeout: 8000 }
  );

  // Suscripción a mejoras (si pasa de IP → Wi‑Fi)
  const watchId = navigator.geolocation.watchPosition(
    pos => {
      const { latitude, longitude, accuracy } = pos.coords;
      considerFix({ lat: latitude, lon: longitude, accuracy, ts: Date.now(), source: "browser-watch" });
    },
    () => {},
    { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
  );
  // Si quieres, limpia más tarde: navigator.geolocation.clearWatch(watchId);
}

// === 2) Respaldo: /coords del servidor (¡suele ser IP! → se filtra)
fetch("/coords")
  .then(r => r.json())
  .then(c => {
    if (typeof c.lat === "number" && typeof c.lon === "number") {
      considerFix({ lat: c.lat, lon: c.lon, accuracy: c.accuracy ?? 10000, ts: Date.now(), source: "server" });
    }
  })
  .catch(()=>{});

// === 3) Último respaldo: localStorage (si no es viejo)
try {
  const s = localStorage.getItem("client_coords");
  if (s) {
    const o = JSON.parse(s);
    if (o && typeof o.lat === "number" && typeof o.lon === "number") {
      considerFix({ lat: +o.lat, lon: +o.lon, accuracy: +o.accuracy || 99999, ts: +o.ts || 0, source: "cache" });
    }
  }
} catch {}

// === Botón “Ubicarme” (🧍) — fuerza re-localización y centrado ===
(function addLocateControl(){
  function setLocating(v){
    const el = document.querySelector('.locate-btn');
    if (el){
      el.classList.toggle('loading', !!v);
      el.innerHTML = v ? "⌛" : "🧍";
      el.title = v ? "Buscando tu ubicación..." : "Ubicarme";
    }
  }

  // Solicita ubicación fresca y centra el mapa
  window.requestFreshLocation = function(forceCenter=true){
    try { localStorage.removeItem("client_coords"); } catch {}
    setLocating(true);

    const fallbackToServer = () => {
      fetch("/coords").then(r=>r.json()).then(c=>{
        setLocating(false);
        if (typeof c.lat === "number" && typeof c.lon === "number"){
          if (forceCenter) hasCentered = false;
          considerFix({ lat:c.lat, lon:c.lon, accuracy: c.accuracy ?? 10000, ts: Date.now(), source:"server-manual" });
          if (forceCenter) { try { map.setView([c.lat, c.lon], Math.max(map.getZoom()||0, 14)); hasCentered = true; } catch {} }
        } else {
          alert("No se pudo obtener tu ubicación. Revisa permisos del navegador.");
        }
      }).catch(()=>{ setLocating(false); alert("No se pudo obtener tu ubicación. Revisa permisos del navegador."); });
    };

    if (!navigator.geolocation) { setLocating(false); return fallbackToServer(); }

    navigator.geolocation.getCurrentPosition(
      pos => {
        setLocating(false);
        const { latitude, longitude, accuracy } = pos.coords;
        if (forceCenter) hasCentered = false;
        considerFix({ lat: latitude, lon: longitude, accuracy, ts: Date.now(), source: "manual" });
        try { map.setView([latitude, longitude], Math.max(map.getZoom()||0, 16)); hasCentered = true; } catch {}
      },
      () => { setLocating(false); fallbackToServer(); },
      { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 }
    );
  };

  // Control Leaflet arriba-izquierda
  const LocateControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function(){
      const a = L.DomUtil.create('a', 'leaflet-bar locate-btn');
      a.href = '#'; a.title = 'Ubicarme'; a.innerHTML = '🧍';
      L.DomEvent.disableClickPropagation(a);
      L.DomEvent.on(a, 'click', (e)=>{ L.DomEvent.preventDefault(e); requestFreshLocation(true); });
      return a;
    }
  });
  map.addControl(new LocateControl());
})();





  
 function hydrateNamesFromMarkersByIds(ids){
  const out = [];
  const idSet = new Set(ids || []);
  const seen = new Set();
  for (const mk of allMarkers){
    const pid = (mk._placeId || "").trim();
    if (!pid || !idSet.has(pid)) continue;
    const nm = (mk._rec?.name || "").trim();
    if (!nm || seen.has(nm.toLowerCase())) continue;
    seen.add(nm.toLowerCase());
    out.push(nm);
  }
  return out;
}
window.addEventListener("message", (ev)=>{
  const data = ev?.data || {};
  if (data.type === "SCT_APPLY_CHAT_FILTER"){
    const ids = Array.isArray(data.place_ids) ? data.place_ids : [];
    let names = Array.isArray(data.names) ? data.names : [];

    chatFilter.placeIds = new Set(ids.filter(Boolean));
    if (!names.length && ids.length) names = hydrateNamesFromMarkersByIds(ids);
    chatFilter.names = names.filter(Boolean);
    chatFilter.active = (chatFilter.placeIds.size > 0) || (chatFilter.names.length > 0);

    // activar el checkbox al llegar un filtro nuevo
    const chk = document.getElementById('chat-main-chk');
    if (chk) chk.checked = true;

    renderChatPanel();

    // ⬇️ NUEVO: toggle modo estricto y carga por place_id
    strictChatMode = chatFilter.placeIds.size > 0;

    (async () => {
      if (strictChatMode) {
        await fetchStrictByPlaceIds([...chatFilter.placeIds]);
        // centra a los resultados estrictos
        const targets = [];
        for (const mk of allMarkers){
          if (chatFilter.placeIds.has(mk._placeId)) targets.push(mk.getLatLng());
        }
        if (targets.length){
          try { map.fitBounds(L.latLngBounds(targets), { padding:[50,50] }); } catch {}
        }
      } else {
        // sale del modo estricto → vuelve al incremental
        await fetchAndRender();
      }
      applyFilters();
    })();
  }
});


});
</script>
"""

m.get_root().html.add_child(folium.Element(js))
m.save(HTML_OUT)
print("✔️  Mapa (carga incremental y filtros por TIPOS_TUI + favoritos backend por tui_id) guardado en", HTML_OUT)
