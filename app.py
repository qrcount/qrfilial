# app.py
import os, io, re, time, threading
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ---- QR / Imágenes ----
from PIL import Image, ImageOps
import numpy as np
import cv2

# ---- DB opcional ----
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ---- Pandas para archivo (CSV/XLSX) ----
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# --------------------------------------------------------------------------------
# Config por entorno
# --------------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "FILE").upper()  # "DB" o "FILE"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ccb.db")
PROMOTORES_TABLE = os.getenv("PROMOTORES_TABLE", "promotores")
PROMOTORES_OP_COLUMN = os.getenv("PROMOTORES_OP_COLUMN", "op")

# Para DATA_SOURCE=FILE
DATA_FILE = os.getenv("DATA_FILE", "promotores.xlsx")  # .csv o .xlsx
SHEET_NAME = os.getenv("SHEET_NAME", None)             # si es .xlsx
HEADER_ROW = int(os.getenv("HEADER_ROW", "1"))         # 0 = 1ra fila, 1 = 2da fila...
ENCODING = os.getenv("CSV_ENCODING", "utf-8")

DEBUG_FLAG = os.getenv("DEBUG", "0") == "1"
PORT = int(os.getenv("PORT", "8000"))

# --------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# DB engine
engine: Optional[Engine] = None
if DATA_SOURCE == "DB":
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --------------------------------------------------------------------------------
# Utilidades comunes
# --------------------------------------------------------------------------------
def ok(data: Any = None, **extra):
    base = {"ok": True}
    if data is not None:
        base["data"] = data
    base.update(extra)
    return jsonify(base)

def fail(msg: str, code: int = 400, **extra):
    base = {"ok": False, "error": msg}
    base.update(extra)
    return jsonify(base), code

def normalize_op(s: str) -> str:
    if s is None:
        return ""
    s2 = str(s).strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    # solo dígitos -> anteponer EX
    if re.fullmatch(r"\d+", s2 or ""):
        return "EX" + s2
    # si contiene dígitos pero no empieza por EX, conservar dígitos
    if not s2.startswith("EX") and re.search(r"\d", s2):
        digits = "".join(ch for ch in s2 if ch.isdigit())
        return "EX" + digits if digits else s2
    return s2

def sql_normalize_expr(column: str) -> str:
    return f"UPPER(REPLACE(REPLACE(REPLACE({column}, ' ', ''), '-', ''), '_', ''))"

# --------------------------------------------------------------------------------
# Modelo de salida
# --------------------------------------------------------------------------------
@dataclass
class PromotorRow:
    op: str
    cliente: str | None = None
    nombre: str | None = None
    descripcion: str | None = None
    cantidad: Any | None = None
    talla: str | None = None
    qr: str | None = None
    enlace: str | None = None

    @classmethod
    def from_mapping(cls, r: Dict[str, Any]):
        # Soporta columnas en distintos idiomas/alias
        return cls(
            op=r.get("op"),
            cliente=r.get("cliente") or r.get("empresa") or r.get("empresa_op") or r.get("empresa op"),
            nombre=r.get("nombre"),
            descripcion=r.get("descripcion") or r.get("tela"),
            cantidad=r.get("cantidad"),
            talla=r.get("talla"),
            qr=r.get("qr"),
            enlace=r.get("enlace") or r.get("link"),
        )

# --------------------------------------------------------------------------------
# Capa FILE: carga y cache del dataset
# --------------------------------------------------------------------------------
_df_cache: Optional[pd.DataFrame] = None
_df_mtime: Optional[float] = None
_df_lock = threading.Lock()

EXPECTED_COLS = ["ITEM", "EMPRESA", "OP", "NOMBRE", "CANTIDAD", "TELA", "TALLA", "LINK"]

def _clean_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c = str(c)
        if c.lower().startswith("unnamed"):
            out.append("")  # descartar luego
        else:
            c = c.strip().upper()
            c = re.sub(r"\s+", " ", c)
            out.append(c)
    return out

def _load_file_df(force: bool = False) -> pd.DataFrame:
    if not HAS_PANDAS:
        raise RuntimeError("Pandas no está instalado en el entorno.")
    global _df_cache, _df_mtime
    with _df_lock:
        try:
            mtime = os.path.getmtime(DATA_FILE)
        except FileNotFoundError:
            raise RuntimeError(f"No se encuentra DATA_FILE='{DATA_FILE}'")
        if (not force) and _df_cache is not None and _df_mtime == mtime:
            return _df_cache

        # Leer CSV/XLSX con HEADER_ROW configurable
        if DATA_FILE.lower().endswith(".csv"):
            df = pd.read_csv(DATA_FILE, header=HEADER_ROW, encoding=ENCODING)
        else:
            # XLSX
            read_kwargs = {"header": HEADER_ROW}
            if SHEET_NAME:
                read_kwargs["sheet_name"] = SHEET_NAME
            df = pd.read_excel(DATA_FILE, **read_kwargs)

        # Normalizar encabezados
        df.columns = _clean_columns(list(df.columns))
        # Quitar columnas vacías
        df = df.loc[:, [c for c in df.columns if c]]

        # Renombrar a claves estándar si vienen alternativas
        rename_map = {
            "EMPRESA OP": "EMPRESA",
            "EMPRESA_OP": "EMPRESA",
        }
        df = df.rename(columns=rename_map)

        # Si falta OP porque el header estaba una fila arriba, intenta autodetectar
        if "OP" not in df.columns:
            # Heurística: busca columna que contenga valores tipo EX-#### o EX####
            for c in df.columns:
                sample = str(df[c].astype(str).head(20).tolist())
                if re.search(r"EX[-\s_]?\d{3,}", sample, re.IGNORECASE):
                    df = df.rename(columns={c: "OP"})
                    break

        # Normalizar OP
        if "OP" in df.columns:
            df["OP"] = df["OP"].map(normalize_op)
        else:
            raise RuntimeError("No se encontró la columna 'OP' después de limpiar encabezados.")

        # Mapear nombres a los esperados por el API
        col_map = {
            "EMPRESA": "cliente",
            "OP": "op",
            "NOMBRE": "nombre",
            "CANTIDAD": "cantidad",
            "TELA": "descripcion",
            "TALLA": "talla",
            "LINK": "enlace",
            "QR": "qr",
        }
        # Mantén sólo columnas conocidas si existen
        keep = [k for k in ["op","cliente","nombre","descripcion","cantidad","talla","qr","enlace"]
                if (k in df.columns) or (k in col_map.keys())]
        # Renombra
        for src, dst in col_map.items():
            if src in df.columns:
                df = df.rename(columns={src: dst})

        # Quitar filas sin OP
        df = df[df["op"].notna() & (df["op"].astype(str).str.len() > 0)]

        _df_cache = df.copy()
        _df_mtime = mtime
        return _df_cache

def _search_in_file(op_raw: str, page: int, limit: int) -> Dict[str, Any]:
    df = _load_file_df()
    needle = normalize_op(op_raw)

    # Coincidencia exacta por OP normalizado
    hits = df[df["op"] == needle]

    # Si no hay exacta, buscar aproximadas (con guion/espacio/números)
    if hits.empty:
        variants = {needle, f"EX-{needle[2:]}", f"EX {needle[2:]}", needle[2:]}
        hits = df[df["op"].isin({normalize_op(v) for v in variants})]

    total = len(hits)
    start = (page - 1) * limit
    end = start + limit
    rows = hits.iloc[start:end].to_dict(orient="records")
    data = [PromotorRow.from_mapping(r).__dict__ for r in rows]
    return {"records": data, "page": page, "limit": limit, "count": total}

# --------------------------------------------------------------------------------
# Capa DB: búsqueda
# --------------------------------------------------------------------------------
def _search_in_db(op_raw: str, page: int, limit: int) -> Dict[str, Any]:
    if engine is None:
        raise RuntimeError("Motor de base de datos no inicializado.")
    needle = normalize_op(op_raw)
    offset = (page - 1) * limit
    norm_expr = sql_normalize_expr(PROMOTORES_OP_COLUMN)

    v_dash  = "EX-" + needle[2:] if needle.startswith("EX") else needle
    v_space = "EX " + needle[2:] if needle.startswith("EX") else needle
    v_nod   = needle

    sql_ilike = f"""
        SELECT *
        FROM {PROMOTORES_TABLE}
        WHERE {norm_expr} = :needle
           OR {PROMOTORES_OP_COLUMN} ILIKE :v1
           OR {PROMOTORES_OP_COLUMN} ILIKE :v2
           OR {PROMOTORES_OP_COLUMN} ILIKE :v3
        ORDER BY {PROMOTORES_OP_COLUMN}
        LIMIT :limit OFFSET :offset
    """
    sql_like_upper = f"""
        SELECT *
        FROM {PROMOTORES_TABLE}
        WHERE {norm_expr} = :needle
           OR UPPER({PROMOTORES_OP_COLUMN}) LIKE :v1
           OR UPPER({PROMOTORES_OP_COLUMN}) LIKE :v2
           OR UPPER({PROMOTORES_OP_COLUMN}) LIKE :v3
        ORDER BY {PROMOTORES_OP_COLUMN}
        LIMIT :limit OFFSET :offset
    """

    with engine.connect() as conn:
        try:
            rows = conn.execute(
                text(sql_ilike),
                {"needle": needle,
                 "v1": f"%{v_dash}%",
                 "v2": f"%{v_space}%",
                 "v3": f"%{v_nod}%",
                 "limit": limit, "offset": offset}
            ).mappings().all()
        except SQLAlchemyError:
            rows = conn.execute(
                text(sql_like_upper),
                {"needle": needle,
                 "v1": f"%{v_dash.upper()}%",
                 "v2": f"%{v_space.upper()}%",
                 "v3": f"%{v_nod.upper()}%",
                 "limit": limit, "offset": offset}
            ).mappings().all()

    data = [PromotorRow.from_mapping(dict(r)).__dict__ for r in rows]
    return {"records": data, "page": page, "limit": limit, "count": len(data)}

# --------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("ccb.html")

@app.get("/healthz")
def health():
    info = {"source": DATA_SOURCE}
    try:
        if DATA_SOURCE == "DB":
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            info["db"] = "ok"
        else:
            if not HAS_PANDAS:
                return fail("Pandas no instalado para modo FILE", 500)
            _load_file_df()
            info["file"] = DATA_FILE
            info["header_row"] = HEADER_ROW
            info["sheet_name"] = SHEET_NAME
            info["status"] = "ok"
        return ok(info)
    except Exception as e:
        return fail(f"health_error: {e}", 500)

@app.get("/api/promotores")
def api_promotores():
    op_raw = request.args.get("op", "", type=str)
    page = max(1, request.args.get("page", 1, type=int))
    limit = min(200, max(1, request.args.get("limit", 50, type=int)))
    if not op_raw:
        return fail("Falta parámetro 'op'.")

    try:
        if DATA_SOURCE == "DB":
            payload = _search_in_db(op_raw, page, limit)
        else:
            payload = _search_in_file(op_raw, page, limit)
        return ok(payload)
    except Exception as e:
        return fail(f"search_error: {e}", 500)

@app.get("/api/promotores/_debug")
def api_promotores_debug():
    op_raw = request.args.get("op", "", type=str)
    needle = normalize_op(op_raw)
    info = {
        "op_raw": op_raw,
        "needle_normalizado": needle,
        "source": DATA_SOURCE,
    }
    try:
        if DATA_SOURCE == "DB":
            info.update({
                "db_url": DATABASE_URL,
                "tabla": PROMOTORES_TABLE,
                "columna_op": PROMOTORES_OP_COLUMN,
            })
            with engine.connect() as c:
                try:
                    n = c.execute(text(f"SELECT COUNT(1) as n FROM {PROMOTORES_TABLE}")).mappings().first()
                    info["tabla_count"] = int(n["n"]) if n else 0
                except Exception as e:
                    info["tabla_error"] = str(e)
        else:
            df = _load_file_df()
            info.update({
                "file": DATA_FILE,
                "sheet": SHEET_NAME,
                "header_row": HEADER_ROW,
                "cols_detectadas": list(df.columns),
                "ejemplo": df.head(3).to_dict(orient="records"),
                "total_rows": int(len(df)),
            })
        return ok(info)
    except Exception as e:
        return fail(f"debug_error: {e}", 500)

# ---- QR (igual que antes) ----
try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except Exception:
    HAS_ZBAR = False

def preprocess(img_pil: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img_pil.convert("RGB"))

def _pil_to_cv(img_pil: Image.Image):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def _variants(img_rgb: np.ndarray) -> List[np.ndarray]:
    out = []
    base = img_rgb.copy()
    h, w = base.shape[:2]
    target = 2000
    scale = target / max(h, w)
    if scale < 0.5:
        base = cv2.resize(base, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    elif scale > 1.2:
        base = cv2.resize(base, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    thr_otsu   = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thr_otsu_i = cv2.bitwise_not(thr_otsu)
    adap       = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 2)
    adap_i     = cv2.bitwise_not(adap)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    open1      = cv2.morphologyEx(thr_otsu, cv2.MORPH_OPEN, k, iterations=1)
    open1_i    = cv2.bitwise_not(open1)
    up2  = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up3  = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    up2b = cv2.adaptiveThreshold(up2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    up3b = cv2.adaptiveThreshold(up3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    variants = [gray, clahe, thr_otsu, thr_otsu_i, adap, adap_i, open1, open1_i, up2, up3, up2b, up3b]
    for v in variants:
        out.extend([v,
                    cv2.rotate(v, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(v, cv2.ROTATE_180),
                    cv2.rotate(v, cv2.ROTATE_90_COUNTERCLOCKWISE)])
    return out

def _try_opencv(img: np.ndarray) -> List[str]:
    det = cv2.QRCodeDetector()
    data, pts, _ = det.detectAndDecode(img)
    return [data] if data else []

def _try_zbar(img: np.ndarray) -> List[str]:
    if not HAS_ZBAR:
        return []
    from PIL import Image as _Image
    pil_img = _Image.fromarray(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = zbar_decode(pil_img)
    return [r.data.decode('utf-8', errors='replace') for r in res]

def decode_qr_strong(img_pil: Image.Image) -> List[str]:
    img_cv = _pil_to_cv(img_pil)
    for idx, v in enumerate(_variants(img_cv)):
        r1 = _try_opencv(v)
        if r1:
            app.logger.info(f"[QR] OpenCV OK en variante {idx}")
            return r1
        r2 = _try_zbar(v)
        if r2:
            app.logger.info(f"[QR] ZBar OK en variante {idx}")
            return r2
    app.logger.warning("[QR] Ninguna variante funcionó")
    return []

@app.post("/upload-qr")
def upload_qr():
    if "file" not in request.files:
        return fail("No se envió archivo.", 400)
    f = request.files["file"]
    if not f or f.filename == "":
        return fail("Archivo no válido.", 400)
    try:
        img = Image.open(io.BytesIO(f.read()))
        img = preprocess(img)
        results = decode_qr_strong(img)
        if not results:
            return fail("No se pudo leer un QR en la imagen seleccionada.", 422)
        return ok({"qr": results[0], "all": results})
    except Exception as e:
        return fail(f"Error procesando imagen: {e}", 500)

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_FLAG)
