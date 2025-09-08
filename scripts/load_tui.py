# scripts/load_tui.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from app import app
from models import db, Tui

def upsert_records(records: list[dict]):
    """Upsert por place_id. Sin limpiezas ni conversión de tipos."""
    if not records:
        return
    dialect = db.engine.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(Tui.__table__).values(records)
        # Actualiza TODO salvo la PK y la clave única
        cols_to_update = [c.name for c in Tui.__table__.columns if c.name not in ("id", "place_id")]
        stmt = stmt.on_conflict_do_update(
            index_elements=["place_id"],
            set_={c: getattr(stmt.excluded, c) for c in cols_to_update}
        )
        db.session.execute(stmt)
        db.session.commit()
    else:
        # Fallback genérico (SQLite, etc.): merge fila a fila
        # (Sigue siendo "subir tal cual", pero sin ON CONFLICT nativo)
        for rec in records:
            db.session.merge(Tui(**rec))
        db.session.commit()

def main():
    csv_path = ROOT / "data" / "BBDD_TUI.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el CSV en: {csv_path}")

    with app.app_context():
        # Columnas reales del modelo (para descartar extras del CSV)
        model_cols = {c.name for c in Tui.__table__.columns}

        processed = 0
        for chunk in pd.read_csv(csv_path, chunksize=2000):  # sin dtype=str → deja que Pandas infiera
            # Renombra PLACE_ID → place_id (clave única en el modelo)
            if "PLACE_ID" in chunk.columns and "place_id" not in chunk.columns:
                chunk = chunk.rename(columns={"PLACE_ID": "place_id"})

            # Reemplaza NaN por None (no es "limpieza", solo nulifica)
            chunk = chunk.where(pd.notnull(chunk), None)

            records = []
            for rec in chunk.to_dict(orient="records"):
                # Quédate solo con las columnas que existen en Tui
                rec = {k: v for k, v in rec.items() if k in model_cols}
                # place_id es obligatorio
                if not rec.get("place_id"):
                    continue
                records.append(rec)

            upsert_records(records)
            processed += len(records)
            print(f"Procesados: {processed}")

        print(f"Carga completada ✅  Total procesados: {processed}")

if __name__ == "__main__":
    main()
