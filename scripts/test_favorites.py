import os
from dotenv import load_dotenv

load_dotenv()

from app import app, db
from models import User, Tui, Favorite
from werkzeug.security import generate_password_hash


def ensure_user(username="testuser", email="test@example.com", password="test1234"):
    u = User.query.filter((User.username == username) | (User.email == email)).first()
    if not u:
        u = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(u)
        db.session.commit()
    return u, password


def main():
    with app.app_context():
        # 1) Asegurar usuario
        user, pwd = ensure_user()
        # 2) Asegurar que hay TUI
        tui = Tui.query.first()
        if not tui:
            print("❌ No hay registros en Tui. Ejecuta: python scripts\\load_tui.py")
            return

        print(f"Usando TUI: id={tui.id} place_id={tui.place_id} nombre={tui.NOMBRE_TUI}")

        # 3) Cliente de pruebas con sesión (login)
        with app.test_client() as c:
            # Login
            r = c.post("/login", data={"username": user.username, "password": pwd}, follow_redirects=True)
            print("Login status:", r.status_code)

            # Estado inicial
            r = c.get("/api/favorites")
            print("GET /api/favorites =>", r.status_code, r.json)

            # 4) Toggle favorito (añadir)
            payload = {
                "tui_id": tui.id,
                "place_id": tui.place_id,
                "name": tui.NOMBRE_TUI,
                "lat": tui.LATITUD_TUI,
                "lon": tui.LONGITUD_TUI,
            }
            r = c.post("/api/favorites/toggle", json=payload)
            print("POST /api/favorites/toggle (add) =>", r.status_code, r.json)

            # Comprobar
            r = c.get("/api/favorites")
            print("GET /api/favorites =>", r.status_code, r.json)

            # 5) Toggle favorito (quitar)
            r = c.post("/api/favorites/toggle", json=payload)
            print("POST /api/favorites/toggle (remove) =>", r.status_code, r.json)

            r = c.get("/api/favorites")
            print("GET /api/favorites =>", r.status_code, r.json)


if __name__ == "__main__":
    main()

