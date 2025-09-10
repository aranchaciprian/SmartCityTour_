# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "usuarios"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Tui(db.Model):
    __tablename__ = "tui"
    id = db.Column(db.Integer, primary_key=True)

    # Por si algún ID crece en el futuro
    place_id = db.Column(db.String(255), unique=True, index=True)

    CATEGORIA_TUI = db.Column(db.Text)
    TIPOS_TUI = db.Column(db.Text)
    NOMBRE_TUI = db.Column(db.String(255), index=True)

    ACCESIBILIDAS_SILLA_RUEDAS = db.Column(db.String(50))
    RESERVA_POSIBLE = db.Column(db.String(50))

    # << Cambios clave: pasamos a TEXT para soportar largos >>
    DIRECCION = db.Column(db.Text)
    WEBSITE = db.Column(db.Text)
    URL = db.Column(db.Text)

    LATITUD_TUI = db.Column(db.Float, index=True)
    LONGITUD_TUI = db.Column(db.Float, index=True)
    RATING_TUI = db.Column(db.Float)

    TOTAL_VALORACIONES_TUI = db.Column(db.Integer)
    PRECIO = db.Column(db.String(50))
    ESTADO_NEGOCIO = db.Column(db.String(50))
    HORARIO = db.Column(db.Text)
    TELEFONO = db.Column(db.String(50))

class Favorite(db.Model):
    __tablename__ = "favoritos"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False, index=True)
    tui_id = db.Column(db.Integer, db.ForeignKey("tui.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "tui_id", name="uq_user_tui"),)

class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False, index=True)

    # Identificador lógico que te envía el front/iframe del widget
    conversation_id = db.Column(db.String(128), nullable=False, index=True)

    # Un “título” opcional (por ejemplo, el primer prompt del usuario recortado)
    title = db.Column(db.String(255))

    # Guardamos el snapshot completo de la conversación (lista de mensajes role/content)
    messages = db.Column(db.JSON)

    # Campos útiles para debug o listados rápidos
    last_place_ids = db.Column(db.JSON)          # últimos place_ids que el bot publicó al mapa
    last_user_text = db.Column(db.Text)          # último mensaje del usuario
    last_assistant_text = db.Column(db.Text)     # última respuesta del asistente

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "conversation_id", name="uq_user_conv"),
    )
