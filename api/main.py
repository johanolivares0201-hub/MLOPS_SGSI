from pathlib import Path
from typing import Literal
import logging

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent

RUTA_MODELO = (
    BASE_DIR
    / "models"
    / "modelo_sgsi_incidentes.joblib"
)



# ============================================================
# LOGGING DE INFERENCIAS
# ============================================================

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "api_ml.log"

logger = logging.getLogger("api_ml")
logger.setLevel(logging.INFO)

if not logger.handlers:

    handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================
# VARIABLES DEL MODELO
# ============================================================

FEATURES = [
    "asset_category",
    "asset_classification",
    "asset_criticality",
    "asset_tipo",
    "threat",
    "vulnerability",
    "probability",
    "impact",
    "treatment"
]


# ============================================================
# CARGA DEL MODELO
# ============================================================

if not RUTA_MODELO.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo en: {RUTA_MODELO}"
    )

modelo = joblib.load(RUTA_MODELO)


# ============================================================
# ESQUEMA DE ENTRADA
# ============================================================

class RiesgoEntrada(BaseModel):

    asset_category: Literal[
        "base de datos",
        "documentación",
        "equipo de red",
        "hardware",
        "información",
        "infraestructura",
        "otros",
        "redes y comunicaciones",
        "servicio",
        "servicios ti",
        "software"
    ]

    asset_classification: Literal[
        "confidencial",
        "interno",
        "publico",
        "reservada",
        "restringido",
        "uso interno"
    ]

    asset_criticality: Literal[
        "alta",
        "baja",
        "critica",
        "media",
        "muy alta"
    ]

    asset_tipo: Literal[
        "digital",
        "físico"
    ]

    threat: Literal[
        "acceso no autorizado",
        "daño físico",
        "divulgación de información",
        "error humano",
        "falla eléctrica",
        "incendio",
        "interrupción del servicio",
        "inundación",
        "malware",
        "pérdida de información",
        "ransomware",
        "robo"
    ]

    vulnerability: Literal[
        "almacenamiento inadecuado",
        "antivirus desactualizado",
        "ausencia de políticas",
        "ausencia de respaldos",
        "configuración incorrecta",
        "contraseñas débiles",
        "equipos obsoletos",
        "falta de capacitación",
        "falta de controles de acceso",
        "sin mantenimiento"
    ]

    probability: int = Field(
        ge=1,
        le=5
    )

    impact: int = Field(
        ge=1,
        le=5
    )

    treatment: Literal[
        "aceptar",
        "evitar",
        "mitigar",
        "transferir"
    ]


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="API Predictiva SGSI",
    description=(
        "Servicio de Machine Learning para la predicción "
        "de incidentes de seguridad de la información."
    ),
    version="1.0.0"
)

# ============================================================
# CORS - PLATAFORMA SGSI
# ============================================================

ORIGENES_PERMITIDOS = [
    "https://main.dgg4ukovaowni.amplifyapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)



# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def inicio():

    return {
        "servicio": "API Predictiva SGSI",
        "estado": "activo",
        "modelo": "Regresión Logística"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "tipo_modelo": type(modelo).__name__
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predict")
def predict(riesgo: RiesgoEntrada):

    datos = pd.DataFrame([
        riesgo.model_dump()
    ])

    datos = datos[FEATURES]

    prediccion = int(
        modelo.predict(datos)[0]
    )

    probabilidad = float(
        modelo.predict_proba(datos)[0][1]
    )

    resultado = (
        "Incidente"
        if prediccion == 1
        else "No incidente"
    )

    logger.info(
        "Predicción realizada | resultado=%s | probabilidad=%.4f",
        resultado,
        probabilidad
    )

    return {
        "prediccion": prediccion,
        "resultado": resultado,
        "probabilidad_incidente": round(
            probabilidad,
            6
        ),
        "probabilidad_porcentaje": round(
            probabilidad * 100,
            2
        )
    }
