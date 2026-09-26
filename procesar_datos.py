"""
MLOps SGSI - Gobierno Regional de Pasco

Script de automatización para realizar inferencias
con el modelo de predicción de incidentes.

Flujo:
Entrada -> Validación -> Modelo -> Salida -> Log
"""

from pathlib import Path
from datetime import datetime
import logging
import joblib
import pandas as pd


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = (
    BASE_DIR
    / "models"
    / "modelo_sgsi_incidentes.joblib"
)

RUTA_ENTRADA = (
    BASE_DIR
    / "data"
    / "entrada_prediccion.csv"
)

RUTA_PREDICCIONES = (
    BASE_DIR
    / "output"
    / "predicciones.csv"
)

RUTA_LOG = (
    BASE_DIR
    / "logs"
    / "mlops_sgsi.log"
)


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

def cargar_modelo():

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    return joblib.load(RUTA_MODELO)


# ============================================================
# VALIDACIÓN DE ENTRADA
# ============================================================

def validar_entrada(datos):

    errores = []

    # Verificar columnas requeridas
    columnas_faltantes = [
        columna
        for columna in FEATURES
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        errores.append(
            f"Faltan columnas: {columnas_faltantes}"
        )

        return False, errores

    # Verificar valores nulos
    if datos[FEATURES].isnull().any().any():
        errores.append(
            "Existen valores nulos en los datos de entrada."
        )

    # Validar probability
    if not datos["probability"].between(1, 5).all():
        errores.append(
            "probability debe encontrarse entre 1 y 5."
        )

    # Validar impact
    if not datos["impact"].between(1, 5).all():
        errores.append(
            "impact debe encontrarse entre 1 y 5."
        )

    return len(errores) == 0, errores


# ============================================================
# PREDICCIÓN
# ============================================================

def realizar_prediccion(modelo, datos):

    predicciones = modelo.predict(
        datos[FEATURES]
    )

    probabilidades = modelo.predict_proba(
        datos[FEATURES]
    )[:, 1]

    resultados = [
        "Incidente" if prediccion == 1 else "No incidente"
        for prediccion in predicciones
    ]

    return pd.DataFrame({
        "prediccion": predicciones.astype(int),
        "resultado": resultados,
        "probabilidad_incidente": probabilidades.astype(float)
    })


# ============================================================
# GUARDADO DE RESULTADOS
# ============================================================

def guardar_resultado(datos, resultados_prediccion):

    salida = datos[FEATURES].copy().reset_index(drop=True)
    resultados_prediccion = resultados_prediccion.reset_index(drop=True)

    salida["prediccion"] = (
        resultados_prediccion["prediccion"]
    )

    salida["resultado"] = (
        resultados_prediccion["resultado"]
    )

    salida["probabilidad_incidente"] = (
        resultados_prediccion["probabilidad_incidente"]
        .round(6)
    )

    salida["fecha_prediccion"] = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    RUTA_PREDICCIONES.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    archivo_existe = RUTA_PREDICCIONES.exists()

    salida.to_csv(
        RUTA_PREDICCIONES,
        mode="a",
        header=not archivo_existe,
        index=False,
        encoding="utf-8-sig"
    )

    return RUTA_PREDICCIONES


# ============================================================
# REGISTRO DE EVENTOS
# ============================================================

def registrar_evento(mensaje):

    RUTA_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger("mlops_sgsi_script")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:

        file_handler = logging.FileHandler(
            RUTA_LOG,
            encoding="utf-8"
        )

        formato = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formato)
        logger.addHandler(file_handler)

    logger.info(mensaje)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("MLOps SGSI - Gobierno Regional de Pasco")
    print("=" * 60)

    modelo = cargar_modelo()

    print(f"Modelo cargado desde: {RUTA_MODELO}")
    print(f"Tipo: {type(modelo).__name__}")

    print()
    print("✅ Modelo cargado correctamente.")

    # --------------------------------------------------------
    # CARGA DE DATOS DE ENTRADA
    # --------------------------------------------------------

    print()
    print("CARGA DE DATOS DE ENTRADA")
    print("=" * 60)

    if not RUTA_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {RUTA_ENTRADA}"
        )

    datos_entrada = pd.read_csv(
        RUTA_ENTRADA,
        encoding="utf-8-sig"
    )

    print(f"Archivo: {RUTA_ENTRADA}")
    print(f"Registros: {len(datos_entrada)}")
    print("✅ Datos de entrada cargados correctamente.")

    entrada_valida, errores = validar_entrada(
        datos_entrada
    )

    print()
    print("VALIDACIÓN DE ENTRADA")
    print("=" * 60)

    if entrada_valida:
        print("✅ Entrada válida.")
        print("El registro puede enviarse al modelo.")

        # Realizar inferencia
        resultado_prediccion = realizar_prediccion(
            modelo,
            datos_entrada
        )

        print()
        print("PREDICCIONES DEL MODELO")
        print("=" * 60)

        for indice, fila in resultado_prediccion.iterrows():

            print(f"Registro {indice + 1}")
            print(
                f"  Predicción       : "
                f"{fila['prediccion']}"
            )
            print(
                f"  Resultado        : "
                f"{fila['resultado']}"
            )
            print(
                f"  Probabilidad     : "
                f"{fila['probabilidad_incidente']:.4f}"
            )
            print(
                f"  Probabilidad (%) : "
                f"{fila['probabilidad_incidente'] * 100:.2f}%"
            )
            print("-" * 40)

        # Guardar resultados automáticamente
        ruta_salida = guardar_resultado(
            datos_entrada,
            resultado_prediccion
        )

        print()
        print("GUARDADO DE RESULTADOS")
        print("=" * 60)
        print(f"Archivo: {ruta_salida}")
        print(
            f"Registros guardados: "
            f"{len(resultado_prediccion)}"
        )
        print("✅ Predicciones almacenadas correctamente.")

        # Registrar cada inferencia en el log
        for indice, fila in resultado_prediccion.iterrows():

            registrar_evento(
                "Inferencia realizada | "
                "modelo=Regresión Logística | "
                f"registro={indice + 1} | "
                f"prediccion={int(fila['prediccion'])} | "
                f"resultado={fila['resultado']} | "
                f"probabilidad_incidente="
                f"{fila['probabilidad_incidente']:.6f}"
            )

        print()
        print("REGISTRO DE EJECUCIÓN")
        print("=" * 60)
        print(f"Archivo: {RUTA_LOG}")
        print(
            f"Eventos registrados: "
            f"{len(resultado_prediccion)}"
        )
        print("✅ Eventos registrados correctamente.")

    else:
        print("❌ Entrada inválida.")

        for error in errores:
            print(f"  - {error}")
