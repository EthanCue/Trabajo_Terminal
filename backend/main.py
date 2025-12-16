import pickle
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from typing import List
from fastapi.middleware.cors import CORSMiddleware

"""
-----------------------------------------------------------------------------
MÓDULO DE INTERFAZ DE PROGRAMACIÓN DE APLICACIONES (API)
TRABAJO TERMINAL - SISTEMA "MEXCINE"
-----------------------------------------------------------------------------
"""

# ------------------------------------------------------------
# DEFINICIÓN DE ESQUEMAS DE DATOS (DTOs)
# ------------------------------------------------------------
class DescripcionRequest(BaseModel):
    descripcion: str

class PeliculaResponse(BaseModel):
    titulo: str
    genero: str
    sinopsis: str
    anio: int

# ------------------------------------------------------------
# CONFIGURACIÓN DE ARTEFACTOS Y RUTAS
# ------------------------------------------------------------
KNN_MODEL_PATH = "model_artifacts/modelo_knn.pkl"
DATA_PATH = "model_artifacts/peliculas_info.pkl"

# [IMPORTANTE] Apuntamos a la carpeta local donde 'download_model.py' guardó los archivos.
# Esto es vital para que Cloud Run no intente descargar nada de internet.
SENTENCE_MODEL_NAME = "./model_files" 

# Estructura global para mantener los modelos en memoria RAM
model_cache = {}

# ------------------------------------------------------------
# GESTIÓN DEL CICLO DE VIDA (LIFESPAN)
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- FASE DE ARRANQUE (STARTUP) ---
    print("--- [SISTEMA] Iniciando servidor y cargando recursos ---")
    
    print(f"-> Cargando modelo de lenguaje (NLP) desde carpeta local: '{SENTENCE_MODEL_NAME}'...")
    # Cargamos el modelo desde los archivos locales (sin internet)
    model_cache["sentence_transformer"] = SentenceTransformer(SENTENCE_MODEL_NAME)
    
    print(f"-> Deserializando modelo predictivo desde: '{KNN_MODEL_PATH}'...")
    with open(KNN_MODEL_PATH, "rb") as f:
        model_cache["knn_model"] = pickle.load(f)
        
    print(f"-> Cargando catálogo de metadatos desde: '{DATA_PATH}'...")
    with open(DATA_PATH, "rb") as f:
        data_dict = pickle.load(f)
        model_cache["peliculas_df"] = pd.DataFrame.from_dict(data_dict)
    
    print("✅ Todos los modelos han sido cargados en memoria RAM.")
    
    yield  # La aplicación corre aquí
    
    # --- FASE DE CIERRE (SHUTDOWN) ---
    print("--- [SISTEMA] Deteniendo servidor ---")
    model_cache.clear()
    print("Memoria liberada correctamente.")


# ------------------------------------------------------------
# INICIALIZACIÓN DE FASTAPI
# ------------------------------------------------------------
app = FastAPI(
    title="API de Recomendación de Cine Mexicano (Mexcine)",
    description="Backend para inferencia de similitud semántica en tiempo real.",
    version="0.1.0",
    lifespan=lifespan 
)

# ------------------------------------------------------------
# CONFIGURACIÓN DE POLÍTICAS CORS (Seguridad)
# ------------------------------------------------------------
# Permitimos TODOS los orígenes ["*"] para evitar problemas entre Cloud Run y el Frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Permitir cualquier origen
    allow_credentials=True,
    allow_methods=["*"],            # Permitir todos los verbos HTTP (GET, POST)
    allow_headers=["*"],
)


# ------------------------------------------------------------
# ENDPOINTS (Puntos de acceso)
# ------------------------------------------------------------

@app.get("/")
def read_root():
    """Health check + Diagnóstico de datos."""
    cantidad = 0
    if "peliculas_df" in model_cache:
        cantidad = len(model_cache["peliculas_df"])
    
    return {
        "status": "online",
        "servicio": "Mexcine API v1.1 con dataset triplicado xd",
        "peliculas_en_memoria": cantidad  
    }


@app.post("/recomendar", response_model=List[PeliculaResponse])
async def post_recomendar(request: DescripcionRequest):
    """
    Endpoint principal de inferencia.
    """
    
    # 1. Recuperamos modelos de la caché
    model_st = model_cache["sentence_transformer"]
    knn = model_cache["knn_model"]
    df_peliculas = model_cache["peliculas_df"]
    
    # 2. Vectorización del input del usuario
    vector_usuario = model_st.encode(request.descripcion).reshape(1, -1)

    # 3. Búsqueda de vecinos (Inferencia)
    # Solicitamos 15 vecinos para tener margen de filtrado posterior
    distancias, indices = knn.kneighbors(vector_usuario, n_neighbors=15)
    
    # 4. Reglas de Negocio
    UMBRAL_MINIMO = 0.80  # Umbral
    MAX_RESULTADOS = 15   # Límite amplio para la paginación
    recomendaciones = []
    
    # Procesamos los vecinos encontrados
    for i in range(len(indices[0])):
        idx = indices[0][i]      # Índice en el DataFrame original
        dist = distancias[0][i]  # Distancia coseno
        
        similitud = 1 - dist
        
        # Filtro de calidad
        if similitud >= UMBRAL_MINIMO:
            pelicula_data = df_peliculas.iloc[idx]
            
            # Mapeo al modelo de respuesta (DTO)
            recomendaciones.append(PeliculaResponse(
                titulo=pelicula_data['titulo'],
                genero=pelicula_data['genero'],
                sinopsis=pelicula_data['sinopsis'],
                anio=pelicula_data['anio']
            ))
        
        # Early exit
        if len(recomendaciones) >= MAX_RESULTADOS:
            break

    return recomendaciones

# Bloque de ejecución para desarrollo local
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
