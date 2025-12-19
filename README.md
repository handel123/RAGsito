# API FastAPI para Sistema RAG - Noticias Chile

##  Descripción

API REST construida con FastAPI que envuelve el sistema RAG (Retrieval-Augmented Generation) para consultas sobre noticias chilenas de sep 2025.

##  Instalación

1. **Instalar dependencias de la API:**
Crear entorno conda e instalar:
```bash
pip install -r requirements.txt
```
Crear archivo de variables de entorno con
API_KEY -> para modelos de razonamiento
API_KEY_EMBEDDINGS -> para entrenar embeddings
API_KEY_COHERE -> para reranker


##  Crear embeddings
```bash
python ingest.py
```


Si se quiere usar embeddings ya entrenados, se pueden descargar de la siguiente manera:

```bash
gdown --folder "https://drive.google.com/drive/folders/15FLbcQEq_mGDz3n1Kn7V_92wTcWxiRy0"

gdown --folder --remaining-ok "https://drive.google.com/drive/folders/1w1hUNddmb05y0jL9ICjBnITYoOoqJETe"
```




##  Ejecutar la API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`




###  `POST /query/stream` - Consulta con Streaming
Envía una pregunta y recibe la respuesta en tiempo real.

**Request Body:**
{
  "query": "¿Han habido paros en los últimos meses en el servicio público?"
}

**Respuesta:** Stream de eventos SSE

###  `GET /health` - Estado de Salud
Verifica que la API esté funcionando.

**Respuesta:**
```json
{
  "status": "healthy",
  "service": "RAG Noticias Chile"
}
```

## Probar la API


### Con cURL

```bash
curl --location 'http://localhost:8000/query/stream' \
--header 'Content-Type: application/json' \
--data '{"query": "¿Han habido paros en los últimos meses?"}'
```

