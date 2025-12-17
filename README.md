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


##  Ejecutar la API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`





### 3. `POST /query/stream` - Consulta con Streaming
Envía una pregunta y recibe la respuesta en tiempo real.

**Request Body:**
{
  "query": "¿Han habido paros en los últimos meses en el servicio público?",
  "k": 10
}

**Respuesta:** Stream de eventos SSE

### 4. `GET /health` - Estado de Salud
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
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Han habido paros en los últimos meses?", "k": 10}'
```


