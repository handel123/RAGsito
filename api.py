from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
from query import  obtener_respuesta_stream
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
app = FastAPI(title="RAG Noticias Chile API")

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 10

class QueryResponse(BaseModel):
    query: str
    response: str
    status: str

@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {
        "message": "RAG Noticias Chile API",
        "status": "active",
        "endpoints": {
            "/query": "POST - Enviar pregunta y recibir respuesta",
            "/query/stream": "POST - Enviar pregunta y recibir respuesta en streaming"
        }
    }

# @app.post("/query", response_model=QueryResponse)
# async def query_endpoint(request: QueryRequest):
#     """
#     Endpoint para realizar consultas al sistema RAG.
    
#     Args:
#         request: Objeto con la consulta y parámetros opcionales
        
#     Returns:
#         QueryResponse con la pregunta original y la respuesta generada
#     """
#     try:
#         if not request.query or request.query.strip() == "":
#             raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
        
#         # Ejecutar la consulta en un thread separado para no bloquear
#         respuesta = await asyncio.to_thread(
#             obtener_respuesta,
#             request.query,
#             api_key2,
#             request.k
#         )
        
#         return QueryResponse(
#             query=request.query,
#             response=respuesta,
#             status="success"
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Endpoint para realizar consultas con respuesta en streaming real de OpenAI.
    
    Args:
        request: Objeto con la consulta y parámetros opcionales
        
    Returns:
        StreamingResponse con la respuesta generada token por token desde OpenAI
    """
    try:
        if not request.query or request.query.strip() == "":
            raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
        
        async def generate_response():
            """Generador asíncrono que transmite el stream real de OpenAI."""
            try:
                # Ejecutar el generador en un thread para no bloquear
                loop = asyncio.get_event_loop()
                
                # Crear el generador
                def sync_generator():
                    return obtener_respuesta_stream(request.query, api_key, request.k)
                
                generator = await loop.run_in_executor(None, sync_generator)
                
                # Iterar sobre los chunks del stream real
                for chunk_data in generator:
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
            except Exception as e:
                error_data = {"type": "error", "content": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de salud de la API."""
    return {
        "status": "healthy",
        "service": "RAG Noticias Chile"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
