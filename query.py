import os
from urllib import response
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore, create_kv_docstore
import json
import openai
from datetime import datetime
from pathlib import Path
import cohere

from dotenv import load_dotenv


load_dotenv()

# API Key para OpenAI
api_key = os.getenv("API_KEY")
api_key_embeddings = os.getenv("API_KEY_EMBEDDINGS")
api_key_cohere = os.getenv("API_KEY_COHERE")
csv_path = "data/dataset_proyecto_chile_septiembre2025.csv"
persist_dir = "./dbs_chroma_definitivo"
docstore_dir = "./docstore_parents"  # Directorio para persistir los chunks parent
logs_dir = "./debug_logs"  # Directorio para logs de debug
# 3. Inicializar Vectorstore (para los chunks child)
embedding_model = "text-embedding-3-large"

# Crear directorio de logs si no existe
Path(logs_dir).mkdir(exist_ok=True)

# Configuración de logging
DOC_CONTENT_LIMIT = 500  # Caracteres máximos a guardar de cada documento

def save_debug_log(log_data, filename=None):
    """Guarda los datos de debug en un archivo JSON."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_log_{timestamp}.json"
    
    filepath = Path(logs_dir) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[DEBUG] Log guardado en: {filepath}")
    return str(filepath)



embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key_embeddings)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, 
    chunk_overlap=150
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4500, 
    chunk_overlap=500
)


vectorstore = Chroma(
    collection_name="noticias_chile",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

file_store = LocalFileStore(docstore_dir)
docstore = create_kv_docstore(file_store)



retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 5}
)








def extraer_json_de_respuesta(raw_content):
    """Extrae y parsea JSON de la respuesta del modelo, manejando markdown code blocks."""
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # Si viene en markdown code block, limpiar los backticks
        cleaned = raw_content.strip()
        if cleaned.startswith('```'):
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline+1:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        else:
            raise ValueError(f"No se pudo extraer JSON de la respuesta: {raw_content}")


def construir_filtro_fecha(date_filter):
    """Construye el diccionario de filtro para ChromaDB según el tipo de filtro de fecha."""
    if not date_filter or not date_filter.get("type"):
        return None
    
    if date_filter["type"] == "exact" and date_filter.get("date"):
        return {"date": date_filter["date"]}
    
    elif date_filter["type"] == "range" and date_filter.get("start_date") and date_filter.get("end_date"):
        # ChromaDB usa $gte y $lte para rangos (aunque no funciona con strings)
        return {
            "$and": [
                {"date": {"$gte": date_filter["start_date"]}},
                {"date": {"$lte": date_filter["end_date"]}}
            ]
        }
    
    return None


def filtrar_documentos_por_fecha(docs, start_date_str, end_date_str):
    """Filtra manualmente una lista de documentos por rango de fechas."""
    from datetime import datetime as dt
    
    try:
        start = dt.strptime(start_date_str, "%b %d, %Y @ %H:%M:%S.%f")
        end = dt.strptime(end_date_str, "%b %d, %Y @ %H:%M:%S.%f")
    except ValueError:
        print(f"[DEBUG] Error parseando fechas de filtro")
        return docs
    
    filtered_results = []
    for doc in docs:
        doc_date_str = doc.metadata.get("date")
        if not doc_date_str:
            continue
        
        doc_date = None
        # Intentar ambos formatos (con y sin padding en el día)
        for fmt in ["%b %d, %Y @ %H:%M:%S.%f", "%b %#d, %Y @ %H:%M:%S.%f"]:
            try:
                doc_date = dt.strptime(doc_date_str, fmt)
                break
            except:
                continue
        
        if doc_date and start <= doc_date <= end:
            filtered_results.append(doc)
    
    return filtered_results


def buscar_con_filtro(vectorstore, subquery, k, filter_dict, date_filter):
    """
    Busca en el vectorstore con filtro. Si falla, busca sin filtro y filtra manualmente.
    Retorna la lista de documentos encontrados.
    """
    try:
        child_results = vectorstore.similarity_search(subquery, k=k, filter=filter_dict)
        print(f"[DEBUG] Resultados encontrados con filtro: {len(child_results)}")
        return child_results
    
    except Exception as e:
        print(f"[DEBUG] Error al buscar con filtro: {e}")
        print(f"[DEBUG] Intentando sin filtro...")
        
        # Buscar sin filtro
        child_results = vectorstore.similarity_search(subquery, k=k * 2)
        print(f"[DEBUG] Resultados sin filtro: {len(child_results)}")
        
        # Filtrar manualmente si es rango de fechas
        if filter_dict and date_filter and date_filter.get("type") == "range":
            print(f"[DEBUG] Aplicando filtrado manual por fechas...")
            child_results = filtrar_documentos_por_fecha(
                child_results,
                date_filter["start_date"],
                date_filter["end_date"]
            )
            print(f"[DEBUG] Resultados después de filtrado manual: {len(child_results)}")
        
        return child_results

def configurar_parametros_rag(intents):
    # Valores por defecto (Balanceados)
    config = {
        "CHILD_K": 10,      
        "RERANK_TOP_N": 5,  
        "TOP_PARENTS": 5    
    }
    
    # 1. Ajuste por ALCANCE (El factor más determinante)
    if intents.get("alcance") == "panorama" or intents.get("alcance") == "varios hechos":
        config["CHILD_K"] = 20      
        config["RERANK_TOP_N"] = 7  
        config["TOP_PARENTS"] = 9   
        
    # 2. Ajuste por PROFUNDIDAD
    elif intents.get("profundidad") == "puntual":
        config["CHILD_K"] = 8        
        config["RERANK_TOP_N"] = 3  
        config["TOP_PARENTS"] = 3    
        
    # 3. Ajuste por PROPÓSITO
    if intents.get("proposito") == "seguimiento temporal":
        config["TOP_PARENTS"] = max(config["TOP_PARENTS"], 8)
        
    # 4. Ajuste por FORMATO
    if intents.get("formato") == "lista":
        config["TOP_PARENTS"] = max(config["TOP_PARENTS"], 8)

    print(f"[DEBUG] Configuración Adaptativa: {config}")
    return config

def construir_instrucciones_estilo(intents,query,context):
    instrucciones = []

    # 1. FORMATO (La estructura visual)
    fmt = intents.get("formato")
    if fmt == "lista":
        instrucciones.append("- Estructura la respuesta usando viñetas (bullet points) o lista numerada para mayor claridad.")
    elif fmt == "resumen":
        instrucciones.append("- Genera un resumen consolidado en un solo bloque de texto o párrafos cortos.")
    elif fmt == "explicacion narrativa":
        instrucciones.append("- Usa una narrativa fluida, conectando los hechos en párrafos bien redactados.")

    # 2. EXTENSIÓN (Longitud deseada)
    ext = intents.get("extension")
    if ext == "corta":
        instrucciones.append("- Sé extremadamente conciso. Ve directo al grano sin rodeos.")
    elif ext == "larga":
        instrucciones.append("- Puedes extenderte en los detalles. No omitas información relevante del contexto.")
    
    # 3. PROPÓSITO (El enfoque mental del modelo)
    prop = intents.get("proposito")
    if prop == "comparar":
        instrucciones.append("- Tu objetivo es CONTRASTAR la información. Usa palabras como 'por otro lado', 'a diferencia de', 'mientras que'.")
    elif prop == "entender causas":
        instrucciones.append("- Enfócate en explicar el POR QUÉ de los sucesos y sus consecuencias.")
    elif prop == "seguimiento temporal":
        instrucciones.append("- Es CRUCIAL que mantengas un orden cronológico estricto en tu respuesta.")
    
    # 4. PROFUNDIDAD (Tono)
    prof = intents.get("profundidad")
    if prof == "analitica":
        instrucciones.append("- Adopta un tono analítico, sintetizando las ideas principales más allá de solo citar.")

    instrucciones_prompt = "\n".join(instrucciones)

    final_prompt = f"""
        Actúa como un asistente de noticias experto y servicial.
        Responde la siguiente pregunta usando SOLO el contexto entregado.
        
        INSTRUCCIONES DE ESTILO Y FORMATO (Síguelas estrictamente):
        {instrucciones_prompt}
        
        REGLAS GLOBALES:
        - La respuesta debe estar OBLIGATORIAMENTE EN MARKDOWN.
        - Si el contexto no tiene la información, dilo honestamente.
        - No opines sobre el contenido, solo informa.
        - Usa un tono conversacional y profesional.

        Pregunta:
        {query}

        Contexto:
        {context}
        """

    return final_prompt



def obtener_respuesta_stream(query, api_hk):

    # Inicializar estructura de log
    debug_log = {
        "timestamp": datetime.now().isoformat(),
        "query_original": query,
        "steps": []
    }

    #client = openai.OpenAI(api_key=api_key)
    client = openai.OpenAI(api_key=api_hk, base_url="https://us.inference.heroku.com/v1/")
    co = cohere.ClientV2(
        api_key=api_key_cohere
    )

    # Yield la query original
    yield {"type": "query", "content": query}

    # Obtener fecha actual del sistema
    fecha_actual = datetime.now()
    fecha_actual_str = fecha_actual.strftime("%b %d, %Y @ %H:%M:%S.000")
    fecha_actual_legible = fecha_actual.strftime("%d de %B del %Y")

    analysis_prompt = f"""
        Analiza la siguiente pregunta de un usuario sobre noticias.
        
        FECHA ACTUAL DEL SISTEMA: {fecha_actual_legible} ({fecha_actual_str})
        Usa esta fecha para calcular referencias temporales relativas.

        Devuelve un JSON con este esquema EXACTO:

        {{
        "intents": {{
            "profundidad": "puntual" | "explicativa" | "analitica",
            "alcance": "un hecho" | "varios hechos" | "panorama",
            "formato": "resumen" | "lista" | "explicacion narrativa",
            "extension": "corta" | "media" | "larga"
            "proposito": "informase" | "comparar" | "entender causas" | "seguimiento temporal"
        
        }},
        "subqueries": list[string],
        "date_filter": {{
            "type": "exact" | "range" | null,
            "date": string | null,
            "start_date": string | null,
            "end_date": string | null
        }}
        }}

        Reglas:
        - debe capturar la intención del usuario según los ejes indicados
        - si no se especifica ningún eje, inferir el más probable según la pregunta
        - debe subdividir la query si es necesario en subqueries con al menos una pregunta concreta
        - las subqueries son subpreguntas formadas de la pregunta original para buscar en la base de datos
        - Las subqueries deben ser SEMÁNTICAMENTE DIFERENTES entre sí
        - No generes subqueries que solo cambien sinónimos
        - Cada subquery debe apuntar a un aspecto distinto de la pregunta
        - no forzar subqueries si la pregunta es simple
        
        Para date_filter:
        - Si no hay referencia temporal: {{"type": null, "date": null, "start_date": null, "end_date": null}}
        - Para fecha exacta (ej: "23 de septiembre"): {{"type": "exact", "date": "Sep 23, 2025 @ 00:00:00.000", "start_date": null, "end_date": null}}
        - Para rangos (ej: "semana pasada", "últimos 3 meses"): {{"type": "range", "date": null, "start_date": "Sep 01, 2025 @ 00:00:00.000", "end_date": "Dec 17, 2025 @ 00:00:00.000"}}
        
        Formato de fechas: "Sep 23, 2025 @ 00:00:00.000"
        
        Ejemplos de rangos a calcular desde {fecha_actual_legible}:
        - "semana pasada": del lunes al domingo de la semana anterior
        - "últimos 3 meses": desde hace 3 meses hasta hoy
        - "este mes": desde el día 1 del mes actual hasta hoy
        - "ayer": solo el día anterior completo
        
        No inventes fechas, usa la FECHA ACTUAL DEL SISTEMA para calcular.
        No agregues texto fuera del JSON.

        Pregunta:
        \"\"\"{query}\"\"\"
    """

    yield {"type": "status", "content": "Analizando pregunta..."}

    try:
        analysis_response = client.chat.completions.create(
            #model="gpt-5-mini",
            model="claude-4-5-sonnet",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
    except Exception as e:
        print(f"[DEBUG] Error en la llamada al modelo: {e}")
        raise

    print(f"[DEBUG] Response completo: {analysis_response}")
    print(f"[DEBUG] Choices: {analysis_response.choices if hasattr(analysis_response, 'choices') else 'NO CHOICES'}")
    
    if not hasattr(analysis_response, 'choices') or not analysis_response.choices:
        raise ValueError("La respuesta del modelo no tiene 'choices'")
    
    if not hasattr(analysis_response.choices[0], 'message'):
        raise ValueError("El choice no tiene 'message'")
    
    raw_content = analysis_response.choices[0].message.content
    print(f"[DEBUG] Respuesta del modelo: '{raw_content[:200]}...'")
    
    if not raw_content or raw_content.strip() == "":
        raise ValueError("La respuesta del modelo está vacía")
    
    # Parsear JSON usando la función auxiliar
    analysis = extraer_json_de_respuesta(raw_content)
    print(f"[DEBUG] JSON parseado exitosamente!")
    
    debug_log["steps"].append({
        "step": "analysis",
        "prompt": analysis_prompt,
        "response": analysis,
        "raw_response": analysis_response.choices[0].message.content,
        "model": "claude-4-5-sonnet"
    })

    intent_axes = analysis["intents"]
    subqueries = analysis["subqueries"]
    date_filter = analysis["date_filter"]

    # Construir filtro usando función auxiliar
    filter_dict = construir_filtro_fecha(date_filter)

    # Mensaje informativo sobre el filtro aplicado
    filter_msg = ""
    if filter_dict and date_filter:
        if date_filter["type"] == "exact":
            filter_msg = f" (fecha: {date_filter['date']})"
        elif date_filter["type"] == "range":
            filter_msg = f" (rango: {date_filter['start_date']} a {date_filter['end_date']})"
    
    yield {"type": "status", "content": f"Buscando información ({len(subqueries)} consultas){filter_msg}..."}

    # CHILD_K = 10
    # RERANK_TOP_N = 5
    # TOP_PARENTS = 5
    rag_config = configurar_parametros_rag(intent_axes)
    CHILD_K = rag_config["CHILD_K"]
    RERANK_TOP_N = rag_config["RERANK_TOP_N"]
    
    parent_scores = {}
    parent_docs = {}
    
    
    retrieval_log = []
    
    for subq in subqueries:
        print(f"[DEBUG] Buscando con subquery: '{subq}'")
        print(f"[DEBUG] Filtro aplicado: {filter_dict}")

        # Usar función auxiliar para buscar con filtro
        child_results = buscar_con_filtro(
            retriever.vectorstore,
            subq,
            CHILD_K,
            filter_dict,
            date_filter
        )

        if not child_results:
            print(f"[DEBUG] No se encontraron resultados para esta subquery")
            continue

        rerank_docs = [doc.page_content for doc in child_results]

        rerank_response = co.rerank(
            model="rerank-v4.0-pro",
            query=subq,
            documents=rerank_docs,
            top_n=min(RERANK_TOP_N, len(rerank_docs))
        )

        subquery_log = {
            "subquery": subq,
            "retrieved_children": len(child_results),
            "reranked_children": len(rerank_response.results),
            "parents": []
        }

        for r in rerank_response.results:
            child_doc = child_results[r.index]
            parent_id = child_doc.metadata.get("id_news")

            if not parent_id:
                continue

            score = r.relevance_score
            parent_scores[parent_id] = parent_scores.get(parent_id, 0) + score

            if parent_id not in parent_docs:
                parent_docs[parent_id] = child_doc

            subquery_log["parents"].append({
                "id_news": parent_id,
                "score": score
            })

        retrieval_log.append(subquery_log)

    TOP_PARENTS = rag_config["TOP_PARENTS"]

    debug_log["steps"].append({
        "step": "retrieval",
        "subqueries_processed": retrieval_log,
        "unique_docs_found": len(parent_docs)
    })

    sorted_parents = sorted(
        parent_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    

    final_docs = [
        parent_docs[parent_id]
        for parent_id, _ in sorted_parents[:TOP_PARENTS]
    ]

    if not final_docs:
        no_results_msg = "No encontré información relevante en la base de datos para responder esta pregunta."
        if filter_dict:
            if date_filter["type"] == "range":
                no_results_msg += f"\n\nBusqué noticias entre {date_filter['start_date']} y {date_filter['end_date']}, pero no encontré resultados en ese rango de fechas."
            elif date_filter["type"] == "exact":
                no_results_msg += f"\n\nBusqué noticias de la fecha {date_filter['date']}, pero no encontré resultados."
        
        debug_log["final_response"] = no_results_msg
        debug_log["status"] = "no_results"
        save_debug_log(debug_log)
        
        yield {"type": "token", "content": no_results_msg}
        yield {"type": "done"}
        return
    
    final_docs = final_docs[:5]
    
    debug_log["steps"].append({
        "step": "final_docs_selection",
        "num_selected": len(final_docs),
        "selected_docs": [
            {
                "id_news": d.metadata.get("id_news"),
                "title": d.metadata.get("title"),
                "media_outlet": d.metadata.get("media_outlet"),
                "date": d.metadata.get("date"),
                "content_length": len(d.page_content),
                "content_preview": d.page_content[:DOC_CONTENT_LIMIT] + ("..." if len(d.page_content) > DOC_CONTENT_LIMIT else "")
            }
            for d in final_docs
        ]
    })

    context = "\n\n".join(
        f"[{d.metadata.get('media_outlet')} - {d.metadata.get('date')}]\n{d.page_content}"
        for d in final_docs
    )

    # final_prompt = f"""
    #     Responde la siguiente pregunta usando SOLO el contexto entregado, si es que responde bien.
    #     Responde de forma conversacional, breve y directa. Evita siempre la misma estructura. Varía entre listas, párrafos y ejemplos.
    #     LA RESPUESTA DEBE ESTAR OBLIGATORIAMENTE EN MARKDOWN, PARA QUE LO LEA EL FRONTEND BIEN.
    #     No opines NUNCA sobre el contexto en la respuesta.

    #     Pregunta:
    #     {query}

    #     Contexto:
    #     {context}
    #     """
    final_prompt = construir_instrucciones_estilo(intent_axes, query, context)

    debug_log["steps"].append({
        "step": "final_prompt",
        "prompt": final_prompt,
        "context_length": len(context),
        "model": "gpt-5-mini"
    })

    yield {"type": "status", "content": "Generando respuesta..."}
    
    # Stream real de OpenAI
    completion = client.chat.completions.create(
        #model="gpt-5-mini",
        model="claude-4-5-sonnet",
        messages=[{"role": "user", "content": final_prompt}],
        stream=True
    )
    
    respuesta_completa = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            respuesta_completa += token
            yield {"type": "token", "content": token}
    
    # Guardar log completo
    debug_log["final_response"] = respuesta_completa
    debug_log["status"] = "success"
    save_debug_log(debug_log)
    
    yield {"type": "done"}


if __name__ == "__main__":
    # Solo ejecuta esto si se corre directamente, no al importar
    query = "¿han habido paros en los ultimos meses en el servicio publico?"
    
    print(f"\n Consultando: {query}\n")
    print("=" * 80)
    
    for chunk in obtener_respuesta_stream(query, api_key):

        print(f" {chunk['content']}")
