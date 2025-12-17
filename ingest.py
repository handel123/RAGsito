import os
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore, create_kv_docstore
from dot_env import load_dotenv
load_dotenv()
api_key2 = os.getenv("API_KEY_EMBEDDINGS")


csv_path = "data/dataset_proyecto_chile_septiembre2025.csv"
persist_dir = "./dbs_chroma_definitivo"
docstore_dir = "./docstore_parents"  # Directorio para persistir los chunks parent



# 1. Definir Modelos
embedding_model = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key2)



# 2. Estrategia Parent-Child con dos splitters
# Child Splitter: Chunks pequeños (900 chars) para búsqueda precisa en el vectorstore
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, 
    chunk_overlap=150
)

# Parent Splitter: Chunks grandes (4500 chars) que se devolverán con más contexto
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4500, 
    chunk_overlap=500
)



# 3. Inicializar Vectorstore (para los chunks child)
vectorstore = Chroma(
    collection_name="noticias_chile",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

# 4. Docstore (para guardar los chunks parent - mayor contexto)
# LocalFileStore persiste los documentos padres en disco
file_store = LocalFileStore(docstore_dir)
docstore = create_kv_docstore(file_store)

# 5. Configurar ParentDocumentRetriever
# Busca con chunks pequeños pero devuelve chunks grandes
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 5}
)

# 6. Cargar Datos (Solo si la DB está vacía para no duplicar)
# Verificamos si Chroma ya tiene datos


def docstore_is_empty(path: str) -> bool:
    return (not os.path.exists(path)) or (len(os.listdir(path)) == 0)


vectorstore_empty = len(vectorstore.get()["ids"]) == 0
docstore_empty = docstore_is_empty(docstore_dir)

existing_data = vectorstore.get()

# Leer CSV COMPLETO
df = pd.read_csv(csv_path)
print(f"CSV cargado: {len(df)} noticias")

# Contar DOCUMENTOS ÚNICOS (no chunks child)
existing_ids = set()
for metadata in existing_data['metadatas']:
    if metadata and 'id_news' in metadata:
        existing_ids.add(metadata['id_news'])

num_chunks_child = len(existing_data['ids'])
num_docs_indexados = len(existing_ids)


if num_docs_indexados < len(df):
    
    # Filtrar solo documentos nuevos
    if existing_ids:
        df = df[~df['id_news'].astype(str).isin(existing_ids)]
    
    # Procesar documentos con barra de progreso
    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creando documentos", unit="doc"):
        # Enriquecer el texto para contexto
        enriched_text = (
            f"TITULO: {row['title']}\n"
            f"FECHA: {row['date']}\n"
            f"MEDIO: {row['media_outlet']}\n"
            f"CONTENIDO:\n{row['text']}"
        )
        
        # Metadatos para filtrado posterior
        # Incluimos id_news en metadatos como identificador estable
        metadata = {
            "id_news": str(row['id_news']),
            "url": row['url'],
            "date": row['date'],
            "country": row['country'],
            "media_outlet": row['media_outlet']
        }
        
        doc = Document(page_content=enriched_text, metadata=metadata)
        docs.append(doc)
    
    

    batch_size = 50  # Procesar de a 50 documentos para mostrar progreso
    total_batches = (len(docs) + batch_size - 1) // batch_size
    





# # Ejemplo de uso rápido
# print("\n--- Prueba de consulta ---")

# filtro = {
    
#     "date": "Sep 23, 2025 @ 00:00:00.000"
    
# }
# retriever.search_kwargs = {
#     "k": 10,           
#     "filter": filtro,
    
# }


# response = retriever.invoke("el partido comunista")



# print(f"Encontrados {len(response)} documentos parent relevantes")
# for i, doc in enumerate(response[:2], 1):
#     print(f"\nDocumento parent {i}:")
#     print(f"Medio: {doc.metadata.get('media_outlet', 'N/A')}")
#     print(f"Fecha: {doc.metadata.get('date', 'N/A')}")
#     print(f"Tamaño: {len(doc.page_content)} caracteres")
#     print(f"Contenido (primeros 200 chars): {doc.page_content[:200]}...")
    
# if len(response) > 0:
#     print(response[0].page_content)