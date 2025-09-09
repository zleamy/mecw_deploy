from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import faiss, pickle, json
from sentence_transformers import SentenceTransformer
import openai
import os

# ========================
# CONFIG
# ========================
# Use environment variable for OpenAI API key for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

OPENAI_MODEL = "gpt-4o"
TOP_K = 15

# For Railway deployment, files should be in the same directory as the app
SCRIPT_DIR = Path(__file__).parent
INDEX_PATH = SCRIPT_DIR / "mecw_faiss.index"
METADATA_PATH = SCRIPT_DIR / "mecw_metadata.pkl"

# Debug: Print paths to verify they're correct
print(f"Script directory: {SCRIPT_DIR}")
print(f"Index path: {INDEX_PATH}")
print(f"Metadata path: {METADATA_PATH}")
print(f"Index exists: {INDEX_PATH.exists()}")
print(f"Metadata exists: {METADATA_PATH.exists()}")

# ========================
# LOAD FAISS AND METADATA
# ========================
print("Loading FAISS index...")
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found at: {METADATA_PATH}")
    
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

if not isinstance(metadata, list):
    raise ValueError("Metadata is not a list. Check the file!")

print(f"Loaded {len(metadata)} metadata entries")

# Load Sentence Transformer
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# ========================
# Pydantic Input Model
# ========================
class QueryRequest(BaseModel):
    query: str
    citation_style: str = "Chicago"

# ========================
# RETRIEVE FUNCTION
# ========================
def retrieve(query: str, k=TOP_K):
    query_vec = model.encode([query])
    scores, indices = index.search(query_vec.astype("float32"), k)
    results = []
    for i, score in zip(indices[0], scores[0]):
        if i >= len(metadata):
            continue
        entry = metadata[i]
        
        # For cloud deployment, we'll include text directly in metadata
        # or store it in a way that doesn't require separate chunk files
        chunk_text = entry.get("text", "[Text not available]")
        if not chunk_text or chunk_text == "[Text not available]":
            # Fallback: try to read from filename if it exists
            chunk_filename = entry.get("filename", "")
            if chunk_filename:
                chunk_file = Path(chunk_filename)
                if not chunk_file.is_absolute():
                    chunk_file = SCRIPT_DIR / chunk_filename
                    
                if chunk_file.exists():
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        chunk_text = f.read()
                else:
                    chunk_text = f"[File not found: {chunk_file}]"
            
        results.append({
            "text": chunk_text,
            "volume": entry.get("volume", "Unknown"),
            "chunk_id": entry.get("chunk_id", "Unknown"),
            "page": entry.get("page", "Unknown"),
            "score": float(score)
        })
    return results

# ========================
# CALL OPENAI
# ========================
def ask_openai(query, context_text, citation_style="Chicago"):
    openai.api_key = OPENAI_API_KEY
    prompt = f"""
You are a scholarly assistant writing a formal essay about the Marx/Engels Collected Works (MECW).
Your task is to answer the user's question by synthesizing information from the provided documents.
Use direct quotes and specific references from the documents to support your claims.

SCHOLARLY INSTRUCTIONS:
1. First, think step by step and outline your reasoning. Analyze the key concepts in the user's query and cross-reference them with the provided context. Consider any contradictions or nuances within the sources. This step is for your internal thought process and will be shown to the user.
2. Based on your reasoning, write a well-structured, coherent essay that directly answers the user's question.
3. For every factual claim you make, you MUST provide an in-text citation pointing to the specific source. Use the format: (MECW, Vol. [Volume Number], p. [Page Number]).
4. If a piece of information is found in multiple documents, cite all of them.
5. Do NOT make up information or cite sources that are not provided.
6. In your essay, you must quote at least 3 distinct passages from the provided documents. Use quotation marks and provide citations for each quote.

Begin your response with the reasoning process, then provide the final essay.
Format references in {citation_style} style.

Context:
{context_text}

Question: {query}
"""
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

# ========================
# FASTAPI APP
# ========================
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MECW Archive API", description="Marx/Engels Collected Works RAG API")

# Allow all origins (for production, you'd want to be more restrictive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "MECW API is running", 
        "status": "ok",
        "metadata_entries": len(metadata),
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "entries": len(metadata)}

@app.post("/query")
def query_mecw(req: QueryRequest):
    try:
        print(f"Received query: {req.query}")
        docs = retrieve(req.query)
        if not docs:
            return {"error": "No relevant chunks found."}
        
        print(f"Retrieved {len(docs)} documents")
        
        # Build context
        context_text = ""
        for doc in docs:
            context_text += f"Volume {doc['volume']} Chunk {doc['chunk_id']} Page {doc['page']}\n{doc['text']}\n---\n"

        print("Calling OpenAI...")
        answer = ask_openai(req.query, context_text, req.citation_style)
        
        return {
            "query": req.query,
            "retrieved_docs": docs,
            "answer": answer
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Railway will automatically detect the port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)