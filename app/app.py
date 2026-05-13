from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import shutil
import uuid
from dotenv import load_dotenv

from services.document_service import DocumentService
from services.llm_service import LLMService
from core.rag_pipeline import VectorlessRAG

load_dotenv()

app = FastAPI(title="Vectorless RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_service = DocumentService(api_key=os.getenv("PAGEINDEX_API_KEY"))
llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY"))
rag_pipeline = VectorlessRAG(llm_service)

trees_db = {}
chats_db = {}

class AskRequest(BaseModel):
    doc_id: str
    chat_id: str
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type")
        
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        tree = doc_service.process_pdf(temp_path)
        doc_id = str(uuid.uuid4())
        trees_db[doc_id] = tree
        
        return {
            "message": "Success",
            "doc_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat/new")
async def create_chat():
    chat_id = str(uuid.uuid4())
    chats_db[chat_id] = []
    return {"chat_id": chat_id}

@app.post("/ask")
async def ask_question(request: AskRequest):
    if request.doc_id not in trees_db:
        raise HTTPException(status_code=404, detail="Document ID not found")
        
    if request.chat_id not in chats_db:
        chats_db[request.chat_id] = []
        
    if len(chats_db[request.chat_id]) > 6:
        summary = await llm_service.summarize_history(chats_db[request.chat_id][:-2])
        chats_db[request.chat_id] = [
            {"role": "system", "content": f"Previous conversation summary:\n{summary}"}
        ] + chats_db[request.chat_id][-2:]
        
    tree = trees_db[request.doc_id]
    chat_history = chats_db[request.chat_id]
    
    async def stream_generator():
        full_answer = ""
        
        async for chunk in rag_pipeline.answer_query_stream(request.query, tree, chat_history):
            full_answer += chunk
            yield chunk
            
        chats_db[request.chat_id].append({"role": "user", "content": request.query})
        chats_db[request.chat_id].append({"role": "assistant", "content": full_answer})

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat ID not found")
    return {"history": chats_db[chat_id]}