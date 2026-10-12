from fastapi import FastAPI
from pydantic import BaseModel
from rag_workflow import run_agent

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    result = run_agent(req.question)
    return result