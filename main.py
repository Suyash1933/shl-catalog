"""
FastAPI service for the SHL Assessment Recommender agent.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import load_index
from agent import chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the FAISS index and embedding model on startup
    print("Loading index and model...")
    load_index()
    print("Ready.")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = chat(messages)
        return ChatResponse(
            reply=result["reply"],
            recommendations=[
                Recommendation(**r) for r in result["recommendations"]
            ],
            end_of_conversation=result["end_of_conversation"],
        )
    except Exception as e:
        print(f"[ERROR] Chat failed: {type(e).__name__}: {e}")
        # Ensure we always return valid schema even on errors
        return ChatResponse(
            reply="I'm sorry, I encountered an issue processing your request. Could you please rephrase your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
