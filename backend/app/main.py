from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
