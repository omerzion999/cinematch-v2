import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import chat, recommend, seeds, show, translate
from app.state import load_state

_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_ENV_FILE)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cinematch = load_state()
    yield


app = FastAPI(title="CineMatch AI v2", lifespan=lifespan)
app.include_router(recommend.router)
app.include_router(seeds.router)
app.include_router(chat.router)
app.include_router(show.router)
app.include_router(translate.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
