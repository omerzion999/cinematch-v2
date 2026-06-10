from fastapi import FastAPI

app = FastAPI(title="CineMatch AI v2")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
