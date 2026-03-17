from fastapi import FastAPI
from api.routes import signals, regime, health

app = FastAPI(title="QUANTSHIFT-SWARM Signal API", version="1.0.0")
app.include_router(health.router)
app.include_router(signals.router, prefix="/signals")
app.include_router(regime.router,  prefix="/regime")

@app.get("/")
async def root():
    return {"name": "QUANTSHIFT-SWARM", "version": "1.0.0", "status": "running"}
