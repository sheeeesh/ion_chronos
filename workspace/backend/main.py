import os, json, uuid, asyncio
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from job_runner import run_job

WORKSPACE = Path(os.getenv('WORKSPACE','/workspace'))
JOBS_DIR = WORKSPACE / 'jobs'
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_INDEX = WORKSPACE / 'jobs_index.json'

app = FastAPI(title='Ion Chronos Dashboard')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

executor = ThreadPoolExecutor(max_workers=2)

def load_index():
    if JOBS_INDEX.exists():
        return json.loads(JOBS_INDEX.read_text())
    return {}
def save_index(idx):
    JOBS_INDEX.write_text(json.dumps(idx, indent=2))

class JobSpec(BaseModel):
    ticker: str = 'SPY'
    start_date: str = '2015-01-01'
    end_date: str = None
    timeframe: str = '1d'
    timesteps: int = 50000

@app.post('/api/jobs/start')
def start_job(spec: JobSpec):
    job_id = str(uuid.uuid4())
    job_path = JOBS_DIR / job_id
    job_path.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    idx[job_id] = {'id': job_id, 'spec': spec.dict(), 'status':'queued', 'artifacts': []}
    save_index(idx)
    executor.submit(_run_and_update, job_id, spec.dict(), str(job_path))
    return {'job_id': job_id}

def _run_and_update(job_id, spec, job_path):
    idx = load_index()
    idx[job_id]['status'] = 'running'
    save_index(idx)
    try:
        run_job(job_id, spec, job_path)
        idx = load_index(); idx[job_id]['status']='success'; save_index(idx)
    except Exception as e:
        idx = load_index(); idx[job_id]['status']='failed'; idx[job_id]['error']=str(e); save_index(idx)

@app.get('/api/jobs')
def list_jobs():
    return load_index()

@app.get('/api/jobs/{job_id}/artifacts')
def list_artifacts(job_id: str):
    job_path = JOBS_DIR / job_id
    if not job_path.exists():
        raise HTTPException(status_code=404, detail='Job not found')
    files = [{'name':p.name,'path':str(p)} for p in job_path.iterdir()]
    return {'artifacts': files}

@app.websocket('/ws/logs')
async def websocket_logs(ws: WebSocket, path: str):
    await ws.accept()
    pos = 0
    try:
        while True:
            p = Path(path)
            if p.exists():
                with p.open('rb') as f:
                    f.seek(pos)
                    data = f.read()
                    if data:
                        await ws.send_text(data.decode('utf-8', errors='replace'))
                        pos = f.tell()
            else:
                await ws.send_text(f'[waiting] file not found: {path}')
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        return
