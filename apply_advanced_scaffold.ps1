# apply_advanced_scaffold.ps1
# Writes an advanced dashboard scaffold into C:\ion_chronos\workspace
# Behavior: does NOT overwrite existing files. If a file exists it writes a .new version.
$ErrorActionPreference = "Stop"

$ROOT = "C:\ion_chronos\workspace"
$TS = Get-Date -Format yyyyMMdd_HHmmss

# Ensure root exists
if (-not (Test-Path $ROOT)) {
  New-Item -ItemType Directory -Path $ROOT -Force | Out-Null
}

function Write-SafeFile([string]$path, [string]$content) {
  $dir = Split-Path $path -Parent
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  if (Test-Path $path) {
    $newp = "$path.new.$TS"
    $content | Out-File -FilePath $newp -Encoding utf8 -Force
    Write-Output "SKIPPED existing file, wrote: $newp"
  } else {
    $content | Out-File -FilePath $path -Encoding utf8 -Force
    Write-Output "WROTE: $path"
  }
}

# Helper to create folder
function New-Directory([string]$p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null } }

# Create base dirs
foreach ($d in @("backend","frontend","jobs","symbols","models","paper","configs","scripts","archive")) {
  New-Directory (Join-Path $ROOT $d)
}

# jobs_index.json placeholder
$jobs_index = Join-Path $ROOT "jobs_index.json"
if (-not (Test-Path $jobs_index)) { '{}' | Out-File -FilePath $jobs_index -Encoding utf8 -Force; Write-Output "WROTE: $jobs_index" }

### backend/requirements.txt
$req = @'
fastapi
uvicorn[standard]
pydantic
python-jose[cryptography]
passlib[bcrypt]
sqlalchemy
asyncpg
alembic
redis
rq
python-multipart
'@
Write-SafeFile (Join-Path $ROOT "backend\requirements.txt") $req

### backend/Dockerfile
$docker_backend = @'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/tools:$PYTHONPATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'@
Write-SafeFile (Join-Path $ROOT "backend\Dockerfile") $docker_backend

### backend/.env.example
$env_backend = @'
# Postgres
POSTGRES_USER=ion
POSTGRES_PASSWORD=ionpass
POSTGRES_DB=iondb
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://redis:6379/0

# JWT
JWT_SECRET=CHANGE_ME_STRONG_SECRET
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Workspace
WORKSPACE=/workspace
'@
Write-SafeFile (Join-Path $ROOT "backend\.env.example") $env_backend

### backend/main.py (FastAPI with JWT skeleton, DB minimal)
$backend_main = @'
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
'@
Write-SafeFile (Join-Path $ROOT "backend\main.py") $backend_main

### backend/job_runner.py that imports from tools (calls astro_dataset, rl_train, backtest)
$job_runner = @'
import os, sys, time, json, traceback
from pathlib import Path

TOOLS = Path(r"C:\ion_chronos\tools")
if TOOLS.exists() and str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

try:
    import astro_dataset as astro_mod
except Exception:
    astro_mod = None
try:
    import rl_train as rl_mod
except Exception:
    rl_mod = None
try:
    import backtest as backtest_mod
except Exception:
    backtest_mod = None

def _log(fp, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    fp.write(f"{ts} {msg}\n")
    fp.flush()

def run_job(job_id, spec, job_path):
    job_path = Path(job_path)
    job_path.mkdir(parents=True, exist_ok=True)
    logp = job_path / "run.log"
    summaryp = job_path / "summary.json"
    try:
        with logp.open("a", encoding="utf-8") as log:
            _log(log, f"[job {job_id}] start spec={spec}")
            _log(log, "Step: build dataset")
            try:
                if astro_mod and hasattr(astro_mod, "astro_dataset"):
                    astro_mod.astro_dataset(
                        ticker=spec.get("ticker"),
                        start_date=spec.get("start_date"),
                        end_date=spec.get("end_date"),
                        timeframe=spec.get("timeframe"),
                    )
                else:
                    _log(log, "(fallback) dummy dataset")
                    (job_path / "dataset.csv").write_text("date,open,close\n2020-01-01,100,101\n")
            except Exception as e:
                _log(log, f"astro_dataset error: {e}")
                _log(log, traceback.format_exc()); raise

            _log(log, "Step: rl_train")
            try:
                if rl_mod and hasattr(rl_mod, "rl_train"):
                    rl_mod.rl_train(
                        ticker=spec.get("ticker"),
                        total_timesteps=int(spec.get("timesteps", 100)),
                    )
                else:
                    _log(log, "(fallback) rl stub")
            except Exception as e:
                _log(log, f"rl_train error: {e}"); _log(log, traceback.format_exc()); raise

            _log(log, "Step: backtest")
            try:
                if backtest_mod and hasattr(backtest_mod, "backtest"):
                    backtest_mod.backtest(
                        ticker=spec.get("ticker"),
                        start_date=spec.get("start_date"),
                    )
                else:
                    (job_path / "trades.csv").write_text("EntryTime,ExitTime,NetPnLPercent\n2020-01-01,2020-01-02,0.5\n")
                    (job_path / "equity.png").write_text("PNG-DUMMY")
                    (job_path / "drawdown.png").write_text("PNG-DUMMY")
            except Exception as e:
                _log(log, f"backtest error: {e}"); _log(log, traceback.format_exc()); raise

            summary = {"job_id": job_id, "status":"success"}
            summaryp.write_text(json.dumps(summary, indent=2))
            _log(log, "Job completed")
    except Exception as e:
        with logp.open("a", encoding="utf-8") as log:
            _log(log, f"Job failed: {e}")
            _log(log, traceback.format_exc())
        summaryp.write_text(json.dumps({"job_id": job_id, "status":"failed", "error": str(e)}))
'@
Write-SafeFile (Join-Path $ROOT "backend\job_runner.py") $job_runner

### backend/worker.py (simple RQ consumer starter)
$worker_py = @'
import os
from rq import Connection, Queue, Worker
import redis

redis_url = os.getenv("REDIS_URL","redis://redis:6379/0")
conn = redis.from_url(redis_url)
queue = Queue("default", connection=conn)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(["default"])
        worker.work()
'@
Write-SafeFile (Join-Path $ROOT "backend\worker.py") $worker_py

### Docker Compose (advanced: redis + postgres + backend + worker + frontend)
$compose = @'
version: "3.8"
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ion
      POSTGRES_PASSWORD: ionpass
      POSTGRES_DB: iondb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    volumes:
      - ./jobs:/workspace/jobs
      - C:/ion_chronos/tools:/opt/tools:ro
      - ./backend:/app
    environment:
      - WORKSPACE=/workspace
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql+asyncpg://ion:ionpass@postgres:5432/iondb
      - PYTHONPATH=/opt/tools:$PYTHONPATH
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

  worker:
    build: ./backend
    command: python worker.py
    volumes:
      - ./jobs:/workspace/jobs
      - C:/ion_chronos/tools:/opt/tools:ro
      - ./backend:/app
    environment:
      - WORKSPACE=/workspace
      - REDIS_URL=redis://redis:6379/0
      - PYTHONPATH=/opt/tools:$PYTHONPATH
    depends_on:
      - redis
      - backend

  frontend:
    build: ./frontend
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
    depends_on:
      - backend

volumes:
  pgdata:
'@
Write-SafeFile (Join-Path $ROOT "docker-compose.yml") $compose

### frontend: package.json, index.html, src files (polished)
$pkg = @'
{
  "name": "astro-dashboard",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "start": "vite"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "plotly.js-dist-min": "^2.24.1"
  },
  "devDependencies": { "vite": "^5.0.0" }
}
'@
Write-SafeFile (Join-Path $ROOT "frontend\package.json") $pkg

$indexhtml = @'
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Ion Chronos — Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
'@
Write-SafeFile (Join-Path $ROOT "frontend\index.html") $indexhtml

$mainjsx = @'
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
createRoot(document.getElementById("root")).render(<App />);
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\main.jsx") $mainjsx

$appjsx = @'
import React from "react";
import JobForm from "./components/JobForm";
import JobList from "./components/JobList";
import "./styles.css";
export default function App(){
  return (
    <div className="app">
      <header className="topbar"><h1>Ion Chronos</h1></header>
      <div className="container">
        <div className="left"><JobForm/></div>
        <div className="right"><JobList/></div>
      </div>
    </div>
  );
}
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\App.jsx") $appjsx

$styles = @'
body{margin:0;font-family:Inter,system-ui;background:#0f172a;color:#e6eef6}
.topbar{background:#071026;padding:12px 20px}
.container{display:flex;gap:16px;padding:20px}
.left{width:420px}
.right{flex:1}
.card{background:rgba(255,255,255,0.03);padding:12px;border-radius:8px}
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\styles.css") $styles

# minimal components (JobForm, JobList, LogViewer) - keep concise but functional
$jobform = @'
import React, {useState} from "react";
export default function JobForm(){
  const [ticker,setTicker]=useState("SPY");
  const [timesteps,setTimesteps]=useState(100);
  async function submit(){
    try{
      const res = await fetch("/api/jobs/start",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({ticker,timesteps,start_date:"2015-01-01"})
      });
      const j = await res.json(); alert("Job started: "+j.job_id);
    }catch(e){ alert("Start failed: "+e) }
  }
  return (
    <div className="card">
      <h3>Start Job</h3>
      <label>Ticker</label><input value={ticker} onChange={e=>setTicker(e.target.value)} />
      <label>Timesteps</label><input value={timesteps} onChange={e=>setTimesteps(Number(e.target.value))} />
      <button onClick={submit}>Start</button>
    </div>
  );
}
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\components\JobForm.jsx") $jobform

$joblist = @'
import React, {useEffect,useState} from "react";
export default function JobList(){
  const [jobs,setJobs]=useState([]);
  async function load(){ try{ const r=await fetch("/api/jobs"); const j=await r.json(); setJobs(Object.values(j||{})); }catch(e){ console.error(e)}}
  useEffect(()=>{ load(); const t=setInterval(load,3000); return ()=>clearInterval(t); },[]);
  return (
    <div className="card">
      <h3>Jobs</h3>
      <ul>
        {jobs.length===0 && <li>No jobs</li>}
        {jobs.map(job=>(
          <li key={job.id} style={{display:"flex",justifyContent:"space-between",padding:8}}>
            <div><strong>{job.id.slice(0,8)}</strong><div style={{fontSize:12,color:"#94a3b8"}}>{job.status}</div></div>
            <div>
              <button onClick={async ()=>{
                const res=await fetch(`/api/jobs/${job.id}/artifacts`);
                const a=await res.json();
                const run = a.artifacts && a.artifacts.find(x=>x.name==="run.log");
                if(run){
                  const ws=new WebSocket(`ws://${window.location.host}/ws/logs?path=${encodeURIComponent(run.path)}`);
                  ws.onmessage=e=>alert(e.data.slice(0,200));
                }else alert("No run.log");
              }}>Tail</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\components\JobList.jsx") $joblist

$logviewer = @'
import React from "react";
export default function LogViewer(){ return (<div/>); }
'@
Write-SafeFile (Join-Path $ROOT "frontend\src\components\LogViewer.jsx") $logviewer

### README
$readme = @'
Ion Chronos Workspace - Advanced scaffold

1) Create .env from backend/.env.example and set real JWT_SECRET and DB credentials.

2) Start with Docker (recommended):
   cd C:\ion_chronos\workspace
   docker-compose up --build

3) Or run locally:
   # backend
   cd C:\ion_chronos\workspace\backend
   python -m pip install -r requirements.txt
   $env:PYTHONPATH='C:\ion_chronos\tools'
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

   # worker (separate shell)
   python worker.py

   # frontend
   cd C:\ion_chronos\workspace\frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 5173

4) Open frontend at http://localhost:5173 and start a job (use small timesteps).
'@
Write-SafeFile (Join-Path $ROOT "README.md") $readme

Write-Output "Advanced scaffold files written (no files overwritten). Review any *.new files for conflicts."
Write-Output "Now: create .env from backend\.env.example, then run docker-compose or start services manually."
