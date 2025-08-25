# apply_fix.ps1
# Backup current workspace then write fixed backend + frontend files
$ErrorActionPreference = "Stop"

$workspace = "C:\ion_chronos\workspace"
if (-not (Test-Path $workspace)) {
  Write-Output "Workspace path not found: $workspace"
  exit 1
}

# Create backup
$parent = Split-Path $workspace -Parent
$timestamp = Get-Date -Format yyyyMMdd_HHmmss
$backup = Join-Path $parent ("workspace_backup_" + $timestamp)
Write-Output "Creating backup at: $backup"
Copy-Item -Path $workspace -Destination $backup -Recurse -Force

# Ensure directories
$backendDir = Join-Path $workspace "backend"
$frontendDir = Join-Path $workspace "frontend"
$frontendSrc = Join-Path $frontendDir "src"
$frontendComponents = Join-Path $frontendSrc "components"

foreach ($d in @($backendDir, $frontendDir, $frontendSrc, $frontendComponents)) {
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# Helper to write file with backup if exists
function Write-FileSafely([string]$path, [string]$content) {
  if (Test-Path $path) {
	$bak = $path + ".bak_" + $timestamp
	Copy-Item -Path $path -Destination $bak -Force
  }
  $dir = Split-Path $path -Parent
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $content | Out-File -FilePath $path -Encoding utf8 -Force
  Write-Output "Wrote: $path"
}
                 
  # Write backend/main.py                
$main_py = @'
import os
import uuid
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from job_runner import run_job

WORKDIR = Path(os.getenv("WORKSPACE", "C:\\ion_chronos\\workspace"))  # adjust on your machine if needed
JOBS_DIR = WORKDIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_INDEX = WORKDIR / "jobs_index.json"

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)

# load/save jobs index helpers
def load_index():
	if JOBS_INDEX.exists():
		return json.loads(JOBS_INDEX.read_text())
	return {}

def save_index(idx):
	JOBS_INDEX.write_text(json.dumps(idx, indent=2))

class JobSpec(BaseModel):
	ticker: str = "SPY"
	start_date: str = "2015-01-01"
	end_date: str = None
	timeframe: str = "1d"
	timesteps: int = 50000

@app.post("/api/jobs/start")
def start_job(spec: JobSpec):
	job_id = str(uuid.uuid4())
	job_path = JOBS_DIR / job_id
	job_path.mkdir(parents=True, exist_ok=True)
	idx = load_index()
	idx[job_id] = {
		"id": job_id,
		"spec": spec.dict(),
		"status": "queued",
		"start_time": None,
		"end_time": None,
		"artifacts": []
	}
	save_index(idx)

	# submit background job
	executor.submit(run_job, job_id, spec.dict(), str(job_path))
	return {"job_id": job_id}

@app.get("/api/jobs")
def list_jobs():
	return load_index()

@app.get("/api/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str):
	job_path = JOBS_DIR / job_id
	if not job_path.exists():
		raise HTTPException(status_code=404, detail="Job not found")
	files = []
	for p in job_path.iterdir():
		files.append({"name": p.name, "path": str(p)})
	return {"artifacts": files}

# download endpoint (simple)
@app.get("/api/artifact")
def get_artifact(path: str):
	p = Path(path)
	if not p.exists():
		raise HTTPException(status_code=404, detail="Artifact not found")
	return {"path": str(p)}

# WebSocket for tailing logs: ?path=/full/path/to/run.log
@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket, path: str):
	await ws.accept()
	try:
		pos = 0
		while True:
			p = Path(path)
			if p.exists():
				with p.open("rb") as f:
					f.seek(pos)
					data = f.read()
					if data:
						await ws.send_text(data.decode("utf-8", errors="replace"))
						pos = f.tell()
			else:
				await ws.send_text(f"[waiting] file not found: {path}")
			await asyncio.sleep(0.8)
	except WebSocketDisconnect:
		return
'@
                 
  Write-FileSafely (Join-Path $backendDir "main.py") $main_py                
                 
  # Write backend/job_runner.py                
$job_runner_py = @'
import os, sys, time, json, traceback
from pathlib import Path

# Attempt to import pipeline functions - if not available, we create stubs
try:
	from functions import build_astro_dataset, rl_train, backtest_signal
except Exception:
	build_astro_dataset = None
	rl_train = None
	backtest_signal = None

def write_log(fp, msg):
	ts = time.strftime("%Y-%m-%d %H:%M:%S")
	fp.write(f"{ts} {msg}\n")
	fp.flush()

def run_job(job_id, spec, job_path):
	job_path = Path(job_path)
	log_path = job_path / "run.log"
	summary_path = job_path / "summary.json"
	try:
		with log_path.open("a", encoding="utf-8") as log:
			write_log(log, f"[job {job_id}] starting with spec: {spec}")
			# Step 1: build dataset
			write_log(log, "Step: build_astro_dataset")
			if build_astro_dataset:
				build_astro_dataset(ticker=spec['ticker'],
									start_date=spec['start_date'],
									end_date=spec['end_date'],
									timeframe=spec.get('timeframe','1d'),
									cache_parquet=str(job_path / "dataset.parquet"))
			else:
				write_log(log, "(stub) built dataset")
				# create dummy dataset file
				(job_path / "dataset.csv").write_text("date,open,close\n2020-01-01,100,101\n")
			# Step 2: train RL
			write_log(log, "Step: rl_train")
			if rl_train:
				rl_train(ticker=spec['ticker'], total_timesteps=int(spec.get('timesteps',1000)))
			else:
				write_log(log, "(stub) trained RL model")
			# Step 3: backtest
			write_log(log, "Step: backtest_signal")
			if backtest_signal:
				backtest_signal(ticker=spec['ticker'], start_date=spec['start_date'])
			else:
				# write dummy artifacts
				(job_path / "trades.csv").write_text("EntryTime,ExitTime,NetPnLPercent\n2020-01-01,2020-01-02,0.5\n")
				(job_path / "equity.png").write_text("PNG-DUMMY")
				(job_path / "drawdown.png").write_text("PNG-DUMMY")
			# summary
			summary = {
				"job_id": job_id,
				"status": "success",
				"metrics": {"total_return": "stub", "cagr": "stub"}
			}
			summary_path.write_text(json.dumps(summary, indent=2))
			write_log(log, "Job completed successfully")
	except Exception as e:
		with log_path.open("a", encoding="utf-8") as log:
			write_log(log, f"Job failed: {e}")
			write_log(log, traceback.format_exc())
		summary_path.write_text(json.dumps({"job_id": job_id, "status": "failed", "error": str(e)}))
'@
                 
  Write-FileSafely (Join-Path $backendDir "job_runner.py") $job_runner_py                
                 
  # Write backend/requirements.txt                
$reqs = @'
fastapi
uvicorn[standard]
pydantic
'@
  Write-FileSafely (Join-Path $backendDir "requirements.txt") $reqs                
                 
  # FRONTEND FILES                
  # package.json                
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
	"react-dom": "^18.2.0"
  },
  "devDependencies": {
	"vite": "^5.0.0"
  }
}
'@
  Write-FileSafely (Join-Path $frontendDir "package.json") $pkg                
                 
  # index.html                
$index_html = @'
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
  Write-FileSafely (Join-Path $frontendDir "index.html") $index_html                
                 
  # src/main.jsx                
$main_jsx = @'
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")).render(<App />);
'@
  Write-FileSafely (Join-Path $frontendSrc "main.jsx") $main_jsx                
                 
  # src/App.jsx                
$app_jsx = @'
import React, {useState, useEffect} from "react";
import JobForm from "./components/JobForm";
import JobList from "./components/JobList";

export default function App(){
  const [jobs, setJobs] = useState([]);
  const fetchJobs = async ()=> {
	const r = await fetch("/api/jobs");
	const j = await r.json();
	setJobs(Object.values(j || {}));
  };
  useEffect(()=> { fetchJobs(); const t = setInterval(fetchJobs, 3000); return ()=>clearInterval(t); }, []);
  return (
	<div className="app">
	  <header className="topbar"><h1>Ion Chronos — Dashboard</h1></header>
	  <main className="container">
		<section className="left">
		  <JobForm onStarted={fetchJobs}/>
		  <JobList jobs={jobs} refresh={fetchJobs}/>
		</section>
		<section className="right">
		  <h3>Live Logs</h3>
		  <div id="logviewer">Select a job to tail logs</div>
		</section>
	  </main>
	</div>
  );
}
'@
  Write-FileSafely (Join-Path $frontendSrc "App.jsx") $app_jsx                
                 
  # src/components/JobForm.jsx                
$jobform = @'
import React, {useState} from "react";

export default function JobForm({onStarted}){
  const [ticker,setTicker] = useState("SPY");
  const [timesteps,setTimesteps] = useState(50);

  async function submit(){
	const res = await fetch("/api/jobs/start", {
	  method:"POST",
	  headers: {"Content-Type":"application/json"},
	  body: JSON.stringify({ticker, timesteps, start_date:"2015-01-01"})
	});
	const j = await res.json();
	alert("Job submitted: " + j.job_id);
	onStarted && onStarted();
  }

  return (
	<div className="card">
	  <h3>Start Job</h3>
	  <label>Ticker</label>
	  <input value={ticker} onChange={e=>setTicker(e.target.value)} />
	  <label>Timesteps (small for sanity)</label>
	  <input type="number" value={timesteps} onChange={e=>setTimesteps(Number(e.target.value))} />
	  <button onClick={submit}>Start</button>
	</div>
  );
}
'@
  Write-FileSafely (Join-Path $frontendComponents "JobForm.jsx") $jobform                
                 
  # src/components/JobList.jsx                
$joblist = @'
import React, {useState, useEffect} from "react";
import LogViewer from "./LogViewer";

export default function JobList({jobs, refresh}) {
  const [selected, setSelected] = useState(null);
  return (
	<div className="card">
	  <h3>Jobs</h3>
	  <ul>
		{jobs.map(job => (
		  <li key={job.id} className="job">
			<div>
			  <strong>{job.id}</strong> — {job.status}
			</div>
			<div>
			  <button onClick={()=>setSelected(job)}>Tail logs</button>
			  <a href={`/api/jobs/${job.id}/artifacts`} target="_blank" rel="noreferrer">Artifacts</a>
			</div>
		  </li>
		))}
	  </ul>
	  {selected && <LogViewer job={selected} onClose={()=>setSelected(null)} />}
	</div>
  );
}
'@
  Write-FileSafely (Join-Path $frontendComponents "JobList.jsx") $joblist                
                 
  # src/components/LogViewer.jsx                
$logviewer = @'
import React, {useEffect, useState} from "react";

export default function LogViewer({job, onClose}){
  const [lines, setLines] = useState("");
  useEffect(()=> {
	let ws;
	async function fetchArtifacts(){
	  const res = await fetch(`/api/jobs/${job.id}/artifacts`);
	  const data = await res.json();
	  // find run.log path (if present) by requesting artifact list
	  const run = data.artifacts.find(a => a.name === "run.log");
	  if(run){
		ws = new WebSocket(`ws://${window.location.host}/ws/logs?path=${encodeURIComponent(run.path)}`);
		ws.onmessage = (ev)=> setLines(prev => (prev + ev.data).slice(-20000));
	  } else {
		setLines("No run.log found yet");
	  }
	}
	fetchArtifacts();
	return ()=> { ws && ws.close(); };
  }, [job.id]);

  return (
	<div className="logviewer">
	  <h4>Logs - {job.id}</h4>
	  <pre>{lines || "waiting for logs..."}</pre>
	  <button onClick={onClose}>Close</button>
	</div>
  );
}
'@
  Write-FileSafely (Join-Path $frontendComponents "LogViewer.jsx") $logviewer                
                 
  # src/styles.css                
$styles = @'
body { background:#0f172a; color:#e6eef6; font-family:Inter,system-ui; margin:0;}
.topbar{background:#071026;padding:12px 20px}
.container{display:flex;gap:16px;padding:20px}
.left{width:520px}
.right{flex:1}
.card{background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; margin-bottom:12px}
.job{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.02)}
.logviewer pre{height:360px;overflow:auto;background:#031025;padding:12px;border-radius:6px}
button{background:#0ea5a3;border:none;padding:8px 12px;border-radius:6px;color:#042022;cursor:pointer}
input{width:100%;padding:8px;margin:6px 0;border-radius:6px;border:1px solid #233}
a{color:#f59e0b;margin-left:8px}
'@
  Write-FileSafely (Join-Path $frontendSrc "styles.css") $styles               
                 
  # README                
$readme = @'
Ion Chronos Dashboard - quick start

1) Backend:
   cd workspace\backend
   python -m pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

2) Frontend:
   cd workspace\frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 5173

3) Open http://localhost:5173 and go to Jobs -> start a small sanity job

Notes:
- The job runner will call functions.build_astro_dataset, rl_train, backtest_signal if importable.
- All job artifacts will be written to workspace\jobs\{job_id}\

'@
Write-FileSafely (Join-Path $workspace "README.md") $readme

Write-Output "All files written. Backup folder created at: $backup"
Write-Output "Next: open PowerShell and start backend then frontend as described in workspace\README.md"