# apply_frontend_overwrite.ps1
$ErrorActionPreference = "Stop"
$workspace = "C:\ion_chronos\workspace"
$frontend = Join-Path $workspace "frontend"
$src = Join-Path $frontend "src"
$components = Join-Path $src "components"
$timestamp = Get-Date -Format yyyyMMdd_HHmmss
$backup = Join-Path $workspace ("frontend_backup_$timestamp")

if (-not (Test-Path $frontend)) {
  Write-Output "Creating frontend folder: $frontend"
  New-Item -ItemType Directory -Path $frontend -Force | Out-Null
}
if (-not (Test-Path $src)) { New-Item -ItemType Directory -Path $src -Force | Out-Null }
if (-not (Test-Path $components)) { New-Item -ItemType Directory -Path $components -Force | Out-Null }

# Backup existing frontend
Write-Output "Backing up existing frontend to: $backup"
Copy-Item -Path $frontend -Destination $backup -Recurse -Force

function WriteFile([string]$path, [string]$content) {
  $dir = Split-Path $path -Parent
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $content | Out-File -FilePath $path -Encoding utf8 -Force
  Write-Output "WROTE: $path"
}

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
	"react-dom": "^18.2.0",
	"plotly.js-dist-min": "^2.24.1"
  },
  "devDependencies": {
	"vite": "^5.0.0"
  }
}
'@
WriteFile (Join-Path $frontend "package.json") $pkg
                                                                                                                                                                                                     
# index.html
$index = @'
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
WriteFile (Join-Path $frontend "index.html") $index
                                                                                                                                                                                                     
# src/main.jsx
$main = @'
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
createRoot(document.getElementById("root")).render(<App />);
'@
WriteFile (Join-Path $src "main.jsx") $main
                                                                                                                                                                                                     
# src/App.jsx
$app = @'
import React, {useState, useEffect} from "react";
import JobForm from "./components/JobForm";
import JobList from "./components/JobList";
import Artifacts from "./components/Artifacts";

export default function App(){
  const [jobs, setJobs] = useState([]);
  const fetchJobs = async ()=> {
    try {
      const r = await fetch("/api/jobs");
      const j = await r.json();
      setJobs(Object.values(j || {}));
    } catch(e) { console.error("fetchJobs", e) }
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
          <h3>Artifacts & Logs</h3>
          <Artifacts />
        </section>
      </main>
    </div>
  );
}
'@
WriteFile (Join-Path $src "App.jsx") $app
                                                                                                                                                                                                     
# styles.css
$styles = @'
:root { --bg:#0f172a; --card:rgba(255,255,255,0.03); --accent:#0ea5a3; --muted:#94a3b8; --accent2:#f59e0b; }
body { background:var(--bg); color:#e6eef6; font-family:Inter,system-ui; margin:0;}
.topbar{background:#071026;padding:12px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.5);}
.container{display:flex;gap:16px;padding:20px}
.left{width:520px}
.right{flex:1}
.card{background:var(--card); padding:14px; border-radius:10px; margin-bottom:12px}
.job{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.02)}
.logviewer pre{height:360px;overflow:auto;background:#031025;padding:12px;border-radius:8px}
button{background:var(--accent);border:none;padding:8px 12px;border-radius:8px;color:#042022;cursor:pointer}
input, select{width:100%;padding:8px;margin:6px 0;border-radius:6px;border:1px solid #233;background:#071826;color:#e6eef6}
a{color:var(--accent2);margin-left:8px}
.small{font-size:0.9rem;color:var(--muted)}
'@
WriteFile (Join-Path $src "styles.css") $styles
                                                                                                                                                                                                     
# components/JobForm.jsx
$jobform = @'
import React, {useState} from "react";

export default function JobForm({onStarted}){
  const [ticker,setTicker] = useState("SPY");
  const [timesteps,setTimesteps] = useState(50);

  async function submit(){
    try {
      const res = await fetch("/api/jobs/start", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ticker, timesteps, start_date:"2015-01-01"})
      });
      const j = await res.json();
      alert("Job submitted: " + j.job_id);
      onStarted && onStarted();
    } catch(e) {
      alert("Submit failed: " + e);
    }
  }

  return (
    <div className="card">
      <h3>Start Job</h3>
      <label>Ticker</label>
      <input value={ticker} onChange={e=>setTicker(e.target.value)} />
      <label>Timesteps (small for sanity)</label>
      <input type="number" value={timesteps} onChange={e=>setTimesteps(Number(e.target.value))} />
      <div style={{display:"flex",gap:8}}>
        <button onClick={submit}>Start</button>
        <button onClick={()=>{ setTicker("SPY"); setTimesteps(50); }}>Reset</button>
      </div>
      <p className="small">Start a quick sanity job or increase timesteps for longer runs.</p>
    </div>
  );
}
'@
WriteFile (Join-Path $components "JobForm.jsx") $jobform
                                                                                                                                                                                                     
# components/JobList.jsx
$joblist = @'
import React, {useState} from "react";
import LogViewer from "./LogViewer";

export default function JobList({jobs, refresh}) {
  const [selected, setSelected] = useState(null);
  return (
    <div className="card">
      <h3>Jobs</h3>
      <ul>
        {jobs.length===0 && <li className="small">No jobs yet</li>}
        {jobs.map(job => (
          <li key={job.id} className="job">
            <div>
              <strong>{job.id.slice(0,8)}</strong> &nbsp; <span className="small">{job.status}</span><br/>
              <span className="small">{job.spec?.ticker} • {job.spec?.timesteps} timesteps</span>
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
WriteFile (Join-Path $components "JobList.jsx") $joblist
                                                                                                                                                                                                     
# components/LogViewer.jsx
$logviewer = @'
import React, {useEffect, useState, useRef} from "react";

export default function LogViewer({job, onClose}){
  const [lines, setLines] = useState("");
  const wsRef = useRef(null);

  useEffect(()=> {
    let mounted = true;
    async function start() {
      setLines("");
      const r = await fetch(`/api/jobs/${job.id}/artifacts`);
      const data = await r.json();
      const run = data.artifacts && data.artifacts.find(a=>a.name==="run.log");
      if (!run) { setLines("No run.log yet"); return; }
      const url = `ws://${window.location.host}/ws/logs?path=${encodeURIComponent(run.path)}`;
      wsRef.current = new WebSocket(url);
      wsRef.current.onmessage = (ev) => {
        if (!mounted) return;
        setLines(prev => (prev + ev.data).slice(-40000));
      };
      wsRef.current.onclose = ()=> setLines(prev => prev + "\n[WS closed]");
    }
    start();
    return ()=> { mounted = false; wsRef.current && wsRef.current.close(); };
  }, [job.id]);

  return (
    <div className="logviewer card">
      <h4>Logs - {job.id}</h4>
      <pre>{lines || "waiting for logs..."}</pre>
      <div style={{display:"flex",gap:8}}>
        <button onClick={onClose}>Close</button>
        <button onClick={()=>{ navigator.clipboard.writeText(lines); }}>Copy</button>
      </div>
    </div>
  );
}
'@
WriteFile (Join-Path $components "LogViewer.jsx") $logviewer
                                                                                                                                                                                                     
# components/Artifacts.jsx
$artifacts = @'
import React, {useState} from "react";

export default function Artifacts(){
  const [jobId, setJobId] = useState("");
  const [files, setFiles] = useState([]);

  async function load(){
    if(!jobId) return alert("Enter job id");
    const r = await fetch(`/api/jobs/${jobId}/artifacts`);
    const j = await r.json();
    setFiles(j.artifacts || []);
  }

  return (
    <div className="card">
      <h3>Artifacts</h3>
      <input placeholder="job id" value={jobId} onChange={e=>setJobId(e.target.value)} />
      <button onClick={load}>Load</button>
      <ul>
        {files.map(f => (
          <li key={f.path}>
            <a href={`/api/artifact?path=${encodeURIComponent(f.path)}`} target="_blank" rel="noreferrer">{f.name}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}
'@
WriteFile (Join-Path $components "Artifacts.jsx") $artifacts
                                                                                                                                                                                                     
# components/EquityChart.jsx (minimal placeholder)
$chart = @'
import React from "react";
import Plotly from "plotly.js-dist-min";

export default function EquityChart({csvPath}) {
  return (
    <div className="card">
      <h3>Equity (preview)</h3>
      <div className="small">Use Artifacts → download trades.csv and upload to plotter (future)</div>
    </div>
  );
}
'@
WriteFile (Join-Path $components "EquityChart.jsx") $chart
                                                                                                                                                                                                     
  Write-Output "Frontend overwrite complete. Backup stored at: $backup"                                                                                                                              
  Write-Output "Now restart the frontend dev server and hard refresh the browser."