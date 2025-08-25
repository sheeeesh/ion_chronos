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
