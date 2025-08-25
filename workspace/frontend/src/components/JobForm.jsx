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
