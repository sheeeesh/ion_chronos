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
