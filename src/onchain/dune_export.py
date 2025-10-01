# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %%

import argparse, os, sys, time, pathlib, json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml
import requests


def to_iso(s: Optional[str]) -> Optional[str]:
    return s if s else None


def run_dune_saved_query(api_key: str, query_id: int, params: Dict[str, Any]) -> pd.DataFrame:
    base = "https://api.dune.com/api/v1"
    headers = {"X-DUNE-API-KEY": api_key, "Content-Type": "application/json"}

    # execute
    r = requests.post(f"{base}/query/{query_id}/execute", headers=headers, json={"query_parameters": params or {}}, timeout=60)
    r.raise_for_status()
    j = r.json()
    exe_id = j.get("execution_id")
    if not exe_id:
        raise RuntimeError(f"Execute failed: {j}")

    # poll with 429 backoff
    delay, max_delay = 1.0, 10.0
    for _ in range(240):
        sr = requests.get(f"{base}/execution/{exe_id}/status", headers=headers, timeout=30)
        if sr.status_code == 429:
            time.sleep(delay); delay = min(delay*1.6, max_delay); continue
        sr.raise_for_status()
        st = sr.json()
        state = st.get("state") or st.get("execution_state")
        if state in {"COMPLETED","QUERY_STATE_COMPLETED"}: break
        if state in {"FAILED","CANCELLED","QUERY_STATE_FAILED"}:
            raise RuntimeError(f"Query failed: {st}")
        time.sleep(min(delay, max_delay))
    else:
        raise TimeoutError("Timed out waiting for Dune execution to complete.")

    # fetch with 429 backoff
    delay = 1.0
    while True:
        rr = requests.get(f"{base}/execution/{exe_id}/results", headers=headers, timeout=60)
        if rr.status_code == 429:
            time.sleep(delay); delay = min(delay*1.6, max_delay); continue
        rr.raise_for_status()
        break

    res = rr.json()
    rows = (res.get("result") or {}).get("rows") or res.get("rows") or []
    return pd.DataFrame(rows)


def load_jobs_yaml(path: str) -> List[Dict[str, Any]]:
    data = yaml.safe_load(open(path, "r"))
    jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(jobs, list):
        raise ValueError("Invalid jobs YAML: expected top-level 'jobs:' list.")
    return jobs


def ensure_outdir(d: str) -> None:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, out_path: str) -> None:
    df.to_parquet(out_path, index=False)


def parse_iso(s: str) -> datetime:
    # naive 'Z' handling
    if s.endswith('Z'): s = s[:-1]
    return datetime.fromisoformat(s.replace('Z','')).replace(tzinfo=timezone.utc)


def daterange(start: datetime, end: datetime, step: timedelta):
    t = start
    while t < end:
        n = min(t + step, end)
        yield t, n
        t = n


def run_job(api_key: str, job: Dict[str, Any], start: Optional[str], end: Optional[str], sleep: float) -> Dict[str, Any]:
    name = job.get("name")
    qid  = job.get("query_id")
    out  = job.get("outfile")
    chunk_hours = float(job.get("chunk_hours", 0))  # 0 = no chunking
    chunk_minutes = float(job.get("chunk_minutes", 0))  # 0 = no chunking

    if not (name and qid and out):
        return {"job": name, "error": "Job missing name/query_id/outfile"}

    out_path = str(pathlib.Path(out)) if out.startswith("data/") else str(pathlib.Path("data/processed/onchain")/out)
    print(f"→ {name}: query_id={qid} → {out_path}", file=sys.stderr)

    # if no chunking requested or no start/end provided, do a single call
    if (not chunk_hours and not chunk_minutes) or not (start and end):
        df = run_dune_saved_query(api_key, int(qid), {"start": to_iso(start), "end": to_iso(end)})
        write_parquet(df, out_path)
        return {"job": name, "rows": int(len(df)), "outfile": out_path}

    # chunked pulls
    start_dt, end_dt = parse_iso(start), parse_iso(end)
    step = (timedelta(minutes=chunk_minutes) if chunk_minutes else timedelta(hours=chunk_hours))
    parts = []
    for i, (s_dt, e_dt) in enumerate(daterange(start_dt, end_dt, step), 1):
        s_iso = s_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        e_iso = e_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            dfp = run_dune_saved_query(api_key, int(qid), {"start": s_iso, "end": e_iso})
            if not dfp.empty:
                parts.append(dfp)
            print(f"   chunk {i}: {s_iso} → {e_iso} rows={len(dfp)}", file=sys.stderr)
            if sleep: time.sleep(sleep)
        except requests.HTTPError as he:
            # If 402 on a chunk, tell the user and stop (can reduce chunk_hours further)
            return {"job": name, "error": f"HTTPError on chunk {s_iso}..{e_iso}: {he}"}
        except Exception as e:
            return {"job": name, "error": f"Chunk {s_iso}..{e_iso} failed: {e}"}

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_parquet(df, out_path)
    return {"job": name, "rows": int(len(df)), "outfile": out_path}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dune saved-query exporter (free-tier friendly, with chunking).")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--job")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between calls (and chunks).")
    args = ap.parse_args()

    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        print("ERROR: DUNE_API_KEY not set.", file=sys.stderr)
        return 2

    jobs = load_jobs_yaml(args.jobs)
    ensure_outdir(args.outdir)

    summary = []
    for job in jobs:
        if args.job and job.get("name") != args.job:
            continue
        # run and write using outdir prefix
        outfile = job.get("outfile"); 
        job["outfile"] = str(pathlib.Path(args.outdir)/outfile)
        res = run_job(api_key, job, args.start, args.end, args.sleep)
        summary.append(res)

    ok = all("error" not in s for s in summary if s.get("job"))
    print(json.dumps({"ok": ok, "summary": summary}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
