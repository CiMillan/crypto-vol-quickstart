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
import argparse, os, sys, json, time, pathlib, textwrap
from datetime import datetime
from typing import Dict, Any, List

# %%
import pandas as pd

# %%
# Hard dependency only at runtime when actually calling Dune
try:
    from dune_client.client import DuneClient
except Exception:  # pragma: no cover
    DuneClient = None  # type: ignore

# %%
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # local import to avoid forcing PyYAML unless you use this script
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# %%
def read_sql(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# %%
def fill_params(sql: str, params: Dict[str, str]) -> str:
    # Very simple token replacement for {{start}} / {{end}}
    for k, v in params.items():
        sql = sql.replace("{{" + k + "}}", v)
    return sql

# %%
def run_dune_sql(client: "DuneClient", sql: str) -> pd.DataFrame:
    """
    Uses dune-client's SQL execution. Some versions expose `client.run_sql(sql)`.
    If not available, try `client.execute_sql(sql)` as a fallback.
    """
    if hasattr(client, "run_sql"):
        res = client.run_sql(sql)  # returns a ResultResponse with .result.rows
        rows = res.get("result", {}).get("rows") if isinstance(res, dict) else getattr(res, "result", {}).get("rows", [])
    elif hasattr(client, "execute_sql"):
        res = client.execute_sql(sql)
        rows = res.get("result", {}).get("rows", [])
    else:
        raise RuntimeError("Your dune-client version does not support run_sql/execute_sql. Please upgrade `dune-client`.")
    if rows is None:
        rows = []
    return pd.DataFrame(rows)

# %%
def ensure_outdir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# %%
def write_parquet(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    # If empty, still write a valid file with schema
    if df.empty:
        # attempt to infer schema-less by adding a dummy row and removing it
        df = df.head(0)
    df.to_parquet(out_path, index=False)

# %%
def main() -> int:
    ap = argparse.ArgumentParser(description="Export on-chain queries from Dune to Parquet.")
    ap.add_argument("--jobs", default="analytics/onchain/config/onchain_jobs.yaml", help="YAML with job list")
    ap.add_argument("--outdir", default="data/processed/onchain", help="Output directory")
    ap.add_argument("--start", dest="start", default=os.getenv("ONCHAIN_DEFAULT_START", ""), help="ISO8601, overrides {{start}}")
    ap.add_argument("--end", dest="end", default=os.getenv("ONCHAIN_DEFAULT_END", ""), help="ISO8601, overrides {{end}}")
    ap.add_argument("--job", dest="only", default="", help="Only run a single job name")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between jobs (API courtesy)")
    args = ap.parse_args()

    dune_key = os.getenv("DUNE_API_KEY")
    if not dune_key:
        print("ERROR: DUNE_API_KEY not set (put it in your environment or .env).", file=sys.stderr)
        return 2

    cfg = load_yaml(args.jobs)
    job_list: List[Dict[str, Any]] = cfg.get("jobs", [])
    if args.only:
        job_list = [j for j in job_list if j.get("name") == args.only]
        if not job_list:
            print(f"ERROR: job '{args.only}' not found in {args.jobs}", file=sys.stderr)
            return 3

    # Simple param pack
    param_map = {}
    if args.start:
        param_map["start"] = args.start
    if args.end:
        param_map["end"] = args.end

    client = DuneClient(api_key=dune_key) if DuneClient else None
    if client is None:
        print("ERROR: dune-client is not installed/available. Add it to requirements and reinstall.", file=sys.stderr)
        return 4

    outdir = pathlib.Path(args.outdir)
    ensure_outdir(outdir)

    summary = []
    for job in job_list:
        name = job["name"]
        sql_path = job["sql"]
        outfile = job["outfile"]
        print(f"→ {name}: {sql_path}")

        sql = read_sql(sql_path)
        sql_filled = fill_params(sql, param_map) if param_map else sql

        try:
            df = run_dune_sql(client, sql_filled)
            out_path = outdir / outfile
            write_parquet(df, out_path)
            n = len(df)
            print(f"   wrote {n} rows → {out_path}")
            summary.append({"job": name, "rows": n, "out": str(out_path)})
        except Exception as e:
            print(f"   ERROR {name}: {e}", file=sys.stderr)
            summary.append({"job": name, "error": str(e)})
        if args.sleep:
            time.sleep(args.sleep)

    # machine-readable summary for logs
    print(json.dumps({"ok": True, "summary": summary}, indent=2))
    return 0

# %%
if __name__ == "__main__":
    raise SystemExit(main())
