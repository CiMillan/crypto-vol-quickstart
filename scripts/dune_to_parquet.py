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
import os, sys, argparse, json, time
from pathlib import Path
import pandas as pd
from dune_client.client import DuneClient
from dune_client.models import DuneError, ExecutionState

# %%
def parse_params(kvs):
    d = {}
    for kv in kvs or []:
        if "=" not in kv:
            print(f"[WARN] Ignoring bad --param '{kv}' (expected key=value)", file=sys.stderr)
            continue
        k, v = kv.split("=", 1)
        d[k] = v
    return d

# %%
class SimpleQuery:
    def __init__(self, query_id: int, params: dict):
        self.query_id = query_id
        self._params = params or {}
    def request_format(self):
        return {"query_id": self.query_id, "query_parameters": self._params}

# %%
def run_with_fallbacks(client: DuneClient, query: SimpleQuery, perf_order):
    last_err = None
    for perf in perf_order:
        try:
            df = client.run_query_dataframe(query, performance=perf)
            print(f"[INFO] Executed query {query.query_id} on '{perf}' tier")
            return df
        except DuneError as e:
            # Try to show the underlying payload
            payload = getattr(e, "payload", None)
            print(f"[WARN] Tier '{perf}' failed with DuneError payload: {payload}", file=sys.stderr)
            last_err = e
        except TypeError:
            try:
                df = client.run_query_dataframe(query)
                print(f"[INFO] Executed query {query.query_id} (no perf kw)")
                return df
            except Exception as e2:
                last_err = e2
    if last_err:
        raise last_err
    raise RuntimeError("Failed to execute query with provided tiers")

# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--performance", default="medium")
    ap.add_argument("--param", action="append", help="key=value (repeatable)")
    args = ap.parse_args()

    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        print("DUNE_API_KEY env var missing", file=sys.stderr); sys.exit(2)

    dune = DuneClient(api_key=api_key)
    params = parse_params(args.param)
    query = SimpleQuery(args.query_id, params)

    perf_order = []
    for p in (args.performance, "medium", "small"):
        if p not in perf_order:
            perf_order.append(p)

    df = run_with_fallbacks(dune, query, perf_order)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for col in ("ts", "ts_block", "first_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df.to_parquet(out, index=False)
    print(f"Saved {out} rows={len(df)} cols={len(df.columns)}")

# %%
if __name__ == "__main__":
    main()
