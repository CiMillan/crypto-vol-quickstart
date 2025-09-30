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
import argparse, json
from scripts.dune_to_parquet import parse_params, SimpleQuery  # re-use helpers
ap = argparse.ArgumentParser()
ap.add_argument("--query-id", type=int, required=True)
ap.add_argument("--param", action="append")
args = ap.parse_args()
params = parse_params(args.param)
q = SimpleQuery(args.query_id, params)
print(json.dumps(q.request_format(), indent=2))
