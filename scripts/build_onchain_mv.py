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
from pathlib import Path
import argparse, sys
from typing import Optional
from src.features.onchain import build_onchain_mv_feature_set

# %%
def opt(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    print(f"[WARN] Missing input: {p} (skipping)", file=sys.stderr)
    return None

# %%
p = argparse.ArgumentParser()
p.add_argument("--start", required=True)
p.add_argument("--end", required=True)
p.add_argument("--out", default="data/processed/onchain/mv_feature_set_hourly.parquet")
p.add_argument("--transfers",   default="data/processed/onchain/transfers.parquet")
p.add_argument("--labels",      default="data/processed/onchain/labels/exchanges.json")
p.add_argument("--swaps",       default="data/processed/onchain/uniswap_swaps.parquet")
p.add_argument("--blocks",      default="data/processed/onchain/eth_blocks.parquet")
p.add_argument("--first_seen",  default="data/processed/onchain/address_first_seen.parquet")
args = p.parse_args()

# %%
df = build_onchain_mv_feature_set(
    ts_start=args.start,
    ts_end=args.end,
    transfers_path=opt(args.transfers),
    labels_exchanges_path=opt(args.labels),
    uniswap_swaps_path=opt(args.swaps),
    eth_blocks_path=opt(args.blocks),
    address_first_seen_path=opt(args.first_seen),
)

# %%
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out)
print(f"Saved {out} rows={len(df)} cols={len(df.columns)}")
