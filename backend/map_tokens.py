"""
Download Angel One instrument master and map ALL Nifty 500 symbols.
Correct URL: https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json
"""
import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

# Download instrument master
url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
print(f"📥 Downloading instrument master from Angel One...")
resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
instruments = resp.json()
print(f"✅ Got {len(instruments)} total instruments")

# Build NSE-EQ lookup: symbol -> {token, symbol, name}
nse_eq = {}
for inst in instruments:
    seg = inst.get("exch_seg", "")
    sym = inst.get("symbol", "")
    if seg == "NSE" and sym.endswith("-EQ"):
        base = sym.replace("-EQ", "")
        nse_eq[base] = {
            "token": inst["token"],
            "trading_symbol": sym,
            "name": inst.get("name", ""),
        }

print(f"📊 NSE-EQ instruments: {len(nse_eq)}")

# Load Nifty 500 list
from data.nifty500_full import NIFTY_500_STOCKS

token_map = {}
missing = []

for stock in NIFTY_500_STOCKS:
    sym = stock["symbol"].replace(".NS", "")
    if sym in nse_eq:
        entry = nse_eq[sym].copy()
        entry["name"] = stock["name"]
        entry["sector"] = stock["sector"]
        token_map[sym] = entry
    else:
        missing.append(sym)

print(f"\n✅ Mapped: {len(token_map)}/{len(NIFTY_500_STOCKS)} stocks")

if missing:
    print(f"⚠️  Missing: {len(missing)}")
    for m in missing:
        print(f"   {m}")

# Save
with open("data/angel_tokens.json", "w") as f:
    json.dump(token_map, f, indent=2)
print(f"\n💾 Saved to data/angel_tokens.json")

# Show samples
print("\nSample mappings:")
for sym in ["TCS", "RELIANCE", "HDFCBANK", "INFY", "SBIN", "BAJFINANCE", "ITC", "ADANIENT", "WIPRO", "LT"]:
    if sym in token_map:
        t = token_map[sym]
        print(f"   {sym:15s} → token={t['token']:>8s}  ({t['trading_symbol']})")

print(f"\n✅ Done! {len(token_map)} stocks ready for Angel One API.")
