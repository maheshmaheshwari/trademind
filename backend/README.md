---
title: TradeMind Backend
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# TradeMind AI — Backend

Private Docker Space running the full TradeMind backend: FastAPI, APScheduler
(collectors, EOD jobs, nightly signal generation, weekend retrain), and the ML
inference stack.

- Models are pulled at boot from the encrypted `trademind-models` Hub repo
  (`scripts/model_store.py sync`); nothing persistent lives on this Space's disk.
- All persistent state is in Timescale Cloud (`trade_signals` table etc.).
- Deployed/redeployed with `python scripts/deploy_space.py push` from the
  development machine — see `docs/DEPLOYMENT.md` in the main repository.
