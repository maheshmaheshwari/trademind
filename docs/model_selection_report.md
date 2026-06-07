# TradeMind ML Model Selection Report

> Generated from `backend/data/retrain_results.csv` — walk-forward retrain across Nifty 500 stocks.

---

## 1. Overall Summary

| Metric | Value |
|--------|-------|
| Total symbols processed | 499 |
| Successfully trained (`ok`) | 385 |
| Below threshold (trained, low recall) | 87 |
| No data | 27 |
| Errors | 0 |
| Unique model types used | 10 |

---

## 2. Model Distribution (all trained stocks)

| Model | Stocks Selected | % of Trained |
|-------|:--------------:|:-------------:|
| StackEnsemble | 101 | 21.4% |
| RandForest | 93 | 19.7% |
| Ensemble | 79 | 16.7% |
| XGB_HiReg | 55 | 11.7% |
| XGBoost | 50 | 10.6% |
| LGB_HiReg | 28 | 5.9% |
| TabNet | 24 | 5.1% |
| GradBoost | 16 | 3.4% |
| CatBoost | 15 | 3.2% |
| LightGBM | 11 | 2.3% |

---

## 3. Best Horizon Distribution (`ok` stocks only)

| Horizon | Count | % |
|---------|:-----:|:---:|
| 1 Week | 24 | 6.2% |
| 2 Weeks | 43 | 11.2% |
| 1 Month | 42 | 10.9% |
| 2 Months | 69 | 17.9% |
| 3 Months | 82 | 21.3% |
| 6 Months | 125 | 32.5% |

---

## 4. Average Metrics by Model (`ok` stocks)

| Model | Count | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 |
|-------|:-----:|:------------:|:-------------:|:----------:|:------:|
| StackEnsemble | 97 | 0.8069 | 0.8148 | 0.3451 | 0.4121 |
| RandForest | 81 | 0.7866 | 0.7773 | 0.1858 | 0.2548 |
| Ensemble | 60 | 0.7943 | 0.7432 | 0.1735 | 0.2413 |
| XGB_HiReg | 48 | 0.7695 | 0.8070 | 0.3245 | 0.3935 |
| XGBoost | 42 | 0.7477 | 0.8156 | 0.1791 | 0.2533 |
| LGB_HiReg | 21 | 0.7612 | 0.8116 | 0.2180 | 0.2705 |
| TabNet | 8 | 0.7519 | 0.6794 | 0.1788 | 0.2320 |
| GradBoost | 10 | 0.7589 | 0.7651 | 0.2059 | 0.2667 |
| CatBoost | 8 | 0.7596 | 0.7569 | 0.2843 | 0.3222 |
| LightGBM | 10 | 0.7552 | 0.8435 | 0.3732 | 0.4507 |

---

## 5. Average Metrics by Horizon (`ok` stocks)

| Horizon | Count | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 |
|---------|:-----:|:------------:|:-------------:|:----------:|:------:|
| 1 Week | 24 | 0.7382 | 0.7364 | 0.0810 | 0.1443 |
| 2 Weeks | 43 | 0.7646 | 0.7772 | 0.0761 | 0.1342 |
| 1 Month | 42 | 0.7572 | 0.7734 | 0.1290 | 0.2055 |
| 2 Months | 69 | 0.7684 | 0.7934 | 0.1754 | 0.2580 |
| 3 Months | 82 | 0.8052 | 0.7828 | 0.2620 | 0.3406 |
| 6 Months | 125 | 0.7980 | 0.8135 | 0.4150 | 0.4664 |

---

## 6. Model × Horizon Cross-Tab (`ok` stocks)

Counts of stocks where a given model was selected at each horizon.

| Model | 1W | 2W | 1M | 2M | 3M | 6M | Total |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:-----:|
| StackEnsemble | 5 | 9 | 14 | 16 | 23 | 30 | 97 |
| RandForest | 4 | 5 | 8 | 14 | 25 | 25 | 81 |
| Ensemble | 7 | 7 | 4 | 8 | 14 | 20 | 60 |
| XGB_HiReg | 3 | 0 | 6 | 10 | 5 | 24 | 48 |
| XGBoost | 1 | 5 | 8 | 12 | 4 | 12 | 42 |
| LGB_HiReg | 1 | 8 | 0 | 4 | 4 | 4 | 21 |
| TabNet | 1 | 2 | 0 | 2 | 2 | 1 | 8 |
| GradBoost | 0 | 3 | 1 | 2 | 2 | 2 | 10 |
| CatBoost | 2 | 1 | 1 | 0 | 2 | 2 | 8 |
| LightGBM | 0 | 3 | 0 | 1 | 1 | 5 | 10 |

---

## 7. Per-Stock Model Selection

Columns: **Symbol** · **Status** · **Best Model** · **Horizon** · **Accuracy** · **Precision** · **Recall** · **F1** · **Train Time (s)**

| Symbol | Status | Best Model | Horizon | Accuracy | Precision | Recall | F1 | Time (s) |
|--------|:------:|:----------:|:-------:|:--------:|:---------:|:------:|:--:|:--------:|
| 360ONE.NS | ✅ | RandForest | 6 Months | 0.8003 | 0.6226 | 0.2200 | 0.3251 | 291.1 |
| 3MINDIA.NS | ⚠️ | RandForest | 2 Months | 0.8388 | 0.4848 | — | — | 255.0 |
| AADHARHFC.NS | ✅ | Ensemble | 2 Weeks | 0.8228 | 0.7000 | 0.0551 | 0.1022 | 26.4 |
| AARTIIND.NS | ✅ | XGBoost | 1 Week | 0.7029 | 0.8519 | 0.1503 | 0.2556 | 142.6 |
| AAVAS.NS | ✅ | StackEnsemble | 2 Months | 0.8494 | 0.8333 | 0.2419 | 0.3750 | 118.8 |
| ABB.NS | ✅ | RandForest | 2 Months | 0.7575 | 0.7692 | 0.1648 | 0.2715 | 275.1 |
| ABBOTINDIA.NS | ⚠️ | XGBoost | 2 Months | 0.8268 | 0.5000 | — | — | 150.6 |
| ABCAPITAL.NS | ✅ | StackEnsemble | 1 Week | 0.7057 | 0.7500 | 0.0561 | 0.1043 | 161.0 |
| ABFRL.NS | ⚠️ | TabNet | 1 Month | 0.7602 | 0.5319 | — | — | 139.9 |
| ABLBL.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ABREL.NS | ✅ | Ensemble | 1 Month | 0.7281 | 0.6875 | 0.0573 | 0.1058 | 146.0 |
| ABSLAMC.NS | ✅ | StackEnsemble | 1 Month | 0.6287 | 0.7778 | 0.0530 | 0.0993 | 170.9 |
| ACC.NS | ✅ | XGBoost | 2 Months | 0.9013 | 0.8571 | 0.0632 | 0.1176 | 154.2 |
| ACE.NS | ✅ | Ensemble | 1 Week | 0.7339 | 0.6111 | 0.0579 | 0.1058 | 150.5 |
| ACMESOLAR.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ADANIENSOL.NS | ✅ | RandForest | 6 Months | 0.8065 | 0.8856 | 0.6642 | 0.7591 | 166.5 |
| ADANIENT.NS | ✅ | StackEnsemble | 2 Months | 0.6777 | 1.0000 | 0.0855 | 0.1575 | 163.1 |
| ADANIGREEN.NS | ✅ | XGBoost | 1 Month | 0.7003 | 1.0000 | 0.0929 | 0.1700 | 161.3 |
| ADANIPORTS.NS | ✅ | RandForest | 3 Months | 0.7376 | 0.7632 | 0.1534 | 0.2555 | 166.0 |
| ADANIPOWER.NS | ✅ | StackEnsemble | 6 Months | 0.7671 | 0.7184 | 0.8681 | 0.7862 | 172.0 |
| AEGISLOG.NS | ✅ | RandForest | 2 Months | 0.8000 | 0.7097 | 0.1507 | 0.2486 | 154.8 |
| AEGISVOPAK.NS | ❌ | — | — | — | — | — | — | 0.2 |
| AFCONS.NS | ❌ | — | — | — | — | — | — | 0.2 |
| AFFLE.NS | ✅ | XGB_HiReg | 6 Months | 0.8955 | 0.7389 | 0.9048 | 0.8135 | 164.3 |
| AGARWALEYE.NS | ❌ | — | — | — | — | — | — | 0.2 |
| AIAENG.NS | ✅ | CatBoost | 3 Months | 0.8061 | 0.7121 | 0.2338 | 0.3521 | 249.1 |
| AIIL.NS | ✅ | StackEnsemble | 1 Month | 0.7047 | 0.8571 | 0.0566 | 0.1062 | 25.2 |
| AJANTPHARM.NS | ⚠️ | Ensemble | 3 Months | 0.8028 | 0.5238 | — | — | 158.0 |
| AKUMS.NS | ✅ | XGBoost | 2 Weeks | 0.7767 | 0.9231 | 0.0723 | 0.1341 | 15.2 |
| AKZOINDIA.NS | ⚠️ | TabNet | 2 Months | 0.7937 | 0.4400 | — | — | 172.1 |
| ALKEM.NS | ✅ | Ensemble | 6 Months | 0.8510 | 0.8421 | 0.1600 | 0.2689 | 163.6 |
| ALKYLAMINE.NS | ⚠️ | StackEnsemble | 3 Months | 0.8199 | 0.5714 | — | — | 161.9 |
| ALOKINDS.NS | ✅ | StackEnsemble | 2 Weeks | 0.8703 | 1.0000 | 0.0625 | 0.1176 | 154.8 |
| AMBER.NS | ✅ | RandForest | 3 Months | 0.6770 | 0.7778 | 0.1765 | 0.2877 | 172.1 |
| AMBUJACEM.NS | ✅ | StackEnsemble | 3 Months | 0.9845 | 0.8000 | 0.5000 | 0.6154 | 151.9 |
| ANANDRATHI.NS | ✅ | RandForest | 3 Months | 0.9239 | 1.0000 | 0.0577 | 0.1091 | 156.8 |
| ANANTRAJ.NS | ⚠️ | Ensemble | 1 Month | 0.7018 | 0.5714 | — | — | 147.9 |
| ANGELONE.NS | ✅ | XGBoost | 3 Months | 0.7904 | 0.7094 | 0.7642 | 0.7358 | 167.1 |
| APARINDS.NS | ✅ | StackEnsemble | 6 Months | 0.7055 | 0.7219 | 0.7578 | 0.7394 | 170.6 |
| APLAPOLLO.NS | ✅ | RandForest | 1 Week | 0.7827 | 0.8095 | 0.0705 | 0.1298 | 356.8 |
| APLLTD.NS | ⚠️ | LGB_HiReg | 6 Months | 0.8716 | 0.5669 | — | — | 116.9 |
| APOLLOHOSP.NS | ⚠️ | Ensemble | 6 Months | 0.7568 | 0.5652 | — | — | 172.6 |
| APOLLOTYRE.NS | ✅ | StackEnsemble | 6 Months | 0.8596 | 1.0000 | 0.0682 | 0.1277 | 164.8 |
| APTUS.NS | ⚠️ | TabNet | 2 Weeks | 0.7464 | 0.5200 | — | — | 162.1 |
| ARE&M.NS | ⚠️ | CatBoost | 2 Weeks | 0.7871 | 0.4286 | — | — | 145.6 |
| ASAHIINDIA.NS | ✅ | RandForest | 3 Months | 0.7003 | 0.8000 | 0.0594 | 0.1106 | 169.4 |
| ASHOKLEY.NS | ✅ | StackEnsemble | 3 Months | 0.6180 | 0.8333 | 0.1439 | 0.2454 | 161.4 |
| ASIANPAINT.NS | ✅ | XGBoost | 2 Months | 0.7003 | 0.7222 | 0.0628 | 0.1156 | 177.3 |
| ASTERDM.NS | ✅ | LightGBM | 6 Months | 0.7072 | 0.8107 | 0.5585 | 0.6614 | 180.0 |
| ASTRAL.NS | ✅ | Ensemble | 3 Months | 0.8075 | 0.7407 | 0.1460 | 0.2439 | 161.3 |
| ASTRAZEN.NS | ✅ | StackEnsemble | 2 Weeks | 0.8415 | 0.8000 | 0.0690 | 0.1270 | 162.6 |
| ATGL.NS | ✅ | GradBoost | 2 Months | 0.7575 | 1.0000 | 0.0242 | 0.0473 | 152.2 |
| ATHERENERG.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ATUL.NS | ✅ | XGB_HiReg | 6 Months | 0.8664 | 0.8082 | 0.7024 | 0.7516 | 120.1 |
| AUBANK.NS | ✅ | LightGBM | 2 Weeks | 0.6916 | 1.0000 | 0.0228 | 0.0446 | 382.0 |
| AUROPHARMA.NS | ✅ | RandForest | 3 Months | 0.7827 | 0.7083 | 0.1328 | 0.2237 | 152.3 |
| AWL.NS | ✅ | Ensemble | 6 Months | 0.9743 | 0.8000 | 0.2222 | 0.3478 | 149.8 |
| AXISBANK.NS | ✅ | XGB_HiReg | 6 Months | 0.8202 | 0.7014 | 0.7990 | 0.7470 | 174.8 |
| BAJAJ-AUTO.NS | ⚠️ | XGBoost | 1 Month | 0.7485 | 0.5472 | — | — | 142.6 |
| BAJAJFINSV.NS | ✅ | RandForest | 6 Months | 0.8870 | 0.8000 | 0.1111 | 0.1951 | 169.5 |
| BAJAJHFL.NS | ❌ | — | — | — | — | — | — | 0.3 |
| BAJAJHLDNG.NS | ⚠️ | TabNet | 1 Month | 0.7164 | 0.3563 | — | — | 166.6 |
| BAJFINANCE.NS | ✅ | StackEnsemble | 1 Month | 0.8070 | 0.9091 | 0.2381 | 0.3774 | 174.7 |
| BALKRISIND.NS | ⚠️ | Ensemble | 3 Months | 0.8944 | 0.5833 | — | — | 165.5 |
| BALRAMCHIN.NS | ✅ | StackEnsemble | 6 Months | 0.8151 | 0.8125 | 0.6250 | 0.7065 | 121.2 |
| BANDHANBNK.NS | ✅ | LightGBM | 2 Months | 0.6807 | 1.0000 | 0.1347 | 0.2374 | 160.1 |
| BANKBARODA.NS | ✅ | RandForest | 3 Months | 0.7981 | 0.7083 | 0.1214 | 0.2073 | 162.1 |
| BANKINDIA.NS | ✅ | CatBoost | 1 Week | 0.7053 | 0.7222 | 0.1696 | 0.2746 | 160.3 |
| BASF.NS | ⚠️ | Ensemble | 1 Week | 0.7854 | 0.5333 | — | — | 172.3 |
| BATAINDIA.NS | ✅ | Ensemble | 2 Months | 0.9127 | 0.8000 | 0.1250 | 0.2162 | 161.4 |
| BAYERCROP.NS | ✅ | XGBoost | 3 Months | 0.8447 | 1.0000 | 0.1228 | 0.2188 | 161.3 |
| BBTC.NS | ⚠️ | RandForest | 2 Weeks | 0.7680 | 0.5500 | — | — | 123.9 |
| BDL.NS | ✅ | StackEnsemble | 2 Months | 0.7892 | 0.8571 | 0.0800 | 0.1463 | 124.4 |
| BEL.NS | ✅ | CatBoost | 6 Months | 0.7872 | 0.7273 | 0.6995 | 0.7131 | 151.9 |
| BEML.NS | ✅ | StackEnsemble | 1 Month | 0.7778 | 0.8000 | 0.1412 | 0.2400 | 168.7 |
| BERGEPAINT.NS | ✅ | StackEnsemble | 3 Months | 0.8540 | 0.7660 | 0.5000 | 0.6050 | 181.3 |
| BHARATFORG.NS | ✅ | XGB_HiReg | 6 Months | 0.6233 | 0.6064 | 1.0000 | 0.7550 | 155.1 |
| BHARTIARTL.NS | ⚠️ | RandForest | 1 Month | 0.7939 | 0.4667 | — | — | 160.2 |
| BHARTIHEXA.NS | ⚠️ | StackEnsemble | 2 Weeks | 0.7464 | 0.3750 | — | — | 24.2 |
| BHEL.NS | ✅ | XGB_HiReg | 6 Months | 0.7021 | 0.9000 | 0.4592 | 0.6081 | 193.6 |
| BIKAJI.NS | ⚠️ | RandForest | 2 Months | 0.7937 | 0.5833 | — | — | 163.3 |
| BIOCON.NS | ✅ | StackEnsemble | 1 Week | 0.7429 | 1.0000 | 0.0526 | 0.1000 | 157.3 |
| BLS.NS | ⚠️ | GradBoost | 1 Week | 0.7186 | 0.3659 | — | — | 165.2 |
| BLUEDART.NS | ✅ | RandForest | 3 Months | 0.8665 | 0.7333 | 0.1183 | 0.2037 | 159.6 |
| BLUEJET.NS | ✅ | XGBoost | 1 Month | 0.6930 | 0.6667 | 0.0556 | 0.1026 | 123.4 |
| BLUESTARCO.NS | ✅ | XGB_HiReg | 6 Months | 0.8767 | 1.0000 | 0.4146 | 0.5862 | 161.1 |
| BOSCHLTD.NS | ✅ | RandForest | 3 Months | 0.6941 | 0.7500 | 0.0585 | 0.1086 | 165.9 |
| BPCL.NS | ✅ | StackEnsemble | 6 Months | 0.7190 | 0.8667 | 0.2889 | 0.4333 | 158.3 |
| BRIGADE.NS | ✅ | Ensemble | 1 Week | 0.7482 | 0.6250 | 0.0556 | 0.1020 | 150.2 |
| BRITANNIA.NS | ✅ | RandForest | 2 Months | 0.8389 | 0.7778 | 0.0625 | 0.1157 | 162.2 |
| BSE.NS | ✅ | StackEnsemble | 6 Months | 0.7226 | 0.7091 | 0.9949 | 0.8280 | 193.0 |
| BSOFT.NS | ⚠️ | XGBoost | 2 Weeks | 0.7608 | 0.5625 | — | — | 153.7 |
| CAMPUS.NS | ✅ | LightGBM | 6 Months | 0.9486 | 0.9683 | 0.6854 | 0.8026 | 145.7 |
| CAMS.NS | ⚠️ | Ensemble | 1 Week | 0.7067 | 0.5714 | — | — | 160.1 |
| CANBK.NS | ✅ | StackEnsemble | 6 Months | 0.7500 | 0.7579 | 0.5902 | 0.6636 | 163.5 |
| CANFINHOME.NS | ✅ | XGBoost | 2 Months | 0.7349 | 0.7377 | 0.2195 | 0.3383 | 163.1 |
| CAPLIPOINT.NS | ✅ | XGB_HiReg | 1 Month | 0.7836 | 0.7200 | 0.1132 | 0.1957 | 162.0 |
| CARBORUNIV.NS | ✅ | Ensemble | 6 Months | 0.9127 | 0.7237 | 0.6471 | 0.6832 | 164.6 |
| CASTROLIND.NS | ✅ | Ensemble | 3 Months | 0.9161 | 0.7500 | 0.0536 | 0.1000 | 157.9 |
| CCL.NS | ✅ | LightGBM | 6 Months | 0.7003 | 0.9191 | 0.4325 | 0.5882 | 196.4 |
| CDSL.NS | ✅ | RandForest | 3 Months | 0.7785 | 0.6667 | 0.1316 | 0.2198 | 158.2 |
| CEATLTD.NS | ✅ | XGBoost | 6 Months | 0.6610 | 1.0000 | 0.0571 | 0.1081 | 163.3 |
| CENTRALBK.NS | ⚠️ | Ensemble | 6 Months | 0.9589 | 0.3750 | — | — | 152.9 |
| CENTURYPLY.NS | ✅ | RandForest | 3 Months | 0.7873 | 0.7143 | 0.1935 | 0.3046 | 108.1 |
| CERA.NS | ✅ | XGB_HiReg | 6 Months | 0.7928 | 0.9000 | 0.0698 | 0.1295 | 156.2 |
| CESC.NS | ⚠️ | LGB_HiReg | 1 Week | 0.6938 | 0.4839 | — | — | 157.8 |
| CGCL.NS | ✅ | RandForest | 2 Weeks | 0.7550 | 0.7368 | 0.0782 | 0.1414 | 171.6 |
| CGPOWER.NS | ✅ | XGB_HiReg | 3 Months | 0.7453 | 0.7364 | 0.4222 | 0.5367 | 166.6 |
| CHALET.NS | ⚠️ | XGB_HiReg | 1 Month | 0.7573 | 0.5556 | — | — | 156.2 |
| CHAMBLFERT.NS | ✅ | XGB_HiReg | 6 Months | 0.9041 | 1.0000 | 0.2432 | 0.3913 | 155.6 |
| CHENNPETRO.NS | ⚠️ | StackEnsemble | 3 Months | 0.5994 | 0.7978 | — | — | 163.0 |
| CHOICEIN.NS | ✅ | Ensemble | 3 Months | 0.7888 | 0.7340 | 0.3833 | 0.5036 | 149.8 |
| CHOLAFIN.NS | ✅ | StackEnsemble | 1 Month | 0.7398 | 0.8333 | 0.0538 | 0.1010 | 164.7 |
| CHOLAHLDNG.NS | ✅ | Ensemble | 6 Months | 0.7930 | 0.7069 | 0.5541 | 0.6212 | 183.3 |
| CIPLA.NS | ⚠️ | RandForest | 3 Months | 0.8944 | 0.3600 | — | — | 157.7 |
| CLEAN.NS | ✅ | Ensemble | 6 Months | 0.8801 | 0.7857 | 0.1410 | 0.2391 | 159.4 |
| COALINDIA.NS | ⚠️ | GradBoost | 2 Weeks | 0.7954 | 0.4375 | — | — | 155.9 |
| COCHINSHIP.NS | ✅ | Ensemble | 1 Month | 0.7675 | 0.8462 | 0.0655 | 0.1215 | 167.1 |
| COFORGE.NS | ✅ | Ensemble | 3 Months | 0.7376 | 0.7727 | 0.0939 | 0.1675 | 206.0 |
| COHANCE.NS | ✅ | Ensemble | 2 Weeks | 0.8271 | 0.7000 | 0.0565 | 0.1045 | 140.1 |
| COLPAL.NS | ⚠️ | TabNet | 1 Week | 0.7825 | 0.3429 | — | — | 141.8 |
| CONCOR.NS | ✅ | Ensemble | 1 Week | 0.7729 | 0.6111 | 0.0675 | 0.1215 | 158.3 |
| CONCORDBIO.NS | ✅ | Ensemble | 6 Months | 0.9024 | 0.7143 | 0.0833 | 0.1493 | 140.8 |
| COROMANDEL.NS | ✅ | XGB_HiReg | 1 Month | 0.7632 | 0.6923 | 0.0539 | 0.1000 | 155.4 |
| CRAFTSMAN.NS | ⚠️ | RandForest | 3 Months | 0.5854 | 1.0000 | — | — | 172.9 |
| CREDITACC.NS | ✅ | XGB_HiReg | 6 Months | 0.6387 | 0.7000 | 0.0639 | 0.1172 | 153.7 |
| CRISIL.NS | ✅ | RandForest | 6 Months | 0.8476 | 0.7273 | 0.0851 | 0.1524 | 156.0 |
| CROMPTON.NS | ✅ | Ensemble | 6 Months | 0.9469 | 0.7500 | 0.2432 | 0.3673 | 151.9 |
| CUB.NS | ✅ | XGBoost | 1 Month | 0.6418 | 0.7222 | 0.0514 | 0.0959 | 149.8 |
| CUMMINSIND.NS | ✅ | XGBoost | 1 Month | 0.6769 | 0.8214 | 0.1790 | 0.2939 | 170.0 |
| CYIENT.NS | ✅ | TabNet | 2 Weeks | 0.7738 | 0.7059 | 0.0732 | 0.1326 | 148.3 |
| DABUR.NS | ✅ | XGB_HiReg | 2 Months | 0.8343 | 1.0000 | 0.0598 | 0.1129 | 143.5 |
| DALBHARAT.NS | ✅ | CatBoost | 3 Months | 0.7562 | 1.0000 | 0.0063 | 0.0126 | 98.7 |
| DATAPATTNS.NS | ✅ | StackEnsemble | 3 Months | 0.8292 | 0.9592 | 0.7421 | 0.8368 | 187.0 |
| DBREALTY.NS | ✅ | StackEnsemble | 6 Months | 0.9683 | 0.8000 | 0.3636 | 0.5000 | 166.4 |
| DCMSHRIRAM.NS | ✅ | StackEnsemble | 3 Months | 0.8758 | 0.7143 | 0.3061 | 0.4286 | 165.0 |
| DEEPAKFERT.NS | ✅ | StackEnsemble | 6 Months | 0.9041 | 0.7213 | 0.8000 | 0.7586 | 154.7 |
| DEEPAKNTR.NS | ✅ | RandForest | 3 Months | 0.8960 | 0.8889 | 0.1081 | 0.1928 | 168.6 |
| DELHIVERY.NS | ✅ | XGB_HiReg | 2 Months | 0.6687 | 0.7000 | 0.2692 | 0.3889 | 164.4 |
| DEVYANI.NS | ✅ | RandForest | 6 Months | 0.9384 | 0.7917 | 0.3800 | 0.5135 | 161.4 |
| DIVISLAB.NS | ✅ | RandForest | 6 Months | 0.8099 | 0.8000 | 0.1000 | 0.1778 | 169.7 |
| DIXON.NS | ✅ | Ensemble | 2 Months | 0.7470 | 0.7500 | 0.0517 | 0.0968 | 147.9 |
| DLF.NS | ⚠️ | Ensemble | 1 Month | 0.7792 | 0.5667 | — | — | 150.2 |
| DMART.NS | ✅ | StackEnsemble | 6 Months | 0.8116 | 0.7143 | 0.0862 | 0.1538 | 154.4 |
| DOMS.NS | ✅ | StackEnsemble | 2 Weeks | 0.8300 | 0.6000 | 0.0500 | 0.0923 | 144.7 |
| DRREDDY.NS | ✅ | Ensemble | 3 Months | 0.8556 | 0.7407 | 0.1887 | 0.3008 | 161.1 |
| ECLERX.NS | ✅ | RandForest | 3 Months | 0.6755 | 1.0000 | 0.0896 | 0.1644 | 175.9 |
| EICHERMOT.NS | ✅ | XGB_HiReg | 2 Months | 0.7063 | 0.7812 | 0.1174 | 0.2041 | 162.1 |
| EIDPARRY.NS | ✅ | Ensemble | 6 Months | 0.7106 | 0.7333 | 0.0625 | 0.1152 | 166.7 |
| EIHOTEL.NS | ⚠️ | RandForest | 1 Month | 0.7865 | 0.5625 | — | — | 147.3 |
| ELECON.NS | ✅ | RandForest | 6 Months | 0.7173 | 0.8571 | 0.0600 | 0.1121 | 161.9 |
| ELGIEQUIP.NS | ✅ | XGBoost | 2 Months | 0.7560 | 0.7931 | 0.2347 | 0.3622 | 167.5 |
| EMAMILTD.NS | ✅ | TabNet | 2 Months | 0.8313 | 0.7143 | 0.0847 | 0.1515 | 160.5 |
| EMCURE.NS | ✅ | StackEnsemble | 2 Months | 0.6898 | 1.0000 | 0.0550 | 0.1043 | 19.1 |
| ENDURANCE.NS | ✅ | XGBoost | 3 Months | 0.7764 | 0.6901 | 0.4949 | 0.5765 | 188.3 |
| ENGINERSIN.NS | ✅ | StackEnsemble | 3 Months | 0.8882 | 0.7521 | 0.9263 | 0.8302 | 124.4 |
| ENRIN.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ERIS.NS | ✅ | LGB_HiReg | 3 Months | 0.8603 | 0.7246 | 0.8013 | 0.7610 | 165.7 |
| ESCORTS.NS | ✅ | XGBoost | 6 Months | 0.8664 | 0.6667 | 0.0741 | 0.1333 | 162.3 |
| ETERNAL.NS | ✅ | XGB_HiReg | 2 Months | 0.7244 | 0.7627 | 0.2103 | 0.3297 | 151.5 |
| EXIDEIND.NS | ⚠️ | TabNet | 6 Months | 0.8955 | 0.4500 | — | — | 147.7 |
| FACT.NS | ✅ | StackEnsemble | 3 Months | 0.8758 | 1.0000 | 0.2157 | 0.3548 | 163.9 |
| FEDERALBNK.NS | ⚠️ | TabNet | 6 Months | 0.5839 | 0.8824 | — | — | 155.8 |
| FINCABLES.NS | ✅ | RandForest | 2 Months | 0.7380 | 0.8000 | 0.0860 | 0.1553 | 164.8 |
| FINPIPE.NS | ✅ | RandForest | 6 Months | 0.8134 | 0.6667 | 0.1043 | 0.1805 | 159.4 |
| FIRSTCRY.NS | ❌ | — | — | — | — | — | — | 0.2 |
| FIVESTAR.NS | ✅ | XGB_HiReg | 3 Months | 0.7857 | 0.7241 | 0.4421 | 0.5490 | 167.7 |
| FLUOROCHEM.NS | ✅ | RandForest | 3 Months | 0.8292 | 0.7647 | 0.2031 | 0.3210 | 202.1 |
| FORCEMOT.NS | ✅ | RandForest | 6 Months | 0.7483 | 0.7705 | 0.8704 | 0.8174 | 217.5 |
| FORTIS.NS | ✅ | XGBoost | 1 Month | 0.6637 | 1.0000 | 0.0650 | 0.1221 | 175.6 |
| FSL.NS | ✅ | XGB_HiReg | 1 Week | 0.6996 | 0.7647 | 0.0594 | 0.1102 | 158.7 |
| GAIL.NS | ✅ | RandForest | 1 Month | 0.8260 | 0.8000 | 0.1221 | 0.2119 | 154.3 |
| GESHIP.NS | ✅ | LGB_HiReg | 2 Months | 0.6506 | 0.8889 | 0.0650 | 0.1212 | 160.4 |
| GICRE.NS | ⚠️ | XGBoost | 1 Week | 0.7096 | 0.3188 | — | — | 157.7 |
| GILLETTE.NS | ✅ | Ensemble | 3 Months | 0.8416 | 0.7805 | 0.2560 | 0.3855 | 157.5 |
| GLAND.NS | ✅ | XGBoost | 3 Months | 0.6988 | 0.7500 | 0.1274 | 0.2177 | 167.3 |
| GLAXO.NS | ✅ | LightGBM | 3 Months | 0.7888 | 0.7073 | 0.3412 | 0.4603 | 204.4 |
| GLENMARK.NS | ✅ | Ensemble | 6 Months | 0.7860 | 0.7206 | 0.8000 | 0.7582 | 208.9 |
| GMDCLTD.NS | ✅ | TabNet | 6 Months | 0.7024 | 0.7722 | 0.8299 | 0.8000 | 187.5 |
| GMRAIRPORT.NS | ✅ | StackEnsemble | 1 Week | 0.7257 | 0.8333 | 0.0962 | 0.1724 | 185.0 |
| GODFRYPHLP.NS | ✅ | RandForest | 2 Months | 0.6355 | 0.9412 | 0.0623 | 0.1168 | 175.2 |
| GODIGIT.NS | ✅ | Ensemble | 1 Week | 0.7897 | 0.6429 | 0.0596 | 0.1091 | 24.4 |
| GODREJAGRO.NS | ✅ | Ensemble | 6 Months | 0.8990 | 0.8000 | 0.1765 | 0.2892 | 181.0 |
| GODREJCP.NS | ✅ | StackEnsemble | 3 Months | 0.9472 | 0.7273 | 0.7500 | 0.7385 | 173.7 |
| GODREJIND.NS | ⚠️ | TabNet | 2 Months | 0.6958 | 0.5000 | — | — | 183.4 |
| GODREJPROP.NS | ⚠️ | XGB_HiReg | 2 Months | 0.7651 | 0.5893 | — | — | 192.9 |
| GPIL.NS | ✅ | GradBoost | 6 Months | 0.7329 | 0.9204 | 0.4143 | 0.5714 | 184.9 |
| GRANULES.NS | ✅ | RandForest | 6 Months | 0.6781 | 1.0000 | 0.0553 | 0.1048 | 197.9 |
| GRAPHITE.NS | ✅ | RandForest | 6 Months | 0.6267 | 1.0000 | 0.1774 | 0.3013 | 135.0 |
| GRASIM.NS | ✅ | LGB_HiReg | 2 Weeks | 0.7968 | 0.6667 | 0.0552 | 0.1019 | 133.8 |
| GRAVITA.NS | ✅ | RandForest | 1 Week | 0.7382 | 0.6667 | 0.0635 | 0.1159 | 182.2 |
| GRSE.NS | ✅ | XGBoost | 2 Weeks | 0.6960 | 1.0000 | 0.0746 | 0.1388 | 185.0 |
| GSPL.NS | ✅ | XGB_HiReg | 2 Months | 0.7956 | 0.6129 | 0.2657 | 0.3707 | 177.3 |
| GUJGASLTD.NS | ✅ | GradBoost | 2 Weeks | 0.7695 | 0.6957 | 0.0947 | 0.1667 | 163.1 |
| GVT&D.NS | ✅ | StackEnsemble | 6 Months | 0.7021 | 1.0000 | 0.5000 | 0.6667 | 188.1 |
| HAL.NS | ✅ | StackEnsemble | 6 Months | 0.9589 | 1.0000 | 0.2500 | 0.4000 | 164.1 |
| HAPPSTMNDS.NS | ✅ | LGB_HiReg | 3 Months | 0.9565 | 0.8000 | 0.2353 | 0.3636 | 184.1 |
| HAVELLS.NS | ✅ | RandForest | 2 Months | 0.8750 | 0.6667 | 0.0920 | 0.1616 | 167.0 |
| HBLENGINE.NS | ✅ | XGB_HiReg | 3 Months | 0.6475 | 1.0000 | 0.2199 | 0.3606 | 182.9 |
| HCLTECH.NS | ⚠️ | RandForest | 6 Months | 0.7432 | 0.4444 | — | — | 206.7 |
| HDFCAMC.NS | ✅ | StackEnsemble | 1 Month | 0.7982 | 0.8000 | 0.0556 | 0.1039 | 184.6 |
| HDFCBANK.NS | ✅ | LGB_HiReg | 6 Months | 0.8288 | 0.7500 | 0.1111 | 0.1935 | 169.2 |
| HDFCLIFE.NS | ✅ | Ensemble | 6 Months | 0.8973 | 0.7053 | 0.9710 | 0.8171 | 181.0 |
| HEG.NS | ✅ | RandForest | 2 Weeks | 0.6931 | 1.0000 | 0.0575 | 0.1088 | 205.3 |
| HEROMOTOCO.NS | ✅ | XGBoost | 2 Months | 0.7892 | 0.7262 | 0.3427 | 0.4656 | 172.5 |
| HEXT.NS | ❌ | — | — | — | — | — | — | 0.2 |
| HFCL.NS | ✅ | LGB_HiReg | 2 Months | 0.7892 | 0.7302 | 0.2722 | 0.3966 | 186.6 |
| HINDALCO.NS | ✅ | XGB_HiReg | 6 Months | 0.7560 | 0.9453 | 0.7280 | 0.8225 | 181.3 |
| HINDCOPPER.NS | ✅ | StackEnsemble | 6 Months | 0.7021 | 0.7422 | 0.8520 | 0.7933 | 179.0 |
| HINDPETRO.NS | ✅ | Ensemble | 6 Months | 0.6832 | 1.0000 | 0.0561 | 0.1063 | 184.5 |
| HINDUNILVR.NS | ✅ | Ensemble | 3 Months | 0.8727 | 0.7333 | 0.1236 | 0.2115 | 164.9 |
| HINDZINC.NS | ✅ | RandForest | 6 Months | 0.6267 | 0.9286 | 0.0565 | 0.1066 | 179.2 |
| HOMEFIRST.NS | ⚠️ | TabNet | 2 Months | 0.7063 | 0.5676 | — | — | 141.4 |
| HONASA.NS | ✅ | StackEnsemble | 3 Months | 0.7578 | 0.8235 | 0.1573 | 0.2642 | 185.3 |
| HONAUT.NS | ✅ | XGB_HiReg | 1 Month | 0.8423 | 0.8000 | 0.1333 | 0.2286 | 179.9 |
| HSCL.NS | ✅ | StackEnsemble | 1 Month | 0.7164 | 0.8500 | 0.1532 | 0.2595 | 183.2 |
| HUDCO.NS | ✅ | StackEnsemble | 3 Months | 0.8323 | 0.8333 | 0.1613 | 0.2703 | 180.7 |
| HYUNDAI.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ICICIBANK.NS | ✅ | Ensemble | 6 Months | 0.8579 | 0.7143 | 0.0581 | 0.1075 | 178.4 |
| ICICIGI.NS | ✅ | RandForest | 2 Months | 0.8389 | 0.8571 | 0.0536 | 0.1008 | 177.8 |
| ICICIPRULI.NS | ⚠️ | Ensemble | 2 Months | 0.8298 | 0.5000 | — | — | 186.2 |
| IDBI.NS | ✅ | XGBoost | 2 Months | 0.6687 | 1.0000 | 0.0756 | 0.1406 | 187.9 |
| IDEA.NS | ✅ | RandForest | 6 Months | 0.7996 | 0.7067 | 0.9422 | 0.8076 | 198.9 |
| IDFCFIRSTB.NS | ⚠️ | TabNet | 1 Week | 0.7382 | 0.5667 | — | — | 192.4 |
| IEX.NS | ✅ | StackEnsemble | 2 Weeks | 0.8520 | 0.9286 | 0.2889 | 0.4407 | 179.7 |
| IFCI.NS | ✅ | XGBoost | 2 Months | 0.6913 | 0.6833 | 0.1806 | 0.2857 | 195.6 |
| IGIL.NS | ❌ | — | — | — | — | — | — | 0.3 |
| IGL.NS | ✅ | Ensemble | 1 Week | 0.7454 | 0.7200 | 0.0952 | 0.1682 | 199.6 |
| IIFL.NS | ✅ | RandForest | 6 Months | 0.6182 | 1.0000 | 0.0785 | 0.1456 | 222.4 |
| IKS.NS | ❌ | — | — | — | — | — | — | 0.4 |
| INDGN.NS | ✅ | Ensemble | 2 Months | 0.8313 | 0.7917 | 0.1508 | 0.2533 | 26.5 |
| INDHOTEL.NS | ⚠️ | Ensemble | 1 Week | 0.7568 | 0.5909 | — | — | 192.7 |
| INDIACEM.NS | ✅ | StackEnsemble | 2 Months | 0.7289 | 0.8571 | 0.0632 | 0.1176 | 209.7 |
| INDIAMART.NS | ✅ | StackEnsemble | 2 Weeks | 0.8473 | 0.8571 | 0.1034 | 0.1846 | 191.0 |
| INDIANB.NS | ✅ | StackEnsemble | 6 Months | 0.7021 | 0.7431 | 0.5786 | 0.6506 | 198.3 |
| INDIGO.NS | ⚠️ | StackEnsemble | 2 Weeks | 0.7867 | 0.5000 | — | — | 134.9 |
| INDUSINDBK.NS | ✅ | LGB_HiReg | 3 Months | 0.6863 | 0.9231 | 0.0563 | 0.1062 | 189.9 |
| INDUSTOWER.NS | ✅ | LightGBM | 2 Weeks | 0.7161 | 0.7333 | 0.0539 | 0.1005 | 199.4 |
| INFY.NS | ⚠️ | CatBoost | 1 Week | 0.7597 | 0.3617 | — | — | 194.5 |
| INOXINDIA.NS | ✅ | RandForest | 1 Month | 0.7368 | 0.7857 | 0.0585 | 0.1089 | 150.7 |
| INOXWIND.NS | ✅ | LGB_HiReg | 2 Weeks | 0.7464 | 0.8333 | 0.0543 | 0.1020 | 189.2 |
| INTELLECT.NS | ✅ | XGBoost | 6 Months | 0.7055 | 1.0000 | 0.0058 | 0.0115 | 192.5 |
| IOB.NS | ⚠️ | TabNet | 1 Week | 0.7740 | 0.5882 | — | — | 195.0 |
| IOC.NS | ✅ | TabNet | 1 Week | 0.7511 | 0.7000 | 0.0769 | 0.1386 | 191.7 |
| IPCALAB.NS | ✅ | Ensemble | 1 Week | 0.7382 | 0.6667 | 0.0532 | 0.0985 | 209.9 |
| IRB.NS | ⚠️ | GradBoost | 1 Month | 0.7822 | 0.5238 | — | — | 184.8 |
| IRCON.NS | ⚠️ | XGB_HiReg | 6 Months | 0.8065 | 0.5185 | — | — | 154.4 |
| IRCTC.NS | ⚠️ | LGB_HiReg | 2 Weeks | 0.8545 | 0.5000 | — | — | 151.2 |
| IREDA.NS | ✅ | XGBoost | 6 Months | 0.9658 | 0.7222 | 0.4643 | 0.5652 | 144.6 |
| IRFC.NS | ✅ | GradBoost | 2 Weeks | 0.8055 | 0.7391 | 0.1164 | 0.2012 | 157.5 |
| ITC.NS | ✅ | Ensemble | 2 Weeks | 0.9164 | 0.7500 | 0.0500 | 0.0938 | 114.3 |
| ITCHOTELS.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ITI.NS | ✅ | StackEnsemble | 3 Months | 0.8261 | 0.8261 | 0.2676 | 0.4043 | 154.2 |
| J&KBANK.NS | ✅ | StackEnsemble | 2 Weeks | 0.7118 | 0.8333 | 0.0926 | 0.1667 | 158.2 |
| JBCHEPHARM.NS | ✅ | RandForest | 6 Months | 0.7791 | 0.7778 | 0.1007 | 0.1783 | 153.9 |
| JBMA.NS | ✅ | XGB_HiReg | 6 Months | 0.8425 | 0.8065 | 0.2252 | 0.3521 | 108.0 |
| JINDALSAW.NS | ✅ | XGBoost | 6 Months | 0.8048 | 0.8049 | 0.2374 | 0.3667 | 151.0 |
| JINDALSTEL.NS | ✅ | StackEnsemble | 2 Months | 0.7199 | 0.7647 | 0.2342 | 0.3586 | 145.1 |
| JIOFIN.NS | ✅ | XGBoost | 6 Months | 0.7791 | 1.0000 | 0.1731 | 0.2951 | 129.5 |
| JKCEMENT.NS | ✅ | StackEnsemble | 3 Months | 0.9410 | 0.7500 | 0.1429 | 0.2400 | 188.7 |
| JKTYRE.NS | ✅ | XGB_HiReg | 6 Months | 0.7329 | 0.7152 | 0.7740 | 0.7434 | 148.1 |
| JMFINANCIL.NS | ✅ | XGB_HiReg | 6 Months | 0.7021 | 1.0000 | 0.1031 | 0.1869 | 115.9 |
| JPPOWER.NS | ✅ | RandForest | 2 Months | 0.7214 | 0.7442 | 0.1553 | 0.2570 | 156.8 |
| JSL.NS | ✅ | Ensemble | 6 Months | 0.7106 | 0.7429 | 0.1398 | 0.2353 | 166.2 |
| JSWCEMENT.NS | ❌ | — | — | — | — | — | — | 0.2 |
| JSWENERGY.NS | ✅ | RandForest | 3 Months | 0.7717 | 0.7059 | 0.0779 | 0.1404 | 164.7 |
| JSWINFRA.NS | ✅ | Ensemble | 2 Weeks | 0.7565 | 0.7059 | 0.0682 | 0.1244 | 126.1 |
| JSWSTEEL.NS | ✅ | StackEnsemble | 1 Month | 0.8304 | 0.7083 | 0.4359 | 0.5397 | 163.6 |
| JUBLFOOD.NS | ✅ | RandForest | 2 Weeks | 0.8300 | 0.7500 | 0.0952 | 0.1690 | 153.8 |
| JUBLINGREA.NS | ✅ | XGB_HiReg | 1 Week | 0.6981 | 0.6875 | 0.0507 | 0.0944 | 154.3 |
| JUBLPHARMA.NS | ✅ | StackEnsemble | 2 Months | 0.7560 | 0.7500 | 0.1348 | 0.2286 | 156.6 |
| JWL.NS | ✅ | Ensemble | 1 Week | 0.7482 | 0.7083 | 0.0914 | 0.1619 | 149.1 |
| JYOTHYLAB.NS | ⚠️ | XGBoost | 1 Week | 0.7725 | 0.4211 | — | — | 153.1 |
| JYOTICNC.NS | ✅ | TabNet | 2 Weeks | 0.7147 | 0.6000 | 0.0739 | 0.1316 | 134.8 |
| KAJARIACER.NS | ✅ | StackEnsemble | 2 Months | 0.9066 | 0.7031 | 0.7895 | 0.7438 | 158.6 |
| KALYANKJIL.NS | ✅ | StackEnsemble | 6 Months | 0.9452 | 0.7500 | 0.1667 | 0.2727 | 141.0 |
| KARURVYSYA.NS | ✅ | LGB_HiReg | 3 Months | 0.6227 | 1.0000 | 0.0581 | 0.1099 | 159.3 |
| KAYNES.NS | ✅ | XGBoost | 6 Months | 0.7586 | 0.6373 | 0.8667 | 0.7345 | 171.8 |
| KEC.NS | ✅ | RandForest | 3 Months | 0.8665 | 0.8000 | 0.0870 | 0.1569 | 156.7 |
| KEI.NS | ✅ | StackEnsemble | 2 Weeks | 0.6513 | 0.9000 | 0.0698 | 0.1295 | 165.7 |
| KFINTECH.NS | ✅ | Ensemble | 2 Weeks | 0.7695 | 0.7500 | 0.0542 | 0.1011 | 179.9 |
| KIMS.NS | ✅ | Ensemble | 6 Months | 0.7175 | 0.7500 | 0.1475 | 0.2466 | 179.4 |
| KIRLOSBROS.NS | ⚠️ | XGB_HiReg | 1 Month | 0.7646 | 0.5600 | — | — | 170.5 |
| KIRLOSENG.NS | ✅ | StackEnsemble | 6 Months | 0.9144 | 0.9144 | 1.0000 | 0.9553 | 170.2 |
| KOTAKBANK.NS | ⚠️ | XGBoost | 3 Months | 0.8199 | 0.4444 | — | — | 173.0 |
| KPIL.NS | ✅ | StackEnsemble | 3 Months | 0.8634 | 0.7045 | 0.5000 | 0.5849 | 163.4 |
| KPITTECH.NS | ⚠️ | XGBoost | 2 Weeks | 0.6830 | 0.3991 | — | — | 164.1 |
| KPRMILL.NS | ✅ | LGB_HiReg | 2 Weeks | 0.6951 | 1.0000 | 0.0685 | 0.1282 | 165.4 |
| KSB.NS | ✅ | StackEnsemble | 3 Months | 0.7733 | 0.6512 | 0.3256 | 0.4341 | 175.2 |
| LALPATHLAB.NS | ✅ | StackEnsemble | 2 Months | 0.9157 | 0.8043 | 0.6607 | 0.7255 | 167.9 |
| LATENTVIEW.NS | ✅ | XGB_HiReg | 1 Month | 0.7354 | 0.6500 | 0.1347 | 0.2232 | 200.3 |
| LAURUSLABS.NS | ✅ | GradBoost | 6 Months | 0.7757 | 0.7718 | 0.8958 | 0.8292 | 176.6 |
| LEMONTREE.NS | ✅ | Ensemble | 6 Months | 0.7962 | 0.7692 | 0.0794 | 0.1439 | 190.0 |
| LICHSGFIN.NS | ⚠️ | CatBoost | 2 Weeks | 0.8055 | 0.5000 | — | — | 178.1 |
| LICI.NS | ✅ | XGB_HiReg | 6 Months | 0.8887 | 0.9778 | 0.4074 | 0.5752 | 177.6 |
| LINDEINDIA.NS | ✅ | RandForest | 3 Months | 0.7981 | 0.7123 | 0.3230 | 0.4444 | 166.4 |
| LLOYDSME.NS | ✅ | StackEnsemble | 6 Months | 0.7740 | 0.7073 | 0.3494 | 0.4677 | 153.8 |
| LODHA.NS | ⚠️ | CatBoost | 1 Month | 0.7222 | 0.5472 | — | — | 168.2 |
| LT.NS | ✅ | XGB_HiReg | 6 Months | 0.8168 | 0.7089 | 0.4000 | 0.5114 | 187.5 |
| LTF.NS | ✅ | Ensemble | 2 Weeks | 0.6830 | 0.6667 | 0.0531 | 0.0984 | 205.9 |
| LTFOODS.NS | ✅ | StackEnsemble | 2 Weeks | 0.7666 | 0.7143 | 0.0595 | 0.1099 | 185.0 |
| LTIM.NS | ✅ | XGB_HiReg | 2 Months | 0.7756 | 0.8421 | 0.0988 | 0.1768 | 209.0 |
| LTTS.NS | ✅ | CatBoost | 2 Weeks | 0.7709 | 0.7143 | 0.0893 | 0.1587 | 186.7 |
| LUPIN.NS | ⚠️ | LGB_HiReg | 1 Week | 0.7668 | 0.3750 | — | — | 189.8 |
| M&M.NS | ✅ | StackEnsemble | 1 Month | 0.8187 | 0.7692 | 0.1449 | 0.2439 | 184.5 |
| M&MFIN.NS | ✅ | RandForest | 1 Month | 0.6871 | 0.6875 | 0.0500 | 0.0932 | 179.0 |
| MAHABANK.NS | ✅ | XGB_HiReg | 6 Months | 0.6233 | 0.9412 | 0.1280 | 0.2254 | 185.3 |
| MAHSCOOTER.NS | ✅ | StackEnsemble | 1 Week | 0.8362 | 0.8889 | 0.1250 | 0.2192 | 205.8 |
| MAHSEAMLES.NS | ✅ | XGBoost | 2 Months | 0.8012 | 0.8571 | 0.0845 | 0.1538 | 158.1 |
| MANAPPURAM.NS | ✅ | XGBoost | 2 Months | 0.6446 | 1.0000 | 0.0560 | 0.1061 | 191.9 |
| MANKIND.NS | ⚠️ | XGBoost | 2 Weeks | 0.7781 | 0.5833 | — | — | 174.8 |
| MANYAVAR.NS | ✅ | RandForest | 1 Month | 0.8841 | 0.6000 | 0.2000 | 0.3000 | 196.7 |
| MAPMYINDIA.NS | ⚠️ | Ensemble | 2 Weeks | 0.8300 | 0.5385 | — | — | 184.3 |
| MARICO.NS | ✅ | XGB_HiReg | 2 Months | 0.8449 | 0.9048 | 0.2774 | 0.4246 | 186.6 |
| MARUTI.NS | ✅ | RandForest | 6 Months | 0.8647 | 0.7185 | 0.7029 | 0.7106 | 204.4 |
| MAXHEALTH.NS | ⚠️ | LGB_HiReg | 1 Week | 0.7239 | 0.5500 | — | — | 185.0 |
| MAZDOCK.NS | ✅ | XGB_HiReg | 6 Months | 0.7449 | 0.7857 | 0.1333 | 0.2280 | 190.0 |
| MCX.NS | ✅ | StackEnsemble | 6 Months | 0.9760 | 0.9760 | 1.0000 | 0.9879 | 218.9 |
| MEDANTA.NS | ⚠️ | LightGBM | 3 Months | 0.7345 | 0.5270 | — | — | 176.1 |
| METROPOLIS.NS | ✅ | Ensemble | 6 Months | 0.7688 | 0.6923 | 0.1241 | 0.2105 | 205.5 |
| MFSL.NS | ✅ | XGB_HiReg | 6 Months | 0.8031 | 0.7072 | 0.6737 | 0.6900 | 183.9 |
| MGL.NS | ✅ | RandForest | 2 Months | 0.8087 | 0.7143 | 0.1103 | 0.1911 | 183.1 |
| MINDACORP.NS | ✅ | XGB_HiReg | 6 Months | 0.8836 | 0.6937 | 0.6937 | 0.6937 | 178.1 |
| MMTC.NS | ✅ | Ensemble | 2 Months | 0.7093 | 0.7000 | 0.0697 | 0.1267 | 185.3 |
| MOTHERSON.NS | ✅ | XGBoost | 2 Weeks | 0.7262 | 0.7010 | 0.2969 | 0.4172 | 194.4 |
| MOTILALOFS.NS | ✅ | XGBoost | 6 Months | 0.6455 | 0.8750 | 0.0639 | 0.1191 | 181.8 |
| MPHASIS.NS | ✅ | RandForest | 2 Months | 0.7666 | 0.7234 | 0.1932 | 0.3049 | 170.4 |
| MRF.NS | ✅ | LGB_HiReg | 6 Months | 0.7637 | 1.0000 | 0.1266 | 0.2247 | 176.0 |
| MRPL.NS | ✅ | TabNet | 3 Months | 0.6599 | 0.6400 | 0.0708 | 0.1275 | 168.6 |
| MSUMI.NS | ✅ | XGBoost | 6 Months | 0.7021 | 1.0000 | 0.0938 | 0.1714 | 168.0 |
| MUTHOOTFIN.NS | ✅ | StackEnsemble | 2 Months | 0.6506 | 0.7000 | 0.0583 | 0.1077 | 174.6 |
| NAM-INDIA.NS | ✅ | CatBoost | 6 Months | 0.7754 | 0.7461 | 0.9502 | 0.8359 | 189.7 |
| NATCOPHARM.NS | ✅ | RandForest | 3 Months | 0.6444 | 0.9231 | 0.0956 | 0.1733 | 192.5 |
| NATIONALUM.NS | ✅ | StackEnsemble | 6 Months | 0.9007 | 0.9580 | 0.9331 | 0.9454 | 176.8 |
| NAUKRI.NS | ⚠️ | RandForest | 1 Month | 0.8099 | 0.5263 | — | — | 159.3 |
| NAVA.NS | ✅ | XGB_HiReg | 2 Months | 0.7108 | 0.7754 | 0.3993 | 0.5271 | 185.4 |
| NAVINFLUOR.NS | ✅ | LGB_HiReg | 2 Weeks | 0.6556 | 1.0000 | 0.0083 | 0.0165 | 154.9 |
| NBCC.NS | ✅ | StackEnsemble | 6 Months | 0.8801 | 0.7143 | 0.1316 | 0.2222 | 160.8 |
| NCC.NS | ⚠️ | RandForest | 2 Months | 0.7892 | 0.5556 | — | — | 125.0 |
| NESTLEIND.NS | ✅ | LGB_HiReg | 2 Weeks | 0.8112 | 0.7778 | 0.0515 | 0.0966 | 155.3 |
| NETWEB.NS | ✅ | LightGBM | 6 Months | 0.7911 | 0.7726 | 0.9159 | 0.8382 | 210.1 |
| NEULANDLAB.NS | ✅ | StackEnsemble | 1 Month | 0.7602 | 0.7273 | 0.2474 | 0.3692 | 167.8 |
| NEWGEN.NS | ⚠️ | Ensemble | 1 Month | 0.7851 | 0.4706 | — | — | 158.5 |
| NH.NS | ✅ | GradBoost | 2 Months | 0.6446 | 0.6842 | 0.0535 | 0.0992 | 178.8 |
| NHPC.NS | ✅ | RandForest | 2 Weeks | 0.7534 | 0.6875 | 0.0948 | 0.1667 | 157.4 |
| NIACL.NS | ⚠️ | GradBoost | 1 Month | 0.8046 | 0.5938 | — | — | 168.1 |
| NIVABUPA.NS | ❌ | — | — | — | — | — | — | 0.2 |
| NLCINDIA.NS | ✅ | Ensemble | 6 Months | 0.7346 | 0.7857 | 0.0675 | 0.1243 | 170.4 |
| NMDC.NS | ✅ | StackEnsemble | 2 Months | 0.7982 | 0.7209 | 0.3605 | 0.4806 | 181.2 |
| NSLNISP.NS | ✅ | Ensemble | 2 Weeks | 0.7118 | 0.7500 | 0.0577 | 0.1071 | 153.7 |
| NTPC.NS | ⚠️ | Ensemble | 1 Week | 0.7969 | 0.5455 | — | — | 140.9 |
| NTPCGREEN.NS | ❌ | — | — | — | — | — | — | 0.2 |
| NUVAMA.NS | ✅ | LGB_HiReg | 6 Months | 0.8425 | 0.7216 | 0.9249 | 0.8107 | 147.5 |
| NUVOCO.NS | ✅ | StackEnsemble | 2 Months | 0.8705 | 0.7500 | 0.0667 | 0.1224 | 162.2 |
| NYKAA.NS | ✅ | LightGBM | 6 Months | 0.7021 | 0.8315 | 0.5068 | 0.6298 | 152.8 |
| OBEROIRLTY.NS | ✅ | Ensemble | 2 Months | 0.8178 | 0.7536 | 0.3333 | 0.4622 | 152.9 |
| OFSS.NS | ✅ | Ensemble | 3 Months | 0.7438 | 0.9000 | 0.0520 | 0.0984 | 164.5 |
| OIL.NS | ⚠️ | CatBoost | 1 Week | 0.7139 | 0.5195 | — | — | 152.6 |
| OLAELEC.NS | ❌ | — | — | — | — | — | — | 0.2 |
| OLECTRA.NS | ✅ | Ensemble | 3 Months | 0.7811 | 0.7015 | 0.4821 | 0.5714 | 193.6 |
| ONESOURCE.NS | ❌ | — | — | — | — | — | — | 0.2 |
| ONGC.NS | ⚠️ | TabNet | 3 Months | 0.7888 | 0.4074 | — | — | 153.4 |
| PAGEIND.NS | ✅ | Ensemble | 3 Months | 0.8494 | 0.7647 | 0.3145 | 0.4457 | 154.4 |
| PATANJALI.NS | ⚠️ | XGB_HiReg | 2 Weeks | 0.8660 | 0.5238 | — | — | 156.1 |
| PAYTM.NS | ✅ | XGB_HiReg | 1 Month | 0.6857 | 1.0000 | 0.0773 | 0.1434 | 160.7 |
| PCBL.NS | ✅ | StackEnsemble | 1 Week | 0.7686 | 0.6000 | 0.1071 | 0.1818 | 159.9 |
| PERSISTENT.NS | ⚠️ | Ensemble | 2 Months | 0.7274 | 0.5946 | — | — | 163.5 |
| PETRONET.NS | ✅ | RandForest | 3 Months | 0.8960 | 0.7500 | 0.3176 | 0.4463 | 157.8 |
| PFC.NS | ✅ | RandForest | 6 Months | 0.8168 | 0.7143 | 0.0885 | 0.1575 | 157.9 |
| PFIZER.NS | ✅ | XGBoost | 2 Weeks | 0.8343 | 0.6667 | 0.0508 | 0.0945 | 150.4 |
| PGEL.NS | ⚠️ | RandForest | 1 Week | 0.6838 | 0.5909 | — | — | 164.4 |
| PGHH.NS | ⚠️ | CatBoost | 2 Weeks | 0.8818 | 0.5000 | — | — | 154.7 |
| PHOENIXLTD.NS | ⚠️ | Ensemble | 2 Weeks | 0.7277 | 0.5882 | — | — | 168.6 |
| PIDILITIND.NS | ✅ | StackEnsemble | 2 Months | 0.8705 | 0.8000 | 0.0870 | 0.1569 | 160.8 |
| PIIND.NS | ✅ | RandForest | 1 Month | 0.8260 | 0.7273 | 0.0645 | 0.1185 | 170.5 |
| PNB.NS | ✅ | XGBoost | 6 Months | 0.7620 | 0.6667 | 0.0694 | 0.1258 | 161.5 |
| PNBHOUSING.NS | ✅ | StackEnsemble | 6 Months | 0.8356 | 0.9238 | 0.7080 | 0.8017 | 185.2 |
| POLICYBZR.NS | ✅ | StackEnsemble | 6 Months | 0.8973 | 0.8000 | 0.1212 | 0.2105 | 168.4 |
| POLYCAB.NS | ✅ | XGB_HiReg | 3 Months | 0.7453 | 0.7200 | 0.4696 | 0.5684 | 170.0 |
| POLYMED.NS | ✅ | RandForest | 3 Months | 0.8214 | 0.6364 | 0.0593 | 0.1085 | 165.4 |
| POONAWALLA.NS | ✅ | XGBoost | 2 Months | 0.6687 | 1.0000 | 0.0984 | 0.1791 | 209.0 |
| POWERGRID.NS | ⚠️ | LGB_HiReg | 1 Month | 0.8143 | 0.4444 | — | — | 166.0 |
| POWERINDIA.NS | ✅ | StackEnsemble | 3 Months | 0.7702 | 0.8317 | 0.8160 | 0.8238 | 186.8 |
| PPLPHARMA.NS | ✅ | RandForest | 1 Month | 0.7968 | 0.7273 | 0.2013 | 0.3153 | 155.6 |
| PRAJIND.NS | ✅ | XGBoost | 2 Weeks | 0.7867 | 0.7273 | 0.0523 | 0.0976 | 176.8 |
| PREMIERENE.NS | ❌ | — | — | — | — | — | — | 0.2 |
| PRESTIGE.NS | ✅ | Ensemble | 2 Months | 0.7470 | 0.7750 | 0.1632 | 0.2696 | 126.3 |
| PTCIL.NS | ✅ | StackEnsemble | 6 Months | 0.8322 | 0.7333 | 0.7253 | 0.7293 | 153.5 |
| PVRINOX.NS | ✅ | CatBoost | 1 Month | 0.7719 | 0.6471 | 0.0683 | 0.1236 | 160.1 |
| RADICO.NS | ✅ | StackEnsemble | 2 Weeks | 0.7320 | 0.8333 | 0.0515 | 0.0971 | 194.3 |
| RAILTEL.NS | ✅ | RandForest | 1 Week | 0.7425 | 0.8421 | 0.0829 | 0.1509 | 171.7 |
| RAINBOW.NS | ✅ | RandForest | 2 Months | 0.7696 | 0.7059 | 0.0750 | 0.1356 | 166.1 |
| RAMCOCEM.NS | ✅ | RandForest | 6 Months | 0.7321 | 0.7778 | 0.0500 | 0.0940 | 175.3 |
| RBLBANK.NS | ✅ | LGB_HiReg | 2 Weeks | 0.6700 | 0.7895 | 0.0625 | 0.1158 | 169.9 |
| RCF.NS | ✅ | XGBoost | 1 Month | 0.8202 | 0.7083 | 0.2378 | 0.3560 | 158.9 |
| RECLTD.NS | ⚠️ | Ensemble | 3 Months | 0.9037 | 0.5000 | — | — | 148.3 |
| REDINGTON.NS | ✅ | StackEnsemble | 1 Month | 0.7895 | 0.7273 | 0.1039 | 0.1818 | 165.0 |
| RELIANCE.NS | ✅ | StackEnsemble | 3 Months | 0.9503 | 0.7143 | 1.0000 | 0.8333 | 157.9 |
| RHIM.NS | ✅ | XGB_HiReg | 6 Months | 0.8990 | 0.8000 | 0.0645 | 0.1194 | 173.1 |
| RITES.NS | ✅ | XGBoost | 2 Months | 0.7846 | 0.6667 | 0.0933 | 0.1637 | 167.7 |
| RKFORGE.NS | ✅ | XGBoost | 1 Month | 0.7719 | 0.6667 | 0.0500 | 0.0930 | 154.4 |
| RPOWER.NS | ✅ | StackEnsemble | 3 Months | 0.9068 | 0.7755 | 0.6667 | 0.7170 | 164.0 |
| RRKABEL.NS | ✅ | Ensemble | 1 Month | 0.7617 | 0.7048 | 0.5065 | 0.5894 | 181.1 |
| RVNL.NS | ⚠️ | XGB_HiReg | 1 Week | 0.7568 | 0.5217 | — | — | 162.3 |
| SAGILITY.NS | ❌ | — | — | — | — | — | — | 0.2 |
| SAIL.NS | ✅ | StackEnsemble | 6 Months | 0.6206 | 0.9368 | 0.4972 | 0.6496 | 162.2 |
| SAILIFE.NS | ❌ | — | — | — | — | — | — | 0.2 |
| SAMMAANCAP.NS | ✅ | StackEnsemble | 3 Months | 0.7516 | 0.7692 | 0.1149 | 0.2000 | 163.8 |
| SAPPHIRE.NS | ✅ | RandForest | 6 Months | 0.9675 | 0.7059 | 0.4615 | 0.5581 | 167.4 |
| SARDAEN.NS | ✅ | RandForest | 6 Months | 0.8379 | 0.7037 | 0.9448 | 0.8066 | 198.7 |
| SAREGAMA.NS | ✅ | RandForest | 3 Months | 0.8675 | 0.8333 | 0.0633 | 0.1176 | 162.2 |
| SBFC.NS | ✅ | StackEnsemble | 1 Month | 0.8275 | 0.9091 | 0.1471 | 0.2532 | 172.4 |
| SBICARD.NS | ✅ | RandForest | 3 Months | 0.7811 | 0.7895 | 0.0987 | 0.1754 | 158.7 |
| SBILIFE.NS | ✅ | StackEnsemble | 3 Months | 0.8727 | 0.7037 | 0.3654 | 0.4810 | 163.4 |
| SBIN.NS | ✅ | StackEnsemble | 6 Months | 0.8617 | 0.8272 | 0.9504 | 0.8845 | 160.2 |
| SCHAEFFLER.NS | ✅ | StackEnsemble | 3 Months | 0.9046 | 0.7241 | 0.5250 | 0.6087 | 162.0 |
| SCHNEIDER.NS | ✅ | LGB_HiReg | 2 Months | 0.7355 | 0.7101 | 0.6627 | 0.6856 | 160.1 |
| SCI.NS | ✅ | StackEnsemble | 6 Months | 0.7391 | 0.7160 | 0.5743 | 0.6374 | 171.1 |
| SHREECEM.NS | ⚠️ | Ensemble | 1 Week | 0.7907 | 0.5000 | — | — | 162.6 |
| SHRIRAMFIN.NS | ✅ | StackEnsemble | 6 Months | 0.7075 | 0.9875 | 0.5197 | 0.6810 | 164.5 |
| SHYAMMETL.NS | ✅ | StackEnsemble | 3 Months | 0.7774 | 0.7143 | 0.0758 | 0.1370 | 162.2 |
| SIEMENS.NS | ✅ | Ensemble | 3 Months | 0.7032 | 0.6316 | 0.0694 | 0.1250 | 161.9 |
| SIGNATURE.NS | ✅ | Ensemble | 2 Months | 0.7850 | 0.6500 | 0.0985 | 0.1711 | 159.9 |
| SJVN.NS | ⚠️ | TabNet | 2 Weeks | 0.7911 | 0.5263 | — | — | 151.6 |
| SOBHA.NS | ✅ | CatBoost | 1 Week | 0.7037 | 0.7857 | 0.0573 | 0.1068 | 166.4 |
| SOLARINDS.NS | ✅ | LGB_HiReg | 2 Weeks | 0.6526 | 0.8235 | 0.0622 | 0.1157 | 168.6 |
| SONACOMS.NS | ✅ | XGBoost | 6 Months | 0.7549 | 0.8214 | 0.1620 | 0.2706 | 179.3 |
| SONATSOFTW.NS | ✅ | StackEnsemble | 2 Months | 0.8225 | 1.0000 | 0.0877 | 0.1613 | 158.1 |
| SRF.NS | ✅ | XGB_HiReg | 6 Months | 0.9127 | 1.0000 | 0.0556 | 0.1053 | 171.9 |
| STARHEALTH.NS | ✅ | XGB_HiReg | 2 Months | 0.7526 | 0.9000 | 0.1118 | 0.1989 | 165.1 |
| SUMICHEM.NS | ✅ | StackEnsemble | 6 Months | 0.9012 | 0.7500 | 0.5385 | 0.6269 | 159.1 |
| SUNDARMFIN.NS | ✅ | Ensemble | 6 Months | 0.7806 | 0.8750 | 0.0598 | 0.1120 | 175.4 |
| SUNDRMFAST.NS | ⚠️ | Ensemble | 3 Months | 0.8781 | 0.5714 | — | — | 171.6 |
| SUNPHARMA.NS | ⚠️ | CatBoost | 2 Weeks | 0.8393 | 0.3810 | — | — | 162.4 |
| SUNTV.NS | ⚠️ | GradBoost | 1 Week | 0.7778 | 0.4375 | — | — | 160.6 |
| SUPREMEIND.NS | ✅ | XGBoost | 1 Month | 0.8102 | 0.7000 | 0.3427 | 0.4601 | 165.2 |
| SUZLON.NS | ✅ | RandForest | 3 Months | 0.7703 | 0.7692 | 0.0730 | 0.1333 | 154.2 |
| SWANCORP.NS | ✅ | GradBoost | 2 Weeks | 0.7597 | 0.6000 | 0.0789 | 0.1395 | 152.5 |
| SWIGGY.NS | ❌ | — | — | — | — | — | — | 0.2 |
| SYNGENE.NS | ✅ | XGB_HiReg | 1 Week | 0.7617 | 0.8000 | 0.0764 | 0.1395 | 160.5 |
| SYRMA.NS | ✅ | XGB_HiReg | 6 Months | 0.8600 | 0.8641 | 0.9020 | 0.8826 | 158.3 |
| TARIL.NS | ✅ | RandForest | 2 Months | 0.7894 | 0.7794 | 0.3292 | 0.4629 | 112.5 |
| TATACHEM.NS | ✅ | GradBoost | 3 Months | 0.8457 | 0.7143 | 0.2451 | 0.3650 | 166.6 |
| TATACOMM.NS | ✅ | XGB_HiReg | 2 Months | 0.7209 | 0.7727 | 0.1818 | 0.2944 | 157.8 |
| TATACONSUM.NS | ✅ | LGB_HiReg | 6 Months | 0.8849 | 0.6800 | 0.2537 | 0.3696 | 166.2 |
| TATAELXSI.NS | ✅ | RandForest | 2 Months | 0.8958 | 0.7500 | 0.3333 | 0.4615 | 155.4 |
| TATAINVEST.NS | ✅ | Ensemble | 3 Months | 0.7358 | 0.8148 | 0.1325 | 0.2280 | 169.9 |
| TATAPOWER.NS | ⚠️ | RandForest | 1 Week | 0.7318 | 0.5000 | — | — | 157.8 |
| TATASTEEL.NS | ✅ | GradBoost | 1 Month | 0.6589 | 0.6923 | 0.0833 | 0.1488 | 162.5 |
| TATATECH.NS | ⚠️ | LGB_HiReg | 3 Months | 0.8191 | 0.5000 | — | — | 116.3 |
| TBOTEK.NS | ✅ | StackEnsemble | 3 Months | 0.8085 | 1.0000 | 0.1000 | 0.1818 | 22.5 |
| TCS.NS | ⚠️ | TabNet | 1 Week | 0.7900 | 0.3556 | — | — | 158.2 |
| TECHM.NS | ✅ | XGB_HiReg | 3 Months | 0.7855 | 0.7429 | 0.1884 | 0.3006 | 164.2 |
| TECHNOE.NS | ✅ | RandForest | 3 Months | 0.6950 | 0.8824 | 0.0811 | 0.1485 | 178.8 |
| TEJASNET.NS | ✅ | RandForest | 3 Months | 0.9397 | 0.7234 | 0.6939 | 0.7083 | 161.9 |
| THELEELA.NS | ❌ | — | — | — | — | — | — | 0.2 |
| THERMAX.NS | ✅ | RandForest | 6 Months | 0.7778 | 0.9286 | 0.1048 | 0.1884 | 159.3 |
| TIINDIA.NS | ✅ | StackEnsemble | 2 Months | 0.7192 | 0.8000 | 0.0909 | 0.1633 | 176.8 |
| TIMKEN.NS | ✅ | StackEnsemble | 3 Months | 0.8191 | 0.9250 | 0.4353 | 0.5920 | 174.8 |
| TITAGARH.NS | ✅ | StackEnsemble | 2 Months | 0.8014 | 0.7500 | 0.3846 | 0.5085 | 153.2 |
| TITAN.NS | ✅ | Ensemble | 3 Months | 0.8245 | 0.7778 | 0.1284 | 0.2205 | 153.9 |
| TMPV.NS | ⚠️ | TabNet | 1 Week | 0.7270 | 0.4167 | — | — | 152.1 |
| TORNTPHARM.NS | ✅ | RandForest | 6 Months | 0.7857 | 0.7070 | 0.6416 | 0.6727 | 148.8 |
| TORNTPOWER.NS | ✅ | Ensemble | 2 Months | 0.7551 | 0.6154 | 0.0548 | 0.1006 | 149.8 |
| TRENT.NS | ✅ | RandForest | 2 Months | 0.7877 | 0.9000 | 0.0682 | 0.1268 | 148.4 |
| TRIDENT.NS | ✅ | GradBoost | 3 Months | 0.8387 | 0.8333 | 0.0526 | 0.0990 | 157.9 |
| TRITURBINE.NS | ✅ | Ensemble | 6 Months | 0.8591 | 0.7143 | 0.5288 | 0.6077 | 163.2 |
| TRIVENI.NS | ⚠️ | XGB_HiReg | 1 Week | 0.7290 | 0.5000 | — | — | 164.7 |
| TTML.NS | ✅ | LGB_HiReg | 2 Months | 0.9007 | 0.7458 | 0.5057 | 0.6027 | 162.0 |
| TVSMOTOR.NS | ✅ | XGB_HiReg | 6 Months | 0.6766 | 1.0000 | 0.2049 | 0.3401 | 153.9 |
| UBL.NS | ⚠️ | GradBoost | 1 Week | 0.8239 | 0.5000 | — | — | 152.1 |
| UCOBANK.NS | ✅ | RandForest | 2 Weeks | 0.7997 | 0.6667 | 0.0630 | 0.1151 | 161.6 |
| ULTRACEMCO.NS | ✅ | RandForest | 1 Month | 0.8096 | 0.7778 | 0.0583 | 0.1085 | 165.5 |
| UNIONBANK.NS | ✅ | XGB_HiReg | 6 Months | 0.7103 | 0.6949 | 0.9094 | 0.7878 | 160.9 |
| UNITDSPR.NS | ⚠️ | TabNet | 2 Weeks | 0.8094 | 0.5455 | — | — | 160.7 |
| UNOMINDA.NS | ✅ | XGBoost | 6 Months | 0.7143 | 1.0000 | 0.2800 | 0.4375 | 180.2 |
| UPL.NS | ✅ | StackEnsemble | 1 Month | 0.7384 | 0.7692 | 0.1163 | 0.2020 | 142.2 |
| USHAMART.NS | ✅ | StackEnsemble | 1 Month | 0.7318 | 0.8750 | 0.0805 | 0.1474 | 165.2 |
| UTIAMC.NS | ✅ | XGB_HiReg | 1 Month | 0.7483 | 0.6500 | 0.0823 | 0.1461 | 166.8 |
| VBL.NS | ⚠️ | Ensemble | 1 Month | 0.7318 | 0.5238 | — | — | 159.1 |
| VEDL.NS | ✅ | LGB_HiReg | 1 Week | 0.6559 | 0.7778 | 0.0921 | 0.1647 | 156.6 |
| VENTIVE.NS | ❌ | — | — | — | — | — | — | 0.3 |
| VGUARD.NS | ✅ | LightGBM | 2 Weeks | 0.8257 | 0.6923 | 0.0804 | 0.1440 | 158.9 |
| VIJAYA.NS | ✅ | RandForest | 1 Week | 0.7189 | 0.6087 | 0.0782 | 0.1386 | 173.4 |
| VMM.NS | ❌ | — | — | — | — | — | — | 0.2 |
| VOLTAS.NS | ✅ | TabNet | 3 Months | 0.8546 | 0.6000 | 0.0714 | 0.1277 | 155.4 |
| VTL.NS | ✅ | RandForest | 6 Months | 0.6488 | 1.0000 | 0.0635 | 0.1194 | 183.1 |
| WAAREEENER.NS | ❌ | — | — | — | — | — | — | 0.2 |
| WELCORP.NS | ✅ | Ensemble | 3 Months | 0.7115 | 0.8182 | 0.0529 | 0.0994 | 163.8 |
| WELSPUNLIV.NS | ✅ | RandForest | 6 Months | 0.7917 | 0.6667 | 0.0909 | 0.1600 | 158.8 |
| WHIRLPOOL.NS | ✅ | Ensemble | 1 Month | 0.7003 | 0.9231 | 0.0625 | 0.1171 | 160.6 |
| WIPRO.NS | ✅ | RandForest | 3 Months | 0.9354 | 0.7037 | 0.2714 | 0.3918 | 302.4 |
| WOCKPHARMA.NS | ✅ | XGBoost | 2 Months | 0.7329 | 0.7111 | 0.1829 | 0.2909 | 166.8 |
| YESBANK.NS | ✅ | StackEnsemble | 6 Months | 0.9365 | 1.0000 | 0.1579 | 0.2727 | 162.5 |
| ZEEL.NS | ✅ | TabNet | 2 Months | 0.7277 | 0.7027 | 0.1494 | 0.2464 | 161.9 |
| ZENSARTECH.NS | ✅ | LGB_HiReg | 2 Weeks | 0.7801 | 0.7000 | 0.0504 | 0.0940 | 165.3 |
| ZENTEC.NS | ✅ | StackEnsemble | 6 Months | 0.9127 | 0.7978 | 0.9467 | 0.8659 | 163.6 |
| ZFCVINDIA.NS | ✅ | RandForest | 6 Months | 0.8135 | 0.7196 | 0.5461 | 0.6210 | 157.8 |
| ZYDUSLIFE.NS | ✅ | RandForest | 1 Month | 0.8427 | 0.7647 | 0.1250 | 0.2149 | 161.8 |

---

## 8. Below-Threshold Stocks (trained but low predictive power)

These stocks trained successfully but recall/F1 fell below the deployment threshold. The best model and horizon are still recorded for reference.

| Symbol | Best Model | Horizon | Accuracy | Precision |
|--------|:----------:|:-------:|:--------:|:---------:|
| 3MINDIA.NS | RandForest | 2 Months | 0.8388 | 0.4848 |
| ABBOTINDIA.NS | XGBoost | 2 Months | 0.8268 | 0.5000 |
| ABFRL.NS | TabNet | 1 Month | 0.7602 | 0.5319 |
| AJANTPHARM.NS | Ensemble | 3 Months | 0.8028 | 0.5238 |
| AKZOINDIA.NS | TabNet | 2 Months | 0.7937 | 0.4400 |
| ALKYLAMINE.NS | StackEnsemble | 3 Months | 0.8199 | 0.5714 |
| ANANTRAJ.NS | Ensemble | 1 Month | 0.7018 | 0.5714 |
| APLLTD.NS | LGB_HiReg | 6 Months | 0.8716 | 0.5669 |
| APOLLOHOSP.NS | Ensemble | 6 Months | 0.7568 | 0.5652 |
| APTUS.NS | TabNet | 2 Weeks | 0.7464 | 0.5200 |
| ARE&M.NS | CatBoost | 2 Weeks | 0.7871 | 0.4286 |
| BAJAJ-AUTO.NS | XGBoost | 1 Month | 0.7485 | 0.5472 |
| BAJAJHLDNG.NS | TabNet | 1 Month | 0.7164 | 0.3563 |
| BALKRISIND.NS | Ensemble | 3 Months | 0.8944 | 0.5833 |
| BASF.NS | Ensemble | 1 Week | 0.7854 | 0.5333 |
| BBTC.NS | RandForest | 2 Weeks | 0.7680 | 0.5500 |
| BHARTIARTL.NS | RandForest | 1 Month | 0.7939 | 0.4667 |
| BHARTIHEXA.NS | StackEnsemble | 2 Weeks | 0.7464 | 0.3750 |
| BIKAJI.NS | RandForest | 2 Months | 0.7937 | 0.5833 |
| BLS.NS | GradBoost | 1 Week | 0.7186 | 0.3659 |
| BSOFT.NS | XGBoost | 2 Weeks | 0.7608 | 0.5625 |
| CAMS.NS | Ensemble | 1 Week | 0.7067 | 0.5714 |
| CENTRALBK.NS | Ensemble | 6 Months | 0.9589 | 0.3750 |
| CESC.NS | LGB_HiReg | 1 Week | 0.6938 | 0.4839 |
| CHALET.NS | XGB_HiReg | 1 Month | 0.7573 | 0.5556 |
| CHENNPETRO.NS | StackEnsemble | 3 Months | 0.5994 | 0.7978 |
| CIPLA.NS | RandForest | 3 Months | 0.8944 | 0.3600 |
| COALINDIA.NS | GradBoost | 2 Weeks | 0.7954 | 0.4375 |
| COLPAL.NS | TabNet | 1 Week | 0.7825 | 0.3429 |
| CRAFTSMAN.NS | RandForest | 3 Months | 0.5854 | 1.0000 |
| DLF.NS | Ensemble | 1 Month | 0.7792 | 0.5667 |
| EIHOTEL.NS | RandForest | 1 Month | 0.7865 | 0.5625 |
| EXIDEIND.NS | TabNet | 6 Months | 0.8955 | 0.4500 |
| FEDERALBNK.NS | TabNet | 6 Months | 0.5839 | 0.8824 |
| GICRE.NS | XGBoost | 1 Week | 0.7096 | 0.3188 |
| GODREJIND.NS | TabNet | 2 Months | 0.6958 | 0.5000 |
| GODREJPROP.NS | XGB_HiReg | 2 Months | 0.7651 | 0.5893 |
| HCLTECH.NS | RandForest | 6 Months | 0.7432 | 0.4444 |
| HOMEFIRST.NS | TabNet | 2 Months | 0.7063 | 0.5676 |
| ICICIPRULI.NS | Ensemble | 2 Months | 0.8298 | 0.5000 |
| IDFCFIRSTB.NS | TabNet | 1 Week | 0.7382 | 0.5667 |
| INDHOTEL.NS | Ensemble | 1 Week | 0.7568 | 0.5909 |
| INDIGO.NS | StackEnsemble | 2 Weeks | 0.7867 | 0.5000 |
| INFY.NS | CatBoost | 1 Week | 0.7597 | 0.3617 |
| IOB.NS | TabNet | 1 Week | 0.7740 | 0.5882 |
| IRB.NS | GradBoost | 1 Month | 0.7822 | 0.5238 |
| IRCON.NS | XGB_HiReg | 6 Months | 0.8065 | 0.5185 |
| IRCTC.NS | LGB_HiReg | 2 Weeks | 0.8545 | 0.5000 |
| JYOTHYLAB.NS | XGBoost | 1 Week | 0.7725 | 0.4211 |
| KIRLOSBROS.NS | XGB_HiReg | 1 Month | 0.7646 | 0.5600 |
| KOTAKBANK.NS | XGBoost | 3 Months | 0.8199 | 0.4444 |
| KPITTECH.NS | XGBoost | 2 Weeks | 0.6830 | 0.3991 |
| LICHSGFIN.NS | CatBoost | 2 Weeks | 0.8055 | 0.5000 |
| LODHA.NS | CatBoost | 1 Month | 0.7222 | 0.5472 |
| LUPIN.NS | LGB_HiReg | 1 Week | 0.7668 | 0.3750 |
| MANKIND.NS | XGBoost | 2 Weeks | 0.7781 | 0.5833 |
| MAPMYINDIA.NS | Ensemble | 2 Weeks | 0.8300 | 0.5385 |
| MAXHEALTH.NS | LGB_HiReg | 1 Week | 0.7239 | 0.5500 |
| MEDANTA.NS | LightGBM | 3 Months | 0.7345 | 0.5270 |
| NAUKRI.NS | RandForest | 1 Month | 0.8099 | 0.5263 |
| NCC.NS | RandForest | 2 Months | 0.7892 | 0.5556 |
| NEWGEN.NS | Ensemble | 1 Month | 0.7851 | 0.4706 |
| NIACL.NS | GradBoost | 1 Month | 0.8046 | 0.5938 |
| NTPC.NS | Ensemble | 1 Week | 0.7969 | 0.5455 |
| OIL.NS | CatBoost | 1 Week | 0.7139 | 0.5195 |
| ONGC.NS | TabNet | 3 Months | 0.7888 | 0.4074 |
| PATANJALI.NS | XGB_HiReg | 2 Weeks | 0.8660 | 0.5238 |
| PERSISTENT.NS | Ensemble | 2 Months | 0.7274 | 0.5946 |
| PGEL.NS | RandForest | 1 Week | 0.6838 | 0.5909 |
| PGHH.NS | CatBoost | 2 Weeks | 0.8818 | 0.5000 |
| PHOENIXLTD.NS | Ensemble | 2 Weeks | 0.7277 | 0.5882 |
| POWERGRID.NS | LGB_HiReg | 1 Month | 0.8143 | 0.4444 |
| RECLTD.NS | Ensemble | 3 Months | 0.9037 | 0.5000 |
| RVNL.NS | XGB_HiReg | 1 Week | 0.7568 | 0.5217 |
| SHREECEM.NS | Ensemble | 1 Week | 0.7907 | 0.5000 |
| SJVN.NS | TabNet | 2 Weeks | 0.7911 | 0.5263 |
| SUNDRMFAST.NS | Ensemble | 3 Months | 0.8781 | 0.5714 |
| SUNPHARMA.NS | CatBoost | 2 Weeks | 0.8393 | 0.3810 |
| SUNTV.NS | GradBoost | 1 Week | 0.7778 | 0.4375 |
| TATAPOWER.NS | RandForest | 1 Week | 0.7318 | 0.5000 |
| TATATECH.NS | LGB_HiReg | 3 Months | 0.8191 | 0.5000 |
| TCS.NS | TabNet | 1 Week | 0.7900 | 0.3556 |
| TMPV.NS | TabNet | 1 Week | 0.7270 | 0.4167 |
| TRIVENI.NS | XGB_HiReg | 1 Week | 0.7290 | 0.5000 |
| UBL.NS | GradBoost | 1 Week | 0.8239 | 0.5000 |
| UNITDSPR.NS | TabNet | 2 Weeks | 0.8094 | 0.5455 |
| VBL.NS | Ensemble | 1 Month | 0.7318 | 0.5238 |

---

## 9. Stocks with No Data

`ABLBL.NS`, `ACMESOLAR.NS`, `AEGISVOPAK.NS`, `AFCONS.NS`, `AGARWALEYE.NS`, `ATHERENERG.NS`, `BAJAJHFL.NS`, `ENRIN.NS`, `FIRSTCRY.NS`, `HEXT.NS`, `HYUNDAI.NS`, `IGIL.NS`, `IKS.NS`, `ITCHOTELS.NS`, `JSWCEMENT.NS`, `NIVABUPA.NS`, `NTPCGREEN.NS`, `OLAELEC.NS`, `ONESOURCE.NS`, `PREMIERENE.NS`, `SAGILITY.NS`, `SAILIFE.NS`, `SWIGGY.NS`, `THELEELA.NS`, `VENTIVE.NS`, `VMM.NS`, `WAAREEENER.NS`
