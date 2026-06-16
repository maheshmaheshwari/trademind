# TradeMind AI — Security Review Report

**Date:** 2026-06-12  
**Scope:** Full codebase (backend API, database, trading engine, ML pipeline, collectors, frontend)  
**Method:** 6 parallel agents covering all layers  

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 12    |
| MEDIUM   | 13    |
| **Total**| **25**|

---

## HIGH Severity Findings

---

### H-1 — All Autopilot Endpoints Are Unauthenticated

**File:** `backend/api/routes/autopilot.py`, lines 178–370  
**Category:** Missing Authentication / Privilege Escalation  
**Confidence:** 10/10

**Description:**  
Every autopilot route — `GET /api/autopilot/status`, `POST /api/autopilot/toggle`, `GET /api/autopilot/trades`, `POST /api/autopilot/trades`, `DELETE /api/autopilot/trades/{trade_id}` — accepts a `user_id` directly from the request body or query string with zero JWT authentication. No `Depends(get_current_user)` guard is present on any of them.

**Exploit Scenario:**  
An unauthenticated attacker sends:
```
POST /api/autopilot/toggle {"user_id": 5}
POST /api/autopilot/trades {"user_id": 5, "symbol": "RELIANCE.NS", "mode": "LIVE",
  "qty": 500, "amount": 1400000, "entry": 2800, "target": 3200, "sl": 2600}
```
This turns on autopilot for victim user 5 and immediately executes a ₹14,00,000 real Angel One market order with no credentials. `DELETE /api/autopilot/trades/{trade_id}` can also revoke any user's active trades with no auth.

**Fix:**  
Add `user=Depends(get_current_user)` to every autopilot route handler and assert `user["id"] == body.user_id` before any DB write.

---

### H-2 — MFA Token Accepted as Full Session Token (MFA Bypass)

**Files:** `backend/api/routes/trading.py` lines 86–97, `backend/api/auth.py` lines 46–55  
**Category:** Authentication Bypass  
**Confidence:** 10/10

**Description:**  
After first-factor login, the server issues a short-lived `mfa_token` JWT with `scope="mfa"`. Only the `/auth/login/mfa` handler checks `scope == "mfa"`. Every other protected route uses `get_current_user`, which only validates the JWT signature and expiry — it never checks `scope`. The `mfa_token` is therefore a valid bearer token for all protected endpoints.

**Exploit Scenario:**  
1. Attacker knows victim's username/password.  
2. `POST /api/trading/login` → `{"mfa_required": true, "mfa_token": "<jwt>"}`.  
3. Attacker uses that `mfa_token` directly on `POST /api/trading/execute-signal` — full trading access without TOTP.

**Fix:**  
In `get_current_user`, add:
```python
if payload.get("scope") != "full":
    raise HTTPException(status_code=401, detail="Incomplete authentication")
```

---

### H-3 — Unauthenticated User Lookup + Legacy Account Creation

**File:** `backend/api/routes/trading.py`, lines 153–179  
**Category:** Missing Authentication / Information Disclosure  
**Confidence:** 10/10

**Description:**  
Three endpoints have no authentication:
- `POST /api/trading/user` — creates an account; default password is `hash_password(req.username)` (password equals username).
- `GET /api/trading/user/{user_id}` — returns full profile (balance, P&L, email) for any sequential integer ID.
- `GET /api/trading/user/by-username/{username}` — same disclosure by username.

**Exploit Scenario:**  
An attacker iterates `GET /api/trading/user/1`, `/2`, … to harvest all users' emails and balances. Separately, `POST /api/trading/user {"username": "victim"}` pre-registers a username with password = username, then immediately logs in with those credentials.

**Fix:**  
Remove these legacy endpoints entirely, or add `user=Depends(get_current_user)` and enforce `user["id"] == user_id`.

---

### H-4 — Password Reset OTP Logged in Plaintext

**File:** `backend/api/routes/auth_routes.py`, line 225  
**Category:** Sensitive Data Exposure  
**Confidence:** 9/10

**Description:**  
```python
logger.info("[DEV] Password reset OTP for %s: %s", req.email, otp)
```
The 6-digit OTP is logged at `INFO` level with no environment guard. Logs are written to `logs/YYYY-MM-DD.log`. Any process with log read access (log aggregators, monitoring dashboards, other workers) can extract valid OTPs.

**Exploit Scenario:**  
Attacker triggers `POST /auth/password/reset-request` for a target email, reads the OTP from the log stream, and calls `POST /auth/password/reset-confirm` within the 15-minute validity window to reset the victim's password without email access.

**Fix:**  
Remove the `logger.info` line entirely, or gate it: `if os.getenv("APP_ENV") == "development": logger.debug(...)`.

---

### H-5 — SQL Injection via Dynamic Column Names in UPDATE Statements

**File:** `backend/api/routes/auth_routes.py`, lines 152–161 and 343–348  
**Category:** SQL Injection (Column-Name Injection)  
**Confidence:** 8/10

**Description:**  
Both `PATCH /auth/me` and `PUT /auth/preferences` build `SET` clauses by interpolating Pydantic model field names directly into SQL f-strings:
```python
set_clause = ", ".join(f"{k} = ?" for k in updates)
_execute(conn, f"UPDATE users SET {set_clause} WHERE id = ?", tuple(values))
```
Values are parameterized, but column names are not. Current Pydantic models only emit safe field names. However, the pattern is one field extension away from allowing an attacker to target privileged columns (`password_hash`, `totp_secret`, `virtual_balance`).

**Exploit Scenario:**  
A developer extends `UpdateProfileRequest` with `virtual_balance: Optional[float]`. An authenticated user sends `PATCH /auth/me {"virtual_balance": 99999999}` and sets their own trading balance to any value.

**Fix:**  
Use an explicit allowlist for updatable columns:
```python
ALLOWED_FIELDS = {"display_name", "email", "phone"}
updates = {k: v for k, v in req.dict().items() if v is not None and k in ALLOWED_FIELDS}
```

---

### H-6 — SQL Injection via Unparameterized Table Name in `_col_names()`

**Files:** `backend/trading/trading_engine.py` line 106, `backend/trading/risk_manager.py` line 18, `backend/trading/price_monitor.py` line 58  
**Category:** SQL Injection (Table-Name Injection)  
**Confidence:** 8/10

**Description:**  
```python
def _col_names(conn, table: str) -> List[str]:
    cur = _execute(conn, f"SELECT * FROM {table} LIMIT 0")
```
The `table` parameter is interpolated directly into SQL. All current callers use hardcoded literals, but the function is public with no allowlist guard.

**Exploit Scenario:**  
Any future caller that passes a variable table name (e.g., from a query parameter or config file) immediately enables arbitrary SQL: `table = "users WHERE 1=1; DROP TABLE orders; --"`.

**Fix:**  
```python
_ALLOWED_TABLES = {"users", "orders", "positions", "risk_settings", "trade_signals"}
if table not in _ALLOWED_TABLES:
    raise ValueError(f"Table '{table}' not permitted")
```

---

### H-7 — Trading Capacity Check Uses Wrong Quantity (Logic Flaw → Oversized Live Orders)

**File:** `backend/trading/trading_engine.py`, lines 290–311  
**Category:** Business Logic Flaw / Financial Integrity  
**Confidence:** 9/10

**Description:**  
The platform capacity check at line 307 evaluates the caller-supplied `max_safe_qty` parameter. The actual order `quantity` is not calculated until line 311 (`quantity = int(investment_amount / buy_price)`). An attacker can submit a small `max_safe_qty` to pass the capacity guard while `investment_amount / buy_price` yields a far larger quantity that is committed to the database unchecked.

**Exploit Scenario:**  
User sends `investment_amount=5000000, buy_price=2800, max_safe_qty=1`. Capacity check sees qty=1 and passes. Actual committed quantity = 1785 shares. In LIVE mode this places a ₹50L Angel One order, exhausting platform-wide signal allocation.

**Fix:**  
Calculate `quantity = int(investment_amount / buy_price)` before the capacity check and use that value consistently in both the guard and the order insertion.

---

### H-8 — Insecure Deserialization: `joblib.load` Without Integrity Check

**Files:** `backend/analysis/signals.py` line 110, `backend/generate_trades.py` line 293  
**Category:** Insecure Deserialization (CWE-502)  
**Confidence:** 9/10

**Description:**  
`joblib` uses pickle internally. `generate_signal()` loads model files from `final_models/` using `joblib.load()` with no checksum or signature verification. `generate_trades.py` loads every `*_final.pkl` file in the directory.

**Exploit Scenario:**  
An attacker with write access to `final_models/` (via path traversal, compromised CI/CD, or container misconfiguration) places a crafted pickle file. The next scheduler run or `POST /api/signals/refresh` call triggers RCE as the FastAPI process user.

**Fix:**  
Store SHA-256 checksums of all model files in the database at training time. Verify before every `joblib.load()`. Ensure `final_models/` is a read-only mount for the web server process.

---

### H-9 — Path Traversal via `symbol` Parameter into `joblib.load`

**File:** `backend/analysis/signals.py`, lines 100–110  
**Category:** Path Traversal (CWE-22)  
**Confidence:** 8/10

**Description:**  
```python
final_path = os.path.join(_backend_dir, "final_models", f"{symbol}_final.pkl")
```
`symbol` is user-supplied via HTTP and is not sanitized. `os.path.join` does not neutralize `..` sequences. A crafted symbol like `../../tmp/evil` resolves outside `final_models/`.

**Exploit Scenario:**  
Attacker controls a writable directory at a traversed path and places a malicious pickle there. Any API path that calls `generate_signal(df, "../../tmp/evil")` loads it with `joblib.load`, achieving RCE.

**Fix:**  
Validate symbol before path construction:
```python
import re
if not re.fullmatch(r'[A-Z0-9]+(?:\.NS)?', symbol.upper()):
    raise ValueError(f"Invalid symbol: {symbol}")
resolved = os.path.realpath(final_path)
assert resolved.startswith(os.path.realpath(os.path.join(_backend_dir, "final_models")))
```

---

### H-10 — SQL Injection via f-string `timeout` in `SET statement_timeout`

**File:** `backend/analysis/model_training.py`, line 294  
**Category:** SQL Injection  
**Confidence:** 8/10

**Description:**  
```python
cur.execute(f"SET statement_timeout = '{timeout}'")
```
The `timeout` parameter is interpolated directly into a PostgreSQL `SET` statement. All current callers pass hardcoded strings (`"30s"`, `"60s"`), but the function signature accepts arbitrary input.

**Exploit Scenario:**  
Any future caller passing a user-influenced `timeout` value enables quote injection that can break out of the string literal context.

**Fix:**  
Validate before interpolation:
```python
if not re.fullmatch(r'\d+[ms]s?', timeout):
    raise ValueError(f"Invalid timeout: {timeout}")
```

---

### H-11 — JWT Stored in `localStorage` (Persistent Token Theft)

**Files:** `frontend/src/api.ts` lines 31/57/72, `frontend/src/AuthContext.tsx` line 39, `frontend/src/pages/AuthPage.tsx` lines 154/168/182  
**Category:** Insecure Token Storage  
**Confidence:** 9/10

**Description:**  
The JWT (`trademind_token`) is written to `localStorage` on every login. `localStorage` is synchronously readable by any JavaScript in the same origin — including compromised npm dependencies, CDN injections, or browser extensions.

**Exploit Scenario:**  
A malicious npm dependency calls `localStorage.getItem('trademind_token')` and exfiltrates it. With a valid JWT the attacker calls `POST /api/trading/execute-signal` and `POST /api/trading/square-off-all/{userId}` to trade or liquidate positions.

**Fix:**  
Store the JWT in an `httpOnly; SameSite=Strict; Secure` cookie set by the backend. The frontend never touches `document.cookie` for httpOnly cookies, making it immune to JS-based theft.

---

### H-12 — "Remember Me" Checkbox Is a No-op — Token Always Persisted

**File:** `frontend/src/pages/AuthPage.tsx`, lines 86 and 154/168/373–376  
**Category:** Broken Security Control  
**Confidence:** 10/10

**Description:**  
A "Remember me" checkbox (defaulted `true`) is rendered on the login form, but the `remember` state variable is never read during token storage. Both login paths unconditionally write to `localStorage` regardless of the checkbox.

**Exploit Scenario:**  
A user on a shared computer unchecks "Remember me" expecting the session to end on tab close. The token persists in `localStorage` indefinitely. The next user of the machine opens the app and is auto-logged-in as the previous user, gaining full access to their portfolio and trade execution.

**Fix:**  
When `remember === false`, write to `sessionStorage` instead of `localStorage`; when `true`, write to `localStorage`.

---

## MEDIUM Severity Findings

---

### M-1 — News and Signal History Endpoints Lack Authentication

**File:** `backend/api/routes/news.py`, lines 30–124  
**Category:** Missing Authentication / IDOR  
**Confidence:** 9/10

**Description:**  
`GET /api/news/watchlist/{user_id}`, `GET /api/news/watchlist/{user_id}/summary`, and `GET /api/signals/history/{user_id}` accept a `user_id` path parameter with no JWT check. They expose which stocks a user tracks and their full trade signal history (symbols, prices, timestamps).

**Fix:**  
Add `user=Depends(get_current_user)` and enforce `user["id"] == user_id`.

---

### M-2 — Broker Encryption Key Shares JWT Secret

**File:** `backend/api/routes/broker_routes.py`, lines 33–38  
**Category:** Insufficient Key Separation  
**Confidence:** 8/10

**Description:**  
Broker access tokens are encrypted with Fernet using a key derived from `JWT_SECRET`:
```python
key = base64.urlsafe_b64encode(hashlib.sha256(jwt_secret.encode()).digest())
```
A single compromised `JWT_SECRET` enables both JWT forgery and broker token decryption.

**Fix:**  
Use a separate `BROKER_ENCRYPTION_KEY` environment variable for Fernet.

---

### M-3 — Insecure Default SSL Mode for Database Connections

**File:** `backend/database/db.py`, lines 33 and 56  
**Category:** Insecure Transport  
**Confidence:** 8/10

**Description:**  
```python
PGSSLMODE = os.getenv("PGSSLMODE", "prefer")
```
`prefer` silently falls back to plaintext if the server does not offer TLS. If the DB is ever moved to a cloud host, all credentials and financial data travel unencrypted unless the operator explicitly overrides this.

**Fix:**  
Change default to `"require"`. Document that `disable` is only acceptable for local Docker dev.

---

### M-4 — Broker Access Tokens and TOTP Secrets Stored in Plaintext

**File:** `backend/database/schema_pg.py`, lines 333–334 and 346  
**Category:** Sensitive Data at Rest  
**Confidence:** 9/10

**Description:**  
`access_token`, `refresh_token`, and `totp_secret` are stored as plaintext `TEXT` columns. A DB credential leak, SQL injection, or direct volume access gives an attacker live broker tokens to place market orders and TOTP secrets to bypass MFA on every account.

**Fix:**  
Encrypt broker tokens at the application layer with Fernet before writing to DB. Decrypt only at call time. Consider eliminating `refresh_token` persistence entirely if sessions are re-authenticated on demand.

---

### M-5 — LIVE Order Placed After DB Commit — No Rollback on Broker Failure

**File:** `backend/trading/trading_engine.py`, lines 431–440  
**Category:** Financial Consistency Logic Flaw  
**Confidence:** 9/10

**Description:**  
In LIVE mode the sequence is: (1) DB transaction committed — balance debited, position created; (2) Angel One BUY order placed. If step 2 fails (API down, session expired), the error is logged but the function returns successfully. The user's balance is debited and a position is tracked, but no real order was placed — TradeMind shows a live position with no real-world counterpart.

**Fix:**  
Place the Angel One order before committing the DB transaction. Only commit if the broker returns a valid `order_id`. Roll back and return `503` on broker failure.

---

### M-6 — `pickle.load` Probe in `retrain_failed_models.py`

**File:** `backend/retrain_failed_models.py`, line 47  
**Category:** Insecure Deserialization  
**Confidence:** 9/10

**Description:**  
```python
with open(path, "rb") as f:
    pickle.load(f)
```
This probes every file in `final_models/` for loadability. `pickle.load` executes `__reduce__` during deserialization — a malicious file placed in the directory executes its payload before the `except Exception` can catch it.

**Fix:**  
Replace the bare `pickle.load` probe with a file header/magic-byte check, or use HMAC signatures (see H-8 fix). Remove bare `pickle.load` from production code paths.

---

### M-7 — `angel_tokens.json` Opened Without Schema Validation at Import Time

**File:** `backend/analysis/model_training.py`, line 44  
**Category:** Unvalidated External Data  
**Confidence:** 8/10

**Description:**  
```python
_raw = _json.load(open(_TOKENS_PATH))
```
The file is in `data/` which is also written to by `generate_trades.py`. If the directory is writable by the web server process, an attacker can replace `angel_tokens.json` with malformed content, poisoning symbol-to-token mappings for all 480 models.

**Fix:**  
Assert expected keys and value types after loading. Restrict `data/` to read-only for the inference process.

---

### M-8 — Unquoted Column Names in `_on_conflict_replace()`

**File:** `backend/database/db.py`, lines 156–158  
**Category:** Structural SQL Injection  
**Confidence:** 8/10

**Description:**  
```python
updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
```
Column names from caller-supplied lists are interpolated unquoted into SQL. All current callers use hardcoded literals. Any future caller passing variable column names enables SQL injection.

**Fix:**  
Use `psycopg2.sql.Identifier` for all column names:
```python
from psycopg2 import sql as pg_sql
updates = pg_sql.SQL(", ").join(
    pg_sql.SQL("{} = EXCLUDED.{}").format(pg_sql.Identifier(c), pg_sql.Identifier(c))
    for c in update_cols
)
```

---

### M-9 — GDELT Raw Date String Stored Verbatim on Parse Failure

**File:** `backend/collectors/gdelt_collector.py`, lines 148–155  
**Category:** Unvalidated External Data Persisted to DB  
**Confidence:** 9/10

**Description:**  
When GDELT's `seendate` field fails to parse, the raw API string is stored directly in the `published_at` column:
```python
except ValueError:
    published_at = raw_date  # ← arbitrary string from external API
```
A network-level adversary can inject an arbitrarily long or structured string into the DB, corrupting analytics queries and timestamp-based aggregates.

**Fix:**  
Reject articles with unparseable dates:
```python
except ValueError:
    continue  # discard rather than persist raw API string
```

---

### M-10 — No Timestamp Bounds Validation on External Candle Data

**Files:** `backend/collectors/angel_collector.py` line 307, `backend/collectors/ltp_fetcher.py` line 176  
**Category:** Unvalidated External Data  
**Confidence:** 8/10

**Description:**  
Timestamps from the Angel One API are parsed with `strptime` but not bounds-checked. A malformed response returning year 9999 would be written to TimescaleDB, causing hypertable chunk explosion and query plan failures.

**Fix:**  
```python
dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
if not (2000 <= dt.year <= 2035):
    logger.warning("Implausible timestamp %r — skipping", ts)
    continue
```

---

### M-11 — MFA Intermediate Token Not Cleared on Modal Dismiss

**File:** `frontend/src/pages/AuthPage.tsx`, lines 104 and 527  
**Category:** Sensitive Credential in Component State  
**Confidence:** 8/10

**Description:**  
After first-factor login, `mfaToken` is stored in React state. The "Back to login" button clears `mfaCode` and `mfaErr` but not `mfaToken`. If an exception occurs and a crash reporter (e.g., Sentry) captures component state, the intermediate credential is leaked.

**Fix:**  
Add `setMfaToken('')` to the dismiss handler:
```tsx
onClick={() => { setShowMfa(false); setMfaCode(''); setMfaErr(''); setMfaToken(''); }}
```

---

### M-12 — CSS Custom Property Injected from Unvalidated `localStorage` Value

**Files:** `frontend/src/pages/RiskSettingsPage.tsx` lines 107–108, `frontend/src/pages/SettingsPage.tsx` lines 451–458  
**Category:** Stored CSS Injection  
**Confidence:** 8/10

**Description:**  
```js
document.documentElement.style.setProperty('--accent', saved)
```
The accent color is read from `localStorage` and set verbatim as a CSS custom property with no format validation. An attacker who achieves transient XSS can persist a malicious CSS value that survives page reloads, spoofing UI elements (e.g., hiding LIVE mode indicators or altering button labels to trick users into unintended trades).

**Fix:**  
Validate before applying:
```js
if (/^#[0-9A-Fa-f]{6}$/.test(saved)) {
    document.documentElement.style.setProperty('--accent', saved);
}
```

---

### M-13 — Full PII User Object Stored in `localStorage`

**Files:** `frontend/src/AuthContext.tsx` lines 44/73/80, `frontend/src/pages/AuthPage.tsx` line 155  
**Category:** Sensitive Data Exposure in Client Storage  
**Confidence:** 9/10

**Description:**  
The full `User` object is serialized to `localStorage` under `trademind_user`, including `email`, `phone`, `virtual_balance`, `total_pnl`, `win_count`, and `loss_count`. This is redundant — the `AuthContext` already calls `getMe()` on startup to repopulate user state. Any JavaScript on the page can read all this PII without needing a valid API token.

**Fix:**  
Remove `trademind_user` from `localStorage` entirely. Rely on the `getMe()` API call to populate user state on startup.

---

## Priority Fix Order

| Priority | Finding | Impact |
|----------|---------|--------|
| 1 | H-1: Autopilot no auth | Anonymous RCE on live trades |
| 2 | H-2: MFA bypass via scope | Defeats entire MFA model |
| 3 | H-3: Unauthenticated user endpoints | Account takeover + PII harvest |
| 4 | H-4: OTP in logs | Password reset bypass |
| 5 | H-8 + H-9: Model file RCE | Arbitrary code execution |
| 6 | H-7: Capacity logic flaw | Oversized live orders |
| 7 | H-11 + H-12: JWT in localStorage | Persistent token theft |
| 8 | H-5 + H-6: Dynamic SQL column/table names | SQL injection (one change away) |
| 9 | M-4: Plaintext broker tokens | Broker account takeover |
| 10 | M-5: Order/DB commit ordering | Financial state corruption |
| 11 | Remaining M findings | Defense in depth |
