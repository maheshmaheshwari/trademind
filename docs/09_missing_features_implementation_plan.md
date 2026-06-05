# TradeMind — Missing Features Implementation Plan

**Date**: 2026-06-04  
**Scope**: All UI buttons / flows that show a UI but have no real backend connection.

---

## Audit Summary

| Area | Broken Feature | Severity |
|------|---------------|----------|
| Auth | Google OAuth button calls wrong handler | High |
| Auth | "Forgot password?" link has no handler | High |
| Auth | "Remember me" checkbox not used anywhere | Low |
| Settings – Profile | Email / Phone hardcoded as static strings | Medium |
| Settings – Profile | Personal info form is read-only, no save | Medium |
| Settings – Trading Prefs | Default account toggle `onChange={() => {}}` | Medium |
| Settings – Brokers | "Connect" buttons have no onClick | High |
| Settings – Brokers | Angel One shown as "Connected" hardcoded | Medium |
| Settings – Notifications | Toggles are local state only, never persisted | Medium |
| Settings – Security | TOTP shown as "Enabled" hardcoded badge | High |
| Settings – Security | "Update Password" button not wired | High |
| Settings – Security | Active sessions list is hardcoded static data | Medium |
| Watchlist | "Add to Watchlist" shows toast but makes no API call | High |
| Dashboard | "Save top 5 to watchlist" shows toast, no API call | Medium |
| Add Position Modal | Submit shows toast, no POST to backend | High |
| Market Page | Clock "15:24:08" is a hardcoded string | Low |

---

## Phase 1 — Backend Endpoints

### 1.1 Password Change (`backend/api/routes/auth_routes.py` — new file)

```
POST /auth/password/change
  Body: { current_password, new_password }
  Auth: Bearer JWT required
  Logic: verify current → bcrypt hash new → UPDATE users SET password_hash
  Returns: { message: "Password updated" }

POST /auth/password/reset-request
  Body: { email }
  Logic: generate 6-digit OTP, store in DB with 15-min expiry, send email via SMTP
  Returns: { message: "OTP sent" }

POST /auth/password/reset-confirm
  Body: { email, otp, new_password }
  Logic: verify OTP, check expiry, hash new password, clear OTP
  Returns: { message: "Password reset" }
```

**DB changes** (`schema_pg.py`):
```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_enabled BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS password_reset_otps (
  id        SERIAL PRIMARY KEY,
  email     TEXT NOT NULL,
  otp_hash  TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  used      BOOLEAN DEFAULT FALSE
);
```

---

### 1.2 Google OAuth (`backend/api/routes/auth_routes.py`)

Use `authlib` library (pip install authlib httpx).

```
GET  /auth/google/login
  Returns: { redirect_url: <Google OAuth URL> }

GET  /auth/google/callback?code=...&state=...
  Logic:
    1. Exchange code for tokens via Google
    2. Fetch user profile (email, name, picture)
    3. Find or create user row (users table) by google_sub
    4. Return JWT + user object same as /auth/login
```

**DB changes**:
```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
```

**Environment variables** (add to `backend/.env`):
```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:3000/auth/google/callback
```

---

### 1.3 TOTP / 2FA (`backend/api/routes/auth_routes.py`)

Use `pyotp` library (pip install pyotp qrcode[pil]).

```
POST /auth/totp/setup
  Auth: Bearer JWT
  Logic: generate pyotp secret, store in users.totp_secret (unconfirmed)
  Returns: { qr_uri: "otpauth://...", secret }

POST /auth/totp/confirm
  Auth: Bearer JWT
  Body: { code }
  Logic: verify code against stored secret, set totp_enabled = true
  Returns: { message: "2FA enabled" }

POST /auth/totp/disable
  Auth: Bearer JWT
  Body: { code }
  Logic: verify code, set totp_enabled = false, clear totp_secret
  Returns: { message: "2FA disabled" }

POST /auth/login  (modify existing)
  If totp_enabled: return { mfa_required: true, mfa_token: <short-lived JWT> }
  Add endpoint:
POST /auth/login/mfa
  Body: { mfa_token, totp_code }
  Returns: full JWT on success
```

---

### 1.4 User Profile CRUD (`backend/api/routes/auth_routes.py`)

```
GET  /auth/me
  Returns: { id, username, display_name, email, phone, avatar_url, totp_enabled, created_at }

PATCH /auth/me
  Body: { display_name?, email?, phone? }
  Auth: Bearer JWT
  Returns: updated user object
```

---

### 1.5 Session Management (`backend/api/routes/auth_routes.py`)

```
GET  /auth/sessions
  Auth: Bearer JWT
  Returns: list of active sessions from user_sessions table

DELETE /auth/sessions/:session_id
  Auth: Bearer JWT
  Logic: invalidate / delete session row

DELETE /auth/sessions  (logout all)
  Auth: Bearer JWT
  Logic: delete all sessions for user except current
```

**DB changes**:
```sql
CREATE TABLE IF NOT EXISTS user_sessions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL,
  device      TEXT,
  ip_address  TEXT,
  location    TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  last_seen   TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 1.6 Notification Preferences (`backend/api/routes/notifications.py` — extend)

```
GET  /api/notifications/preferences
  Auth: Bearer JWT
  Returns: { signal_change, price_alert, trade_executed, news_sentiment, eod_summary, weekly_report, channels: { email, push, sms } }

PUT  /api/notifications/preferences
  Auth: Bearer JWT
  Body: same shape
  Returns: updated preferences
```

**DB changes**:
```sql
CREATE TABLE IF NOT EXISTS notification_preferences (
  user_id        INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  signal_change  BOOLEAN DEFAULT TRUE,
  price_alert    BOOLEAN DEFAULT TRUE,
  trade_executed BOOLEAN DEFAULT TRUE,
  news_sentiment BOOLEAN DEFAULT FALSE,
  eod_summary    BOOLEAN DEFAULT TRUE,
  weekly_report  BOOLEAN DEFAULT FALSE,
  ch_email       BOOLEAN DEFAULT TRUE,
  ch_push        BOOLEAN DEFAULT TRUE,
  ch_sms         BOOLEAN DEFAULT FALSE,
  updated_at     TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 1.7 Trading Preferences (`backend/api/routes/auth_routes.py`)

```
GET  /auth/preferences
  Auth: Bearer JWT
  Returns: { default_account: "PAPER" | "LIVE", currency: "INR" }

PUT  /auth/preferences
  Body: { default_account?, currency? }
  Returns: updated preferences
```

**DB changes**:
```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS default_account TEXT DEFAULT 'PAPER';
ALTER TABLE users ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'INR';
```

---

### 1.8 Broker OAuth Connection (`backend/api/routes/broker_routes.py` — new file)

**Angel One** (SmartAPI — already partially integrated):
```
GET  /api/brokers
  Auth: Bearer JWT
  Returns: list of brokers with connected status from DB

POST /api/brokers/angel-one/connect
  Body: { client_id, password, totp }
  Logic: call SmartAPI login, store encrypted token in DB
  Returns: { connected: true, client_id }

DELETE /api/brokers/angel-one/disconnect
  Logic: remove stored token
  Returns: { connected: false }
```

**Zerodha / Upstox / Groww** — OAuth 2.0 flow (same pattern as Google):
```
GET  /api/brokers/zerodha/login     → redirect URL
GET  /api/brokers/zerodha/callback  → exchange code, store token

GET  /api/brokers/upstox/login
GET  /api/brokers/upstox/callback

(Groww does not have a public API — mark as "Coming Soon")
```

**DB changes**:
```sql
CREATE TABLE IF NOT EXISTS broker_connections (
  id             SERIAL PRIMARY KEY,
  user_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,
  broker         TEXT NOT NULL,  -- 'angel', 'zerodha', 'upstox'
  access_token   TEXT,           -- AES-256 encrypted
  refresh_token  TEXT,
  client_id      TEXT,
  expires_at     TIMESTAMPTZ,
  connected      BOOLEAN DEFAULT FALSE,
  created_at     TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, broker)
);
```

---

### 1.9 Watchlist Backend (`backend/api/routes/watchlist.py` — extend)

The file exists but verify these endpoints are complete and tested:

```
GET    /api/watchlist
POST   /api/watchlist          Body: { symbol }
DELETE /api/watchlist/:symbol
```

---

### 1.10 Add Position (`backend/api/routes/orders.py` — extend)

```
POST /api/portfolio/positions
  Auth: Bearer JWT
  Body: { symbol, quantity, buy_price, account_type, notes? }
  Logic: INSERT into positions, recalculate portfolio P&L
  Returns: created position object
```

---

### 1.11 Market Clock (minor)

```
GET /api/market/status
  Returns: { is_open: bool, current_time: "HH:MM:SS IST", next_open?, next_close? }
```

Logic: check if current IST time is Mon–Fri 09:15–15:30.

---

## Phase 2 — Frontend Changes

### 2.1 Google OAuth — `AuthPage.tsx`

Replace the broken `onClick={handleSubmit}` with:

```tsx
const handleGoogleLogin = () => {
  // Fetch redirect URL from backend, then redirect browser
  api.get('/auth/google/login').then(res => {
    window.location.href = res.data.redirect_url;
  });
};
```

Add a new route `/auth/google/callback` in `App.tsx` that:
1. Reads `?code=` and `?state=` from URL params
2. POSTs to backend callback (or backend redirects back with JWT in query)
3. Stores JWT, redirects to `/dashboard`

---

### 2.2 Forgot Password — `AuthPage.tsx`

Add `useState` for a `forgotMode` boolean. When true, show a two-step form:
1. Step 1: Email input → POST `/auth/password/reset-request`
2. Step 2: OTP + new password → POST `/auth/password/reset-confirm`

Wire the `<a>` tag: `onClick={() => setForgotMode(true)}`.

---

### 2.3 Settings — Profile Tab (`SettingsPage.tsx`)

- Replace hardcoded `'maheshmaheshwari983@gmail.com'` with `user?.email ?? ''`
- Replace `'+91 ··········'` with `user?.phone ?? ''`
- Remove `readOnly` from inputs (except username)
- Add a "Save Changes" button that calls `PATCH /auth/me`
- Add `email` and `phone` fields to the user object in `AuthContext.tsx`

---

### 2.4 Settings — Trading Preferences (`SettingsPage.tsx`)

```tsx
// Replace onChange={() => {}} with real API call
const [defaultAccount, setDefaultAccount] = useState<'PAPER'|'LIVE'>('PAPER');

useEffect(() => {
  api.get('/auth/preferences').then(r => setDefaultAccount(r.data.default_account));
}, []);

const handleAccountChange = (val: string) => {
  setDefaultAccount(val as 'PAPER'|'LIVE');
  api.put('/auth/preferences', { default_account: val });
};
```

---

### 2.5 Settings — Brokers Tab (`SettingsPage.tsx`)

- On mount: `GET /api/brokers` to get real connected status (replace hardcoded `BROKERS` array)
- Angel One "Connect" button: open a modal with ClientID / Password / TOTP fields, POST to `/api/brokers/angel-one/connect`
- Zerodha / Upstox "Connect" button: `window.location.href = <broker_login_url_from_api>`
- Connected → show "Disconnect" button that calls `DELETE /api/brokers/<broker>/disconnect`
- Groww: disable button, show "Coming Soon" tooltip

---

### 2.6 Settings — Notifications Tab (`SettingsPage.tsx`)

- On mount: `GET /api/notifications/preferences`, populate state
- Add a debounced or explicit "Save" that calls `PUT /api/notifications/preferences`
- Show a save-success toast on persist

---

### 2.7 Settings — Security Tab (`SettingsPage.tsx`)

**Password change**:
```tsx
const [pwForm, setPwForm] = useState({ current: '', next: '', confirm: '' });

const handlePasswordUpdate = async () => {
  if (pwForm.next !== pwForm.confirm) { setErr('Passwords do not match'); return; }
  await api.post('/auth/password/change', { current_password: pwForm.current, new_password: pwForm.next });
  toast({ type: 'success', title: 'Password updated' });
};
```
Wire to the "Update Password" button's `onClick`.

**TOTP / 2FA**:
- Fetch `user.totp_enabled` from `/auth/me`
- "Enable 2FA" button → `POST /auth/totp/setup` → show QR code (use `qrcode.react` package) → confirm with 6-digit code → `POST /auth/totp/confirm`
- "Disable 2FA" button → prompt for current TOTP code → `POST /auth/totp/disable`

**Active sessions**:
- On mount: `GET /auth/sessions`, render real list
- "Revoke" button per row: `DELETE /auth/sessions/:id`

---

### 2.8 Watchlist — Add Button (`WatchlistPage.tsx`)

The existing button shows a toast but never calls the API. Fix:

```tsx
// Replace toast-only handler with:
const handleAdd = async (symbol: string) => {
  await addToWatchlist(symbol);   // RTK Query mutation already exists
  toast({ type: 'success', title: `${symbol} added to watchlist` });
};
```

Verify `addToWatchlist` mutation is wired to `POST /api/watchlist`.

---

### 2.9 Dashboard — "Save to Watchlist" (`DashboardPage.tsx`)

```tsx
const handleSaveTopSignals = async () => {
  const top5 = signals.slice(0, 5).map(s => s.symbol);
  await Promise.all(top5.map(sym => addToWatchlist(sym)));
  toast({ type: 'success', title: 'Top 5 signals saved to your watchlist' });
};
```

---

### 2.10 Add Position Modal (`AddPositionModal.tsx`)

Replace toast-only submit with:

```tsx
const handleSubmit = async () => {
  await api.post('/api/portfolio/positions', {
    symbol: form.symbol,
    quantity: Number(form.qty),
    buy_price: Number(form.price),
    account_type: form.account,
    notes: form.notes,
  });
  onClose();
  toast({ type: 'success', title: `${form.symbol} position added` });
};
```

---

### 2.11 Market Page Clock (`MarketPage.tsx`)

Replace hardcoded `"15:24:08"` with a real-time clock:

```tsx
const [time, setTime] = useState('');
useEffect(() => {
  const tick = () => {
    const ist = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
    setTime(ist);
  };
  tick();
  const id = setInterval(tick, 1000);
  return () => clearInterval(id);
}, []);
```

---

## Phase 3 — API Service Layer (`src/services/tradeMindApiService.ts`)

Add RTK Query endpoints for all new backend routes:

```ts
// Auth
getMe, updateMe, changePassword, requestPasswordReset, confirmPasswordReset,
googleLoginUrl, totpSetup, totpConfirm, totpDisable,
getSessions, revokeSession,

// Preferences
getPreferences, updatePreferences,

// Notifications
getNotifPreferences, updateNotifPreferences,

// Brokers
getBrokers, connectBrokerAngelOne, disconnectBroker,
getBrokerLoginUrl,  // zerodha / upstox

// Market
getMarketStatus,
```

---

## Implementation Order

| Step | Task | Est. |
|------|------|------|
| 1 | DB schema migrations (all ALTER TABLE / CREATE TABLE) | 1h |
| 2 | `GET /auth/me` + `PATCH /auth/me` + fix Settings Profile tab | 2h |
| 3 | Password change + forgot password (backend + frontend) | 3h |
| 4 | Watchlist add button + Add Position modal (backend verify + frontend fix) | 1h |
| 5 | Notification preferences (backend + frontend) | 1.5h |
| 6 | Trading preferences default account toggle | 1h |
| 7 | Session management (backend + frontend) | 2h |
| 8 | Market clock real-time | 0.5h |
| 9 | TOTP / 2FA full flow | 3h |
| 10 | Angel One broker connect modal | 2h |
| 11 | Google OAuth full flow | 4h |
| 12 | Zerodha / Upstox OAuth | 3h |

**Total estimate: ~24 hours**

---

## Dependencies to Install

**Backend**:
```bash
pip install authlib httpx pyotp qrcode[pil] cryptography
```

**Frontend**:
```bash
npm install qrcode.react
```

---

## Notes

- All new backend endpoints must use the existing `get_current_user` JWT dependency for auth.
- Broker tokens (Angel One, Zerodha) must be AES-256 encrypted before storing; use `cryptography.fernet`.
- Google OAuth redirect URI must be added to the Google Cloud Console Authorized URIs.
- For email (forgot password OTPs): add `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` to `backend/.env`.
- TOTP QR codes should use `otpauth://totp/TradeMind:<username>?secret=<secret>&issuer=TradeMind` URI format.
