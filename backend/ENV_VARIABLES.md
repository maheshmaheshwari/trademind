# TradeMind AI — Environment Variables Guide

All variables go in `backend/.env`. Copy `.env.example` to `.env` and fill in each value using the guide below.

---

## 1. TimescaleDB (PostgreSQL)

Used by: `database/db.py` — every DB read/write in the app.

| Variable | Example Value | Required |
|---|---|---|
| `PGHOST` | `abc123.tsdb.cloud.timescale.com` | Yes |
| `PGPORT` | `5432` | Yes |
| `PGDATABASE` | `tsdb` | Yes |
| `PGUSER` | `tsdbadmin` | Yes |
| `PGPASSWORD` | `your_password` | Yes |

### Where to find these values

1. Go to **[cloud.timescale.com](https://cloud.timescale.com)** and sign up (free)
2. Click **Create service** → choose **TimescaleDB** → select free tier
3. After the service is created, click on it → **Connection info** tab
4. All 5 values (`Host`, `Port`, `Database`, `User`, `Password`) are listed there
5. The password is only shown once at creation — save it immediately

> For local development with Homebrew PostgreSQL, use:
> `PGHOST=localhost`, `PGPORT=5432`, `PGDATABASE=trademind`, `PGUSER=trademind`, `PGPASSWORD=trademind`

---

## 2. Angel One SmartAPI

Used by: all collectors, trading engine, GTT manager, LTP fetcher — anything that fetches live prices or places orders.

| Variable | Description | Required |
|---|---|---|
| `ANGEL_API_KEY` | API key for your registered app | Yes |
| `ANGEL_SECRET_KEY` | Secret key for your registered app | Yes |
| `ANGEL_CLIENT_ID` | Your Angel One login username | Yes |
| `ANGEL_PASSWORD` | Your Angel One login password | Yes |
| `ANGEL_TOTP_SECRET` | Base32 TOTP secret for 2FA | Yes |

### Where to find these values

**ANGEL_CLIENT_ID and ANGEL_PASSWORD**
- These are your Angel One trading account credentials
- Client ID is your login username (e.g. `A1234567`)
- Password is your Angel One account password

**ANGEL_API_KEY and ANGEL_SECRET_KEY**
1. Log in at **[smartapi.angelone.in](https://smartapi.angelone.in)**
2. Go to **My Apps** → click your app (or create one if none exists)
3. `API Key` and `Secret Key` are shown on the app detail page

**ANGEL_TOTP_SECRET**
1. Log in to your Angel One account → **Profile** → **Security Settings**
2. Enable TOTP authenticator
3. When the QR code is shown, there will be a text field with the raw base32 secret (looks like `JBSWY3DPEHPK3PXP`)
4. Copy that string — this is `ANGEL_TOTP_SECRET`
5. If you only see a QR code with no text, use a QR decoder (e.g. [zxing.org](https://zxing.org/w/decode.jspx)) to extract the `secret=` parameter from the URL

---

## 3. JWT Secret

Used by: `api/auth.py` — signs and verifies login tokens.

| Variable | Description | Required |
|---|---|---|
| `JWT_SECRET` | A long random string used to sign JWTs | Yes |

### Where to find this value

Generate it yourself — run this in your terminal:

```bash
openssl rand -hex 32
```

Copy the output (e.g. `a3f8c2e1b4d7...`) and use it as the value. Keep it secret and do not share it.

---

## 4. NewsAPI

Used by: `collectors/news_collector.py` — fetches financial news headlines.

| Variable | Description | Required |
|---|---|---|
| `NEWSAPI_KEY` | API key from newsapi.org | No (news disabled if missing) |

### Where to find this value

1. Sign up at **[newsapi.org](https://newsapi.org)**
2. After registration, your API key is shown on the dashboard immediately
3. Free tier: 100 requests/day (enough for the hourly news job)

---

## 5. Alpha Vantage

Used by: `collectors/alphavantage_collector.py` — fallback market data source.

| Variable | Description | Required |
|---|---|---|
| `ALPHAVANTAGE_API_KEY` | API key from alphavantage.co | No (only used as fallback) |

### Where to find this value

1. Go to **[alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)**
2. Enter your email — the key is emailed instantly, no account required
3. Free tier: 25 requests/day

---

## 6. App Config

Used by: `main.py` — controls server startup.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the FastAPI server listens on |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

These have sensible defaults — only set them if you need to change from the defaults.

---

## Summary Table

| Variable | Required | Where to get |
|---|---|---|
| `PGHOST` | Yes | Timescale Cloud → Connection info |
| `PGPORT` | Yes | Timescale Cloud → Connection info |
| `PGDATABASE` | Yes | Timescale Cloud → Connection info |
| `PGUSER` | Yes | Timescale Cloud → Connection info |
| `PGPASSWORD` | Yes | Timescale Cloud → Connection info (save at creation) |
| `ANGEL_API_KEY` | Yes | smartapi.angelone.in → My Apps |
| `ANGEL_SECRET_KEY` | Yes | smartapi.angelone.in → My Apps |
| `ANGEL_CLIENT_ID` | Yes | Your Angel One login username |
| `ANGEL_PASSWORD` | Yes | Your Angel One login password |
| `ANGEL_TOTP_SECRET` | Yes | Angel One → Profile → Security → TOTP setup |
| `JWT_SECRET` | Yes | Generate: `openssl rand -hex 32` |
| `NEWSAPI_KEY` | No | newsapi.org → Dashboard |
| `ALPHAVANTAGE_API_KEY` | No | alphavantage.co/support/#api-key |
| `PORT` | No | Default: `8000` |
| `LOG_LEVEL` | No | Default: `INFO` |
