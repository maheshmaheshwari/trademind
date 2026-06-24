# SEC-001 — MFA Scope Bypass: Partial-Auth Token Accepted by Account-Management Routes

| Field | Value |
|---|---|
| **Severity** | High |
| **Category** | Authentication Bypass |
| **File** | `backend/api/routes/auth_routes.py` |
| **Status** | Fixed |

---

## The Problem

TradeMind's login flow issues two different JWT tokens:

1. **`scope=mfa`** — short-lived (5 min), issued after correct password, before TOTP is verified.
2. **`scope=full`** — full-access, issued only after TOTP verification succeeds.

The `get_current_user` dependency (lines 45–57) correctly enforced the `scope == "full"` check:

```python
async def get_current_user(authorization: Optional[str] = Header(None)):
    ...
    if payload.get("scope") != "full":
        raise HTTPException(status_code=401, detail="Incomplete authentication — please complete MFA")
```

However, **none of the 12 account-management route handlers used this dependency**. Each one
inlined its own token decode — without any scope check — meaning a `scope=mfa` token was
silently accepted as valid authentication:

```python
# BEFORE — vulnerable pattern (repeated in 12 routes)
@router.post("/auth/totp/disable")
async def totp_disable(req: TotpDisableRequest, authorization: Optional[str] = Header(None)):
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)        # succeeds for scope=mfa token
    if not payload:
        raise HTTPException(...)
    user = get_user(payload["user_id"])  # no scope check → proceeds with partial-auth token
```

### Affected Endpoints

| Endpoint | Impact if exploited |
|---|---|
| `POST /auth/totp/disable` | Disable victim's 2FA entirely |
| `POST /auth/totp/setup` | Overwrite victim's TOTP secret with attacker-controlled one |
| `POST /auth/totp/confirm` | Enable 2FA using attacker's secret |
| `POST /auth/password/set` | Set password on Google-only account |
| `PATCH /auth/me` | Modify profile (email, display name, phone) |
| `GET /auth/preferences` | Read account preferences |
| `PUT /auth/preferences` | Change account preferences |
| `GET /auth/sessions` | Enumerate active sessions |
| `DELETE /auth/sessions/{id}` | Revoke specific session |
| `DELETE /auth/sessions` | Revoke all sessions (lock user out) |
| `POST /auth/password/change` | Requires old password — partially mitigated |

---

## Exploit Scenario

1. Attacker intercepts a victim's `scope=mfa` token (e.g., via a phishing page that
   forwards credentials, or a MitM on an unencrypted connection). The token is valid
   for 5 minutes after the victim enters their password.

2. Attacker calls **`POST /auth/totp/setup`** with the stolen `scope=mfa` token.
   This overwrites the victim's `totp_secret` in the database with a new secret the
   attacker controls. No secondary verification is required — the endpoint writes
   unconditionally.

3. Attacker calls **`POST /auth/totp/confirm`** with a valid TOTP code from *their own*
   authenticator (seeded with the secret just written). 2FA is now linked to the
   attacker's device.

4. Attacker calls **`POST /auth/totp/disable`** (since they control the TOTP secret)
   or simply logs in using the victim's password + their own TOTP code.

5. **Result:** Full account takeover of a trading account with real or virtual financial
   assets. The victim loses access and has no indication their TOTP secret was replaced.

---

## The Fix

All 12 route handlers now use `Depends(get_current_user)` instead of inlining their own
token decode. The existing dependency already enforced `scope == "full"` — routes just
weren't using it.

```python
# AFTER — fixed pattern applied to all 12 routes
@router.post("/auth/totp/disable")
async def totp_disable(req: TotpDisableRequest, user: dict = Depends(get_current_user)):
    # get_current_user raises 401 if scope != "full"
    # user is already resolved and validated
    ...
```

### Changes Made

**`backend/api/routes/auth_routes.py`**:

- Added `Depends` to the FastAPI import line.
- Removed the duplicated 4-line token-decode boilerplate from all 12 protected handlers.
- Each handler now declares `user: dict = Depends(get_current_user)` as its auth parameter.
- The `Optional[str] = Header(None)` parameter was removed from all affected handlers
  (the dependency handles header extraction internally).

No logic changes were made beyond delegating auth to the shared dependency.

---

## Why This Happened

The `get_current_user` dependency was written correctly, but each route handler was written
before (or independently of) it, copying a simpler inline pattern that predated the MFA
flow. When the `scope` check was added to `get_current_user`, the inline copies were never
updated to match.

**Rule going forward:** All protected routes must use `Depends(get_current_user)`.
Never inline `decode_token` + `get_user` directly in a handler — the dependency is the
single source of truth for authentication.
