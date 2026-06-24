# Google OAuth — "Continue with Google" Implementation Plan

## Current State

| Layer | Status |
|---|---|
| Frontend button | Rendered — `onClick` is a stub showing "coming soon" toast |
| `GoogleIcon` component | Already exists in `AuthPage.tsx` |
| DB columns | `google_sub TEXT` and `avatar_url TEXT` already on `users` table (schema_pg.py:348–349) |
| Backend OAuth route | Does not exist |
| Google Client ID | Not configured |

---

## Chosen Approach — Google Identity Services (token flow)

Use the **Google Identity Services (GSI)** library with the **token-based flow**:

1. Frontend loads the GSI script and initialises it with a `client_id`
2. User clicks "Continue with Google" → Google popup appears
3. User picks account → Google returns a signed **`credential` (JWT `id_token`)**
4. Frontend POSTs the `id_token` to `POST /auth/google`
5. Backend verifies the token with Google, upserts the user, issues a TradeMind JWT
6. Frontend stores the JWT → navigates to dashboard

This avoids a redirect flow entirely — no callback URL, no session state needed.

---

## Step 1 — Google Cloud Console Setup

> Done once by the developer. Not in code.

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → **APIs & Services → Credentials**
2. Create a project named `trademind` (or use existing)
3. Create an **OAuth 2.0 Client ID** → Application type: **Web application**
4. Add Authorised JavaScript origins:
   - `http://localhost:5173` (Vite dev)
   - `https://yourdomain.com` (production)
5. No redirect URIs needed for the token flow
6. Copy the **Client ID** (looks like `123456789-abc.apps.googleusercontent.com`)
7. Add to `backend/.env`:
   ```
   GOOGLE_CLIENT_ID=123456789-abc.apps.googleusercontent.com
   ```
8. Add to `frontend/.env` (Vite):
   ```
   VITE_GOOGLE_CLIENT_ID=123456789-abc.apps.googleusercontent.com
   ```

---

## Step 2 — Backend: Install Dependency

```bash
cd backend
source venv/bin/activate
pip install google-auth
```

Add to `requirements.txt`:
```
google-auth>=2.29.0
```

---

## Step 3 — Backend: New Route `POST /auth/google`

**File:** `backend/api/routes/auth_routes.py`

### Request model
```python
class GoogleAuthRequest(BaseModel):
    credential: str   # the id_token from Google GSI
```

### Route logic
```python
@router.post("/auth/google")
async def google_auth(req: GoogleAuthRequest):
    """
    Verify a Google id_token, then sign in or auto-register the user.
    Returns the same {status, user, token} shape as POST /login.
    """
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    # 1. Verify the id_token with Google's public keys
    try:
        idinfo = google_id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {e}")

    google_sub  = idinfo["sub"]           # stable unique Google user ID
    email       = idinfo.get("email", "")
    name        = idinfo.get("name", "")
    avatar_url  = idinfo.get("picture", "")
    email_verified = idinfo.get("email_verified", False)

    if not email_verified:
        raise HTTPException(status_code=400, detail="Google email not verified")

    conn = get_connection()
    try:
        # 2a. Try to find existing user by google_sub (returning user)
        cur = _execute(conn, "SELECT * FROM users WHERE google_sub = ?", (google_sub,))
        user = _row_to_dict(cur)

        if not user:
            # 2b. Try to find by email (existing password user — link accounts)
            cur = _execute(conn, "SELECT * FROM users WHERE email = ?", (email,))
            user = _row_to_dict(cur)
            if user:
                # Link the Google account to the existing user
                _execute(conn, """
                    UPDATE users SET google_sub = ?, avatar_url = ? WHERE id = ?
                """, (google_sub, avatar_url, user["id"]))
                conn.commit()

        if not user:
            # 2c. New user — auto-register
            username = email.split("@")[0][:30]
            # Make username unique if taken
            cur = _execute(conn, "SELECT id FROM users WHERE username = ?", (username,))
            if _row_to_dict(cur):
                username = f"{username}_{google_sub[:6]}"

            _execute(conn, """
                INSERT INTO users
                  (username, display_name, email, password_hash, google_sub, avatar_url, virtual_balance)
                VALUES (?, ?, ?, '', ?, ?, 1000000)
            """, (username, name, email, google_sub, avatar_url))
            conn.commit()

            # Fetch the newly created user
            cur = _execute(conn, "SELECT * FROM users WHERE google_sub = ?", (google_sub,))
            user = _row_to_dict(cur)

        if not user:
            raise HTTPException(status_code=500, detail="Failed to create or find user")

        # 3. Issue TradeMind JWT
        from api.auth import create_token
        from trading.trading_engine import _safe_user
        token = create_token(user["id"], user["username"])
        return {"status": "success", "user": _safe_user(user), "token": token}

    finally:
        release_connection(conn)
```

### Key behaviours
| Scenario | Outcome |
|---|---|
| Returning Google user | Look up by `google_sub` → login |
| Existing password user (same email) | Link `google_sub` to account → login |
| Brand new user | Auto-register with Google profile, `password_hash = ''` |
| Unverified Google email | 401 rejected |
| `GOOGLE_CLIENT_ID` not set | 500 with clear message |

---

## Step 4 — Frontend: Install GSI Library

```bash
cd frontend
npm install @react-oauth/google
```

This is the official React wrapper for Google Identity Services.

---

## Step 5 — Frontend: Wrap App with Google Provider

**File:** `frontend/src/main.tsx`

```tsx
import { GoogleOAuthProvider } from '@react-oauth/google';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? '';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
    <App />
  </GoogleOAuthProvider>
);
```

---

## Step 6 — Frontend: Add API call in `tradeMindApiService.ts`

Add a mutation for the new endpoint:

```ts
googleAuth: builder.mutation<{ status: string; user: User; token: string }, { credential: string }>({
  query: (data) => ({ url: '/auth/google', method: 'POST', data }),
}),
```

Export: `useGoogleAuthMutation`

---

## Step 7 — Frontend: Update `AuthPage.tsx`

Replace the stub `handleGoogleLogin` with the real flow:

```tsx
import { useGoogleLogin } from '@react-oauth/google';
// OR for the One Tap / popup flow:
import { GoogleLogin } from '@react-oauth/google';

// Replace handleGoogleLogin stub:
const [googleAuth, { isLoading: googleLoading }] = useGoogleAuthMutation();

const handleGoogleLogin = async (credentialResponse: { credential?: string }) => {
  if (!credentialResponse.credential) return;
  try {
    const res = await googleAuth({ credential: credentialResponse.credential }).unwrap();
    login(res.token, res.user);   // same AuthContext.login() used by password flow
    navigate('/');
  } catch (err: any) {
    toast({ type: 'error', title: 'Google sign-in failed', message: err?.data?.detail ?? 'Please try again' });
  }
};
```

Replace the existing button with the GSI-powered button **or** keep the existing styled button and use `useGoogleLogin` hook:

```tsx
// Option A — keep existing styled button, use the hook
const googleSignIn = useGoogleLogin({
  onSuccess: async (tokenResponse) => { /* exchange access_token for id_token */ },
  flow: 'implicit',
});

// Option B — replace button with GSI component (handles all styling internally)
<GoogleLogin
  onSuccess={handleGoogleLogin}
  onError={() => toast({ type: 'error', title: 'Google sign-in failed' })}
  width="100%"
  shape="rectangular"
  theme="outline"
/>
```

**Recommended: Option A** — keeps the existing button design consistent with the TradeMind UI. Use the `useGoogleLogin` hook with `flow: 'auth-code'` to get an `id_token` back.

Actually with `@react-oauth/google`, the cleanest approach is:

```tsx
import { useGoogleLogin } from '@react-oauth/google';

// Inside component:
const initiateGoogleLogin = useGoogleLogin({
  onSuccess: async (codeResponse) => {
    // codeResponse.credential is the id_token when using implicit flow
    await handleGoogleLogin(codeResponse);
  },
  onError: () => toast({ type: 'error', title: 'Google sign-in cancelled' }),
});

// Keep existing button, just wire it:
<button onClick={() => initiateGoogleLogin()} ...>
  <GoogleIcon size={18} /> Continue with Google
</button>
```

---

## Step 8 — DB Migration (already done)

`google_sub` and `avatar_url` columns are already added via `ALTER TABLE IF NOT EXISTS` in `schema_pg.py:348–349`. No migration needed — `init_database()` handles it idempotently on startup.

Only change: ensure `password_hash` allows empty string for Google-only users. The column is `TEXT NOT NULL` — an empty string `''` satisfies this. No schema change needed.

---

## Step 9 — Edge Cases & Security

| Case | Handling |
|---|---|
| Google token tampered | `verify_oauth2_token` raises `ValueError` → 401 |
| Email not verified | Explicit check → 400 |
| Username collision on auto-register | Append first 6 chars of `google_sub` to make unique |
| Google-only user tries password login | `password_hash = ''` → `verify_password` returns False → login rejected gracefully |
| User unlinks Google (future) | Set `google_sub = NULL`, require them to set a password first |
| `GOOGLE_CLIENT_ID` missing in prod | 500 with clear error, not a silent failure |

---

## File Changelist

| File | Change |
|---|---|
| `backend/.env` | Add `GOOGLE_CLIENT_ID=...` |
| `backend/requirements.txt` | Add `google-auth>=2.29.0` |
| `backend/api/routes/auth_routes.py` | Add `POST /auth/google` route |
| `frontend/.env` | Add `VITE_GOOGLE_CLIENT_ID=...` |
| `frontend/package.json` | Add `@react-oauth/google` |
| `frontend/src/main.tsx` | Wrap with `<GoogleOAuthProvider>` |
| `frontend/src/services/tradeMindApiService.ts` | Add `googleAuth` mutation + export hook |
| `frontend/src/pages/AuthPage.tsx` | Replace stub with real `handleGoogleLogin` |

No DB schema changes needed — columns already exist.

---

## Implementation Order

1. Google Cloud Console setup (get Client ID)
2. Add `GOOGLE_CLIENT_ID` to `.env` files
3. Backend: `pip install google-auth` + add route
4. Frontend: `npm install @react-oauth/google`
5. Frontend: wrap `main.tsx` with provider
6. Frontend: add API mutation to service
7. Frontend: wire up `AuthPage.tsx`
8. Test: new user, returning user, existing password user email match
