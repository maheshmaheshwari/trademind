# SEC-002 — Weak PRNG for Password-Reset OTP Generation

| Field | Value |
|---|---|
| **Severity** | Medium |
| **Category** | Cryptographic Weakness |
| **File** | `backend/api/routes/auth_routes.py` |
| **Status** | Fixed |

---

## The Problem

The password-reset OTP was generated using Python's built-in `random` module:

```python
# BEFORE — non-cryptographic PRNG
import random
import string

otp = "".join(random.choices(string.digits, k=6))
```

Python's `random` module is a **Mersenne Twister PRNG**, designed for simulations and
statistical sampling — not for security-sensitive values. It is explicitly documented by
Python as unsuitable for security or cryptographic use:

> "Warning: The pseudo-random generators of this module should not be used for security
> purposes. For security or cryptographic uses, see the `secrets` module."

### Why It Matters

A 6-digit numeric OTP (`000000`–`999999`) has only 1,000,000 possible values. The security
model relies on the OTP being *unpredictable* — not just that the keyspace is large enough
to resist brute force within the 15-minute window.

If the Mersenne Twister state were reconstructed (requiring approximately 624 consecutive
raw 32-bit outputs from the same process), an attacker could predict every future OTP
generated in that process, reducing the effective keyspace from 1,000,000 to 1.

While state reconstruction is difficult in practice in a multi-worker FastAPI deployment
(many sources of interleaved randomness), the fix is a one-line change with zero
trade-offs — there is no reason to use `random` here when `secrets` exists.

---

## Exploit Scenario (Theoretical Path)

1. Attacker creates many accounts / triggers many OTP-generating events that use the same
   `random` module in the same worker process, observing enough output to reconstruct the
   Mersenne Twister state.
2. Attacker predicts the next OTP generated for a password-reset request targeting a
   victim account.
3. Attacker calls `POST /auth/password/reset-confirm` with the predicted OTP and a new
   password of their choosing.
4. **Result:** Account takeover without any brute-force; a single request succeeds.

This is a difficult attack in practice but becomes significantly more feasible when
combined with other `random`-using code paths in the same process.

---

## The Fix

Replace `random.choices` with `secrets.choice` from the Python standard library.
`secrets` uses the OS CSPRNG (`/dev/urandom` on Linux/macOS, `BCryptGenRandom` on
Windows) and is the correct tool for any security-sensitive random value.

```python
# AFTER — cryptographically secure PRNG
import secrets

otp = "".join(secrets.choice("0123456789") for _ in range(6))
```

### Changes Made

**`backend/api/routes/auth_routes.py`**:

- Removed `import random` and `import string` (no longer used anywhere in the file).
- Added `import secrets`.
- Changed OTP generation in `password_reset_request` from `random.choices(string.digits, k=6)`
  to `secrets.choice("0123456789")` in a loop.

No other logic was changed. The OTP length (6 digits), hashing (`hash_password`),
storage, and expiry (15 minutes) remain identical.

---

## Rule Going Forward

Use `secrets` for all security-sensitive random values:

| Use case | Correct module |
|---|---|
| OTPs, reset tokens, session IDs | `secrets` |
| API keys, CSRF tokens | `secrets` |
| Simulations, shuffling, sampling | `random` |
| Unique IDs (non-security) | `uuid.uuid4()` |

Never use `random` for values whose unpredictability affects security.
