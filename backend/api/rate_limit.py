"""
Shared rate limiter (audit findings H2, H3, M2, M15).

Keyed by remote IP. Imported both by api/server.py (to register the
SlowAPIMiddleware + exception handler once) and by individual route modules
(to apply @limiter.limit(...) to the specific endpoints that need it —
login, password-reset, signal-refresh).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
