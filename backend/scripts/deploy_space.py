"""
Deploy the backend to a private Hugging Face Docker Space.

The Space repo is a filtered copy of backend/ (code + Dockerfile + README.md).
Secrets, models, data files, archives, and the SQLite fallback are never
uploaded: secrets go through the Space's secret store, models/data come from
the encrypted model store at boot (scripts/model_store.py).

Env (backend/.env): HF_TOKEN (write scope), HF_SPACE_REPO (optional,
default <username>/trademind).

Usage:
    python scripts/deploy_space.py create    # create the private Space + push secrets from .env
    python scripts/deploy_space.py push      # upload the code (initial deploy / redeploy)
    python scripts/deploy_space.py secrets   # re-push secrets from .env only
"""
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")

logger = logging.getLogger(__name__)

# Secrets copied from local .env into the Space's secret store.
# CORS_ALLOWED_ORIGINS is optional locally (defaults to dev origins in
# server.py) but should be set for production.
SECRET_KEYS = [
    "PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD",
    "ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PASSWORD", "ANGEL_MPIN",
    "ANGEL_TOTP_SECRET",
    "JWT_SECRET", "HF_TOKEN", "MODEL_KEY", "HF_MODELS_REPO",
    "CORS_ALLOWED_ORIGINS",
]

# Never uploaded to the Space repo. .env exclusion is non-negotiable.
IGNORE_PATTERNS = [
    ".env*", "venv/**", "final_models/**", "model_archives/**", "logs/**",
    "data/**", "*.db", "**/__pycache__/**", "*.pyc", ".pytest_cache/**",
    "tests/**", "*.log", "htmlcov/**", ".coverage*", ".DS_Store",
]


def _api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    return HfApi(token=token)


def _repo_id(api) -> str:
    return os.environ.get("HF_SPACE_REPO") or f"{api.whoami()['name']}/trademind"


def push_secrets(api=None) -> None:
    api = api or _api()
    repo_id = _repo_id(api)
    for key in SECRET_KEYS:
        value = os.environ.get(key)
        if not value:
            logger.warning("skipping secret %s — not set in .env", key)
            continue
        api.add_space_secret(repo_id, key, value)
        logger.info("secret set: %s", key)


def create() -> str:
    api = _api()
    repo_id = _repo_id(api)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker",
                    private=True, exist_ok=True)
    logger.info("space ready: https://huggingface.co/spaces/%s", repo_id)
    push_secrets(api)
    return repo_id


def push() -> str:
    api = _api()
    repo_id = _repo_id(api)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(_BACKEND_DIR),
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="deploy backend",
    )
    logger.info("pushed — build starts automatically: https://huggingface.co/spaces/%s", repo_id)
    return getattr(info, "commit_url", str(info))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "create":
        print(create())
    elif cmd == "push":
        print(push())
    elif cmd == "secrets":
        push_secrets()
    else:
        print(__doc__)
        sys.exit(1)
