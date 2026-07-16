"""
Encrypted model store on Hugging Face Hub.

The private HF Hub repo is the durable, versioned home of the production
models. Every file is Fernet-encrypted with MODEL_KEY before upload, so the
Hub only ever holds ciphertext. Each upload is one git commit — the commit
history is the dated archive of every past model set (rollback = pass
`revision=` to sync_models).

Hub layout (repo: HF_MODELS_REPO, default <username>/trademind-models):
    models/{SYMBOL}.NS_final.pkl.enc
    data/angel_tokens.json.enc
    data/retrain_results.csv.enc

Local layout (backend/):
    final_models/{SYMBOL}.NS_final.pkl
    data/angel_tokens.json
    data/retrain_results.csv

Env (backend/.env): HF_TOKEN (write scope), MODEL_KEY, HF_MODELS_REPO (optional).

Usage:
    python scripts/model_store.py upload            # encrypt + push everything (bootstrap / post-retrain)
    python scripts/model_store.py sync              # pull + decrypt latest (server boot)
    python scripts/model_store.py history           # list commits (date, hash, message)
    python scripts/model_store.py sync <commit>     # restore the model set as of that commit
                                                    # (pin final-batch commits, i.e. "[N/N]")
"""
import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")

FINAL_DIR = _BACKEND_DIR / "final_models"
DATA_DIR = _BACKEND_DIR / "data"
DATA_FILES = ["angel_tokens.json", "retrain_results.csv"]

logger = logging.getLogger(__name__)


def _fernet() -> Fernet:
    key = os.environ.get("MODEL_KEY")
    if not key:
        raise RuntimeError("MODEL_KEY is not set — cannot encrypt/decrypt models")
    return Fernet(key.encode())


def _api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    return HfApi(token=token)


def _repo_id(api) -> str:
    return os.environ.get("HF_MODELS_REPO") or f"{api.whoami()['name']}/trademind-models"


BATCH_SIZE = 75          # files per commit — one big commit 504s on HF's commit endpoint
COMMIT_RETRIES = 3


def upload_all(commit_message: str = "model upload") -> str:
    """Encrypt final_models/*.NS_final.pkl + tracked data files and push.

    Pushes in batches of BATCH_SIZE files per commit (a single ~1 GB commit
    times out server-side). The final batch's commit completes the set — a
    snapshot at any final-batch commit is a complete, consistent model set.
    Called for the one-time bootstrap and at the end of every retrain run.
    """
    from huggingface_hub import CommitOperationAdd

    fernet = _fernet()
    api = _api()
    repo_id = _repo_id(api)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        (staging / "models").mkdir()
        (staging / "data").mkdir()

        models = sorted(FINAL_DIR.glob("*.NS_final.pkl"))
        if not models:
            raise RuntimeError(f"no models found in {FINAL_DIR}")
        for src in models:
            (staging / "models" / (src.name + ".enc")).write_bytes(
                fernet.encrypt(src.read_bytes())
            )
        for name in DATA_FILES:
            src = DATA_DIR / name
            if src.exists():
                (staging / "data" / (name + ".enc")).write_bytes(
                    fernet.encrypt(src.read_bytes())
                )

        files = sorted(staging.rglob("*.enc"))
        batches = [files[i : i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
        commit_url = ""
        for i, batch in enumerate(batches, 1):
            ops = [
                CommitOperationAdd(
                    path_in_repo=p.relative_to(staging).as_posix(),
                    path_or_fileobj=str(p),
                )
                for p in batch
            ]
            msg = f"{commit_message} [{i}/{len(batches)}]"
            for attempt in range(1, COMMIT_RETRIES + 1):
                try:
                    info = api.create_commit(
                        repo_id=repo_id, operations=ops, commit_message=msg
                    )
                    commit_url = getattr(info, "commit_url", "") or commit_url
                    logger.info("committed batch %d/%d (%d files)", i, len(batches), len(batch))
                    break
                except Exception as e:
                    if attempt == COMMIT_RETRIES:
                        raise
                    wait = 10 * attempt
                    logger.warning(
                        "batch %d attempt %d failed (%s) — retrying in %ds", i, attempt, e, wait
                    )
                    import time

                    time.sleep(wait)

    logger.info("Uploaded %d models to %s (%s)", len(models), repo_id, commit_message)
    return commit_url


def sync_models(revision: Optional[str] = None) -> int:
    """Download the encrypted store and decrypt into final_models/ and data/.

    Idempotent and incremental: snapshot_download reuses its local cache, and
    a file is only decrypted when the target is missing or the ciphertext is
    newer. Pass `revision` (a commit hash) to restore a past model set.
    Returns the number of files decrypted.
    """
    from huggingface_hub import snapshot_download

    fernet = _fernet()
    api = _api()
    repo_id = _repo_id(api)

    snapshot = Path(
        snapshot_download(repo_id, revision=revision, token=os.environ["HF_TOKEN"])
    )

    FINAL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    decrypted = 0
    for enc in snapshot.rglob("*.enc"):
        target = (FINAL_DIR if enc.parent.name == "models" else DATA_DIR) / enc.name[: -len(".enc")]
        if target.exists() and target.stat().st_mtime >= enc.stat().st_mtime and revision is None:
            continue
        target.write_bytes(fernet.decrypt(enc.read_bytes()))
        decrypted += 1
    logger.info("sync_models: %d file(s) decrypted from %s", decrypted, repo_id)
    return decrypted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "upload":
        msg = sys.argv[2] if len(sys.argv) > 2 else "model upload"
        print(upload_all(commit_message=msg))
    elif cmd == "sync":
        rev = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"{sync_models(revision=rev)} file(s) decrypted" + (f" @ {rev}" if rev else ""))
    elif cmd == "history":
        api = _api()
        for c in api.list_repo_commits(_repo_id(api)):
            print(f"{c.created_at:%Y-%m-%d %H:%M}  {c.commit_id[:8]}  {c.title}")
    else:
        print(__doc__)
        sys.exit(1)
