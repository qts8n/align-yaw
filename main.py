import os

from dotenv import find_dotenv, load_dotenv
import uvicorn


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 1))


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))

    host = os.getenv("HOST", "0.0.0.0")
    port = _env_int("PORT", 8000)
    workers = _env_int("UVICORN_WORKERS", _default_workers())
    log_level = os.getenv("LOG_LEVEL", "info")
    reload = _env_bool("RELOAD", False)

    if reload:
        workers = 1

    uvicorn.run(
        "equirect_shift.service:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    main()
