"""Start, stop or check a local Postgres on a machine that has no Docker.

    uv run python scripts/local_postgres.py start
    uv run python scripts/local_postgres.py status
    uv run python scripts/local_postgres.py stop

This is a convenience for development, not the project's database. The
project's database is the Postgres 16 image pinned in compose.yaml, and
reported results always come from that image. The Postgres started here comes
bundled inside a Python package, pixeltable-pgserver. It is Postgres 16 like
the pinned image, but not the same build, and its pgvector is not the pinned
build either. Use it
to write code and run the integration tests, never for numbers that go into
the report.

"start" keeps the database files in the folder .localdb at the top of the
repository (git ignores it), creates the databases "guides" and "guides_test"
if they are missing, and prints the two lines to put in .env. The server keeps
running after this script ends, until "stop" or until the machine restarts.
Running "start" again is harmless.

The package is not installed by default. Install it with:

    uv sync --group localdb
"""

import argparse
from pathlib import Path

import psycopg
from psycopg import sql

# The package is optional, so its absence is reported in plain words by main()
# instead of stopping here with a traceback.
try:
    import pixeltable_pgserver
    from pixeltable_pgserver.utils import PostmasterInfo
except ImportError:
    pixeltable_pgserver = None

# The folder with the database files: .localdb next to the scripts folder.
DATA_DIR = Path(__file__).resolve().parent.parent / ".localdb"

# The file in which a running Postgres records its process id.
PID_FILE = "postmaster.pid"

# The package bundles more than one Postgres. 16 is the major version of the
# image pinned in compose.yaml, so local behaviour is as close to it as it can be.
POSTGRES_VERSION = 16

# The real index, and a separate database the integration tests may empty.
DATABASES = ("guides", "guides_test")

NOT_INSTALLED = (
    "The package pixeltable-pgserver is not installed.\n"
    "Install it with: uv sync --group localdb"
)


def is_running() -> bool:
    """Say whether a Postgres server is running on the files in DATA_DIR."""
    # A running Postgres keeps a small file in its data folder that holds its
    # process id. The package reads that file and checks the process is alive.
    info = PostmasterInfo.read_from_pgdata(DATA_DIR)
    if info is None or not info.is_running():
        return False
    # After a crash, or a restart of the machine, the file is left behind and
    # its process id may by now belong to some other program.
    return info.process.name().startswith("postgres")


def create_missing_databases(admin_url: str) -> None:
    """Create each database in DATABASES that does not exist yet."""
    # CREATE DATABASE is refused inside a transaction, so autocommit is on.
    with psycopg.connect(admin_url, autocommit=True) as conn:
        for name in DATABASES:
            found = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
            if found is None:
                # A database name cannot be passed as a %s parameter.
                # sql.Identifier quotes it safely as a name instead.
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def start() -> int:
    if not is_running():
        # A file left behind by a server that is gone would make the package
        # believe that server is still there, so it is cleared away first.
        (DATA_DIR / PID_FILE).unlink(missing_ok=True)

    # get_server starts Postgres, or finds it if it is already running. The
    # first start also creates the files in DATA_DIR. cleanup_mode=None means:
    # leave the server running when this script ends.
    server = pixeltable_pgserver.get_server(DATA_DIR, cleanup_mode=None, postgres_version=POSTGRES_VERSION)
    create_missing_databases(server.get_uri())

    print("Postgres is running. Its files are in .localdb")
    print("Put these two lines in .env, in place of the two that are there:")
    print()
    print("DATABASE_URL=" + server.get_uri(database="guides"))
    print("TEST_DATABASE_URL=" + server.get_uri(database="guides_test"))
    return 0


def stop() -> int:
    # get_server would start a server if none were running, so check first.
    if not is_running():
        print("Postgres is not running.")
        return 0
    server = pixeltable_pgserver.get_server(DATA_DIR, cleanup_mode=None, postgres_version=POSTGRES_VERSION)
    server.stop()
    print("Postgres is stopped. Its files are kept in .localdb")
    return 0


def status() -> int:
    if is_running():
        print("Postgres is running.")
        return 0
    print("Postgres is not running. Start it with: uv run python scripts/local_postgres.py start")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A local Postgres for development, without Docker.")
    parser.add_argument("command", choices=["start", "stop", "status"])
    args = parser.parse_args(argv)

    if pixeltable_pgserver is None:
        print(NOT_INSTALLED)
        return 1

    if args.command == "start":
        return start()
    if args.command == "stop":
        return stop()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
