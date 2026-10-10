"""The database: how to connect to Postgres, and the one table it holds.

Everything the search needs lives in a single table, `passages`: the passage
text, its vector for search by meaning, and a keyword-search column that
Postgres fills in by itself.

This file also defines the two errors that mean "we could not look":

- DatabaseUnavailable: Postgres could not be reached;
- IndexNotReady: Postgres answered, but the table is missing or holds no
  passages for the embedding model in use.

Neither is ever turned into an empty search result. "We could not look" and
"we looked and found nothing" must not look the same to the caller.
"""

import psycopg
from pgvector.psycopg import register_vector

# How long to wait for Postgres to answer before giving up. Without a limit,
# an address that never answers would keep a request waiting for minutes.
CONNECT_TIMEOUT_SECONDS = 5

# The statements of design section 4.4. Each one says IF NOT EXISTS, so the
# whole text can be run any number of times.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS passages (
  id              bigserial PRIMARY KEY,
  guide_id        integer NOT NULL,      -- the help centre's own id
  slug            text    NOT NULL,
  title           text    NOT NULL,
  url             text    NOT NULL,
  heading         text    NOT NULL,      -- '' for the intro
  position        integer NOT NULL,      -- order within the guide
  content         text    NOT NULL,      -- passage text, as shown to a reader
  search_text     text    NOT NULL,      -- title, heading and content
  content_hash    text    NOT NULL,      -- sha256 of search_text
  embedding_model text    NOT NULL,      -- the embedder's model_id
  embedding       vector(384) NOT NULL,
  tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
  UNIQUE (guide_id, position)
);

CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages USING GIN (tsv);
"""


class DatabaseUnavailable(Exception):
    """The database could not be reached."""


class IndexNotReady(Exception):
    """The table is missing, or holds no passages for the embedding model in use."""


def connect(url: str) -> psycopg.Connection:
    """Open a connection to Postgres that is ready for this project's queries.

    Raises DatabaseUnavailable when Postgres cannot be reached.
    """
    try:
        # autocommit: every statement is saved at once, unless it is inside a
        # "with conn.transaction():" block. Ingestion uses such a block so that
        # all its changes are kept together or not at all.
        conn = psycopg.connect(url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable(f"could not connect to the database: {error}") from error

    # Keyword search hands Postgres one word of the question at a time, and
    # Postgres sends a NOTICE for every word that is too common to search for
    # ("the", "of"). Those notices are expected, so ask not to be sent them.
    conn.execute("SET client_min_messages = warning")

    # A brand-new database has no vector type yet. ensure_schema creates it
    # and then does this step itself.
    if _has_vector_extension(conn):
        register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the extension, the table and the index if they do not exist."""
    conn.execute(SCHEMA_SQL)
    # register_vector teaches this connection to send and read vectors. It
    # looks the vector type up in the database, so it only works once the
    # extension exists, and it must come before any cursor that is kept and
    # reused: a cursor copies the connection's known types when it is made.
    register_vector(conn)


def count_passages(conn: psycopg.Connection, model_id: str) -> int:
    """Return how many passages are stored for this embedding model.

    Raises IndexNotReady when the table is missing or holds none for the
    model. The number returned is therefore never 0.
    """
    try:
        row = conn.execute(
            "SELECT count(*) FROM passages WHERE embedding_model = %s",
            (model_id,),
        ).fetchone()
    except psycopg.errors.UndefinedTable as error:
        raise IndexNotReady("the passages table does not exist; run the ingestion first") from error

    count = row[0]
    if count == 0:
        raise IndexNotReady(
            f"no passages are stored for the embedding model {model_id!r}; run the ingestion first"
        )
    return count


def _has_vector_extension(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'").fetchone()
    return row[0] > 0
