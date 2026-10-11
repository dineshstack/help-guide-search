#!/bin/sh
# Runs once, when the database volume is first created.
#
# Creates the account the service connects as. It can log in and nothing more.
# Ingestion later gives it the right to read the passages table, and that is
# all it ever gets: it cannot change, add or delete anything.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v reader_password="$READER_PASSWORD" <<'SQL'
CREATE ROLE guides_reader LOGIN PASSWORD :'reader_password';
SQL
