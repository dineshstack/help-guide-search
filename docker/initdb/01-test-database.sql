-- Runs once, when the database volume is first created.
-- The integration tests use their own database so they can never empty the real index.
CREATE DATABASE guides_test;
