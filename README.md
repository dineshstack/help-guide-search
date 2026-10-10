# help-guide-search

A small Python service that searches the public help guides of a workshop
management product. Given a question, it looks for the passages that answer
it, by meaning as well as by keyword. It also measures how well that works,
against the simpler method of letting an AI model pick guides from a list of
their titles.

It reads the public guides and nothing else. It is not connected to any live
system and holds no customer data.

## Status

In development; not deployed anywhere.

## The guide text

The guides are kept in this repository as one file, `data/guides.json`, so
that every run uses the same input. The file is written by
`scripts/fetch_guides.py` from the public help-centre API.

- The guide text belongs to its publisher and is included here with permission.
- It is copied unchanged, and each guide keeps the address of its public page.
- The MIT licence in `LICENSE` covers the code in this repository.
  It does not cover the guide text.

## Running the unit tests

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and its libraries.

```bash
uv python install 3.13
uv sync --group dev
uv run pytest
```

The unit tests need no database, no model download and no network.
