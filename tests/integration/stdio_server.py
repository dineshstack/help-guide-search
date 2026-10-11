"""Started as a child process by test_mcp_service.py. Not a test itself.

It is the real MCP server on the test database, with the fake embedder in
place of the model, so that the stdio test needs no download and starts fast.
"""

import os

from guide_search.config import Settings
from guide_search.embedding import FakeEmbedder
from guide_search.mcp_server import build_server
from guide_search.service import Service

if __name__ == "__main__":
    service = Service(
        os.environ["TEST_DATABASE_URL"],
        FakeEmbedder(),
        Settings(threshold=0.3, default_mode="hybrid"),
    )
    build_server(service).run(show_banner=False)
