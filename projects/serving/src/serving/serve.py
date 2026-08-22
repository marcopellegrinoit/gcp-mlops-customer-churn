"""Container entrypoint: starts the Vertex AI batch-prediction HTTP server."""

import uvicorn

from serving.settings import settings

if __name__ == "__main__":
    uvicorn.run("serving.app:app", host="0.0.0.0", port=settings.aip_http_port)
