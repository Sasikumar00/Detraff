import uvicorn
import argparse
from fastapi import FastAPI

# OpenEnv imports
try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync' or 'pip install openenv-core'"
    ) from e

# Internal Project imports
# We use absolute imports to ensure compatibility with Docker and HF Spaces
try:
    from models import DetraffAction, DetraffObservation
    from server.detraff_env_environment import DetraffEnvironment
except ImportError:
    # Fallback for different execution contexts
    from .models import DetraffAction, DetraffObservation
    from .detraff_env_environment import DetraffEnvironment

# 1. Create the app instance
app = create_app(
    DetraffEnvironment,
    DetraffAction,
    DetraffObservation,
    env_name="detraff_env",
    max_concurrent_envs=1,
)

# 2. Define the main function (MANDATORY for Validator Step 3)
def main():
    """
    Standard entry point for the environment server.
    The validator calls this function to verify the server can start.
    """
    parser = argparse.ArgumentParser(description="Run the Detraff Env Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

# 3. Execution block
if __name__ == "__main__":
    main()