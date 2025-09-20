#!/usr/bin/env python3
"""
Startup script for Vyuu Copilot v2 API Server.

This script starts the FastAPI server with proper configuration and logging.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vyuu_copilot_v2.config.settings import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Start the API server."""
    try:
        # Load configuration
        config = get_config()
        
        logger.info("Starting Vyuu Copilot v2 API Server...")
        logger.info(f"Environment: {config.api.environment}")
        logger.info(f"Host: {config.api.host}")
        logger.info(f"Port: {config.api.port}")
        logger.info(f"Debug: {config.api.debug}")
        logger.info(f"CORS Origins: {config.api.cors_origins}")
        
        # Start the server
        uvicorn.run(
            "vyuu_copilot_v2.api:app",
            host=config.api.host,
            port=config.api.port,
            reload=config.api.debug,
            log_level="info",
            access_log=True,
            use_colors=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
