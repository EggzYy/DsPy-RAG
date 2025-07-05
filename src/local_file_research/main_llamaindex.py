"""
Main entry point for the local file research system using LlamaIndex.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import webbrowser

# Apply LiteLLM patch to fix __annotations__ error
try:
    from .litellm_patch import apply_patch, patch_dspy
    apply_patch()
    patch_dspy()
except ImportError:
    try:
        from .litellm_patch import apply_patch, patch_dspy
        apply_patch()
        patch_dspy()
    except ImportError:
        print("Warning: Could not import litellm_patch module")

# Set up logging (console only, no file logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_api():
    """
    Run the API server.
    """
    try:
        from .api_llamaindex_main import run_api as run_api_server
        run_api_server()
    except ImportError:
        logger.error("Failed to import API module. Running as subprocess.")
        subprocess.run([sys.executable, "-m", "src.local_file_research.api_llamaindex_main"])

def run_ui():
    """
    Run the UI server.
    """
    try:
        from .ui_llamaindex import main
        main()
    except ImportError:
        logger.error("Failed to import UI module. Running as subprocess.")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/local_file_research/ui_llamaindex.py"])

def run_both():
    """
    Run both API and UI servers.
    """
    # Import API port from config
    try:
        from .config import API_PORT
        logger.info(f"Using API port {API_PORT} from config")
    except ImportError:
        try:
            from .config import API_PORT
            logger.info(f"Using API port {API_PORT} from config")
        except ImportError:
            API_PORT = 8006
            logger.warning(f"Could not import API_PORT from config, using default: {API_PORT}")

    # Set environment variable for API port
    os.environ["API_PORT"] = str(API_PORT)

    # Perform cleanup before starting servers
    logger.info("Performing cleanup of temporary files...")

    # Clean up embeddings directory
    try:
        from .database_cleanup import cleanup_embeddings_directory
        logger.info("Cleaning up embeddings directory...")
        embedding_stats = cleanup_embeddings_directory()
        logger.info(f"Embeddings cleanup: removed {embedding_stats['total_files_removed']} files, freed {embedding_stats['human_readable_bytes_freed']}")
    except Exception as e:
        logger.error(f"Error cleaning up embeddings directory: {e}")

    # Clean up storage/content directory
    try:
        from .document_cleanup import cleanup_storage_files
        logger.info("Cleaning up storage/content directory...")
        storage_stats = cleanup_storage_files()
        logger.info(f"Storage cleanup: removed {storage_stats['total_files_removed']} files, freed {storage_stats['human_readable_bytes_freed']}")
    except Exception as e:
        logger.error(f"Error cleaning up storage directory: {e}")

    # Clean up storage/projects directory
    try:
        from .document_cleanup import cleanup_projects_folder
        logger.info("Cleaning up storage/projects directory...")
        projects_stats = cleanup_projects_folder()
        logger.info(f"Projects folder cleanup: removed {projects_stats['total_files_removed']} files, freed {projects_stats['human_readable_bytes_freed']}")
    except Exception as e:
        logger.error(f"Error cleaning up projects folder: {e}")

    # Start API server in a separate process
    api_process = subprocess.Popen([sys.executable, "-m", "src.local_file_research.api_llamaindex_main"])
    logger.info(f"Started API server with PID {api_process.pid} on port {API_PORT}")

    # Wait for API server to be fully initialized and ready to accept connections
    import requests
    max_retries = 30  # Maximum number of retries (30 * 0.5 seconds = 15 seconds max wait time)
    retry_count = 0
    api_ready = False

    logger.info(f"Waiting for API server to be ready at http://localhost:{API_PORT}...")

    while not api_ready and retry_count < max_retries:
        try:
            # Try to connect to the API server
            response = requests.get(f"http://localhost:{API_PORT}/health", timeout=1)
            if response.status_code == 200:
                api_ready = True
                logger.info(f"API server is ready at http://localhost:{API_PORT}")
            else:
                retry_count += 1
                time.sleep(0.5)
        except requests.exceptions.RequestException:
            retry_count += 1
            time.sleep(0.5)

    if not api_ready:
        logger.warning(f"API server did not respond after {max_retries * 0.5} seconds. Continuing anyway...")
    else:
        logger.info(f"API server is ready after {retry_count * 0.5} seconds")

    # Start UI server
    try:
        # Set environment variable for API URL
        os.environ["API_URL"] = f"http://localhost:{API_PORT}"

        # Create environment for the UI process
        ui_env = os.environ.copy()
        ui_env["API_URL"] = f"http://localhost:{API_PORT}"

        # Start UI process with the environment in headless mode (no auto-browser)
        ui_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/local_file_research/ui_llamaindex.py",
                "--server.headless=true",  # Run in headless mode to prevent auto-opening browser
                "--server.enableStaticServing=false",  # Disable static serving to reduce console output
                "--server.enableCORS=false",  # Disable CORS warnings
                "--server.enableXsrfProtection=false",  # Disable XSRF warnings
                "--logger.level=error"  # Only show error logs
            ],
            env=ui_env,
            stdout=subprocess.DEVNULL,  # Redirect stdout to /dev/null
            stderr=subprocess.PIPE  # Capture stderr for debugging
        )
        logger.info(f"Started UI server with PID {ui_process.pid}")
        logger.info(f"UI configured to connect to API at http://localhost:{API_PORT}")

        # Start Auth UI process with the environment in headless mode and redirect output
        auth_ui_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/local_file_research/auth_app.py",
                "--server.port=8502",
                "--server.headless=true",  # Run in headless mode to prevent auto-opening browser
                "--server.enableStaticServing=false",  # Disable static serving to reduce console output
                "--server.enableCORS=false",  # Disable CORS warnings
                "--server.enableXsrfProtection=false",  # Disable XSRF warnings
                "--logger.level=error"  # Only show error logs
            ],
            env=ui_env,
            stdout=subprocess.DEVNULL,  # Redirect stdout to /dev/null
            stderr=subprocess.DEVNULL   # Redirect stderr to /dev/null
        )
        logger.info(f"Started Auth UI server with PID {auth_ui_process.pid} on port 8502")
        logger.info(f"Auth UI configured to connect to API at http://localhost:{API_PORT}")
        logger.info(f"Auth UI available at http://localhost:8502 (not auto-opened)")

        # Wait for UI server to be ready
        logger.info("Waiting for UI server to be ready at http://localhost:8501...")

        # Wait for UI server to be fully initialized and ready
        ui_ready = False
        retry_count = 0
        max_retries = 30  # Maximum number of retries (30 * 0.5 seconds = 15 seconds max wait time)

        while not ui_ready and retry_count < max_retries:
            try:
                # Try to connect to the UI server
                response = requests.get("http://localhost:8501", timeout=1)
                if response.status_code == 200:
                    ui_ready = True
                    logger.info("UI server is ready at http://localhost:8501")
                else:
                    retry_count += 1
                    time.sleep(0.5)
            except requests.exceptions.RequestException:
                retry_count += 1
                time.sleep(0.5)

        if not ui_ready:
            logger.warning(f"UI server did not respond after {max_retries * 0.5} seconds. Continuing anyway...")
        else:
            logger.info(f"UI server is ready after {retry_count * 0.5} seconds")

        # Manually open the browser to the UI
        import webbrowser
        ui_url = "http://localhost:8501"
        logger.info(f"Opening browser to {ui_url}")
        webbrowser.open(ui_url)

        # Wait for all processes to complete
        api_process.wait()
        ui_process.wait()
        auth_ui_process.wait()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Stopping servers.")
        api_process.terminate()
        ui_process.terminate()
        auth_ui_process.terminate()
    except Exception as e:
        logger.error(f"Error running servers: {e}")
        api_process.terminate()
        ui_process.terminate()
        auth_ui_process.terminate()

def run_test():
    """
    Run tests for the LlamaIndex implementation.
    """
    try:
        from .test_llamaindex import main as run_test_main
        run_test_main()
    except ImportError:
        logger.error("Failed to import test module. Running as subprocess.")
        subprocess.run([sys.executable, "-m", "src.local_file_research.test_llamaindex"])

def run_migrate():
    """
    Run migration to LlamaIndex.
    """
    try:
        from .migrate_to_llamaindex import main as run_migrate_main
        run_migrate_main()
    except ImportError:
        logger.error("Failed to import migration module. Running as subprocess.")
        subprocess.run([sys.executable, "-m", "src.local_file_research.migrate_to_llamaindex"])

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Local File Research System")
    parser.add_argument("command", choices=["api", "ui", "both", "test", "migrate"], help="Command to run")
    args = parser.parse_args()

    # Create required directories
    os.makedirs("project_indices", exist_ok=True)
    os.makedirs("sessions", exist_ok=True)
    os.makedirs("storage", exist_ok=True)

    # Run command
    if args.command == "api":
        run_api()
    elif args.command == "ui":
        run_ui()
    elif args.command == "both":
        run_both()
    elif args.command == "test":
        run_test()
    elif args.command == "migrate":
        run_migrate()

if __name__ == "__main__":
    main()
