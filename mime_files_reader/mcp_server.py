import anyio
import click
import mcp.types as types
import os
import logging
from mcp.server.lowlevel import Server

# Import our reader class
try:
    from mime_files_reader.reader import MimeFilesReader
except ImportError:
    logging.error("Failed to import MimeFilesReader. Ensure it's in the Python path.")
    import sys
    sys.exit(1)

# Configure basic logging for the server
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - MCP Server - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Get required configuration from environment variables
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Default model

if not API_KEY:
    logger.error("FATAL: GEMINI_API_KEY environment variable not set.")
    # In a real scenario, you might exit or raise a specific config error
    # For now, we'll let the MimeFilesReader raise its own error later


# --- Tool Definition ---
TOOL_NAME = "read_files"
TOOL_DESCRIPTION = (
    "Processes a list of local file paths with a question using a Generative AI model "
    "(like Gemini) capable of understanding various MIME types (images, PDFs, audio, etc.). "
    "Returns the generated text response."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "required": ["question", "files"],
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to ask the AI model about the provided files.",
        },
        "files": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "A list of strings, where each string is a path to a local file "
                "accessible by the server running this tool."
            ),
        },
        "output": {
            "type": "string",
            "description": (
                "Optional. A local path where the AI's response should be saved. "
                "If provided, the tool returns a confirmation message instead of the full response."
            ),
        },
         "auto_cleanup": {
            "type": "boolean",
            "default": True,
            "description": (
                "Optional (defaults to True). Whether to automatically delete files "
                "uploaded to the AI service backend after processing."
            ),
        }
    },
}


# --- MCP Server Implementation ---

@click.command()
# Add transport options later if needed, default to stdio for now
# @click.option("--transport", type=click.Choice(["stdio"]), default="stdio", help="Transport type")
def main() -> int:
    """Runs the MimeFilesReader MCP Server using stdio."""

    if not API_KEY:
        # Log again or exit cleanly if needed, prevents crash further down
        logger.error("Cannot start server: GEMINI_API_KEY is not set.")
        return 1

    logger.info(f"Starting MimeFilesReader MCP Server (Model: {MODEL_NAME})...")
    app = Server("mcp-mime-reader")

    @app.list_tools()
    async def list_available_tools() -> list[types.Tool]:
        """Lists the tools provided by this server."""
        logger.info("Received list_tools request.")
        return [
            types.Tool(
                name=TOOL_NAME,
                description=TOOL_DESCRIPTION,
                inputSchema=TOOL_INPUT_SCHEMA,
            )
        ]

    @app.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        """Handles incoming tool call requests."""
        logger.info(f"Received call_tool request for tool: {name}")
        if name != TOOL_NAME:
            logger.error(f"Unknown tool requested: {name}")
            # Consider raising a specific MCP error type later
            return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

        # Validate required arguments
        if "question" not in arguments or "files" not in arguments:
            logger.error(f"Missing required arguments 'question' or 'files'")
            return [types.TextContent(type="text", text="Error: Missing required arguments 'question' or 'files'")]
        if not isinstance(arguments["question"], str) or not isinstance(arguments["files"], list):
             logger.error(f"Invalid argument types for 'question' or 'files'")
             return [types.TextContent(type="text", text="Error: Invalid argument types for 'question' (string) or 'files' (list of strings)")]

        question = arguments["question"]
        files = arguments["files"]
        output = arguments.get("output") # Optional
        auto_cleanup = arguments.get("auto_cleanup", True) # Optional with default

        logger.info(f"Processing request: question='{question[:50]}...', files={files}, output={output}, cleanup={auto_cleanup}")

        try:
            # Instantiate the reader for each call (or manage a shared instance if preferred)
            # Assumes the server's CWD is appropriate for resolving relative paths if needed by reader
            # but client should ideally send paths accessible *from the server's perspective*.
            reader = MimeFilesReader(
                model_name=MODEL_NAME,
                google_genai_key=API_KEY,
                # working_dir="." # Or configure via env/arg if needed
            )
            result_text = await anyio.to_thread.run_sync(
                 reader.read, question, files, output, auto_cleanup
            )
            logger.info(f"Tool execution successful. Result starts: '{result_text[:100]}...'")
            return [types.TextContent(type="text", text=result_text)]

        except FileNotFoundError as e:
            logger.error(f"File not found during tool execution: {e}")
            return [types.TextContent(type="text", text=f"Error: File not found - {e}")]
        except ValueError as e:
             logger.error(f"Value error during tool execution: {e}")
             return [types.TextContent(type="text", text=f"Error: Invalid input or configuration - {e}")]
        except Exception as e:
            logger.exception(f"Unexpected error during tool execution: {e}")
            # Return a generic error message to the client
            return [types.TextContent(type="text", text=f"Error: An unexpected server error occurred: {type(e).__name__}")]


    # --- Run the server using stdio ---
    from mcp.server.stdio import stdio_server

    async def arun():
        try:
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        except Exception as e:
             logger.exception(f"Server main loop crashed: {e}")
        finally:
            logger.info("MCP Server shutting down.")


    try:
        anyio.run(arun)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl+C).")

    return 0

if __name__ == "__main__":
    main()
