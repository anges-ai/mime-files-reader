# MimeFilesReader

Reads and processes various MIME type files (images, PDFs, audio, etc.) using Google's Generative AI models (like Gemini Pro/Flash) capable of multi-modal understanding. Provides both a command-line interface (CLI) and a Model Context Protocol (MCP) server interface.

## Features

*   Leverages Google's Generative AI (Gemini) for understanding content within various file types.
*   Supports common MIME types like images (PNG, JPEG), PDFs, audio (MP3, WAV - depending on model support), etc.
*   Provides a simple CLI for quick queries about local files.
*   Offers an MCP server interface for integration with AI Agent frameworks (LangChain, LlamaIndex, AutoGen, custom agents).
*   Handles file uploads to the Google AI backend and optional automatic cleanup.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anges-ai/mime-files-reader.git # Replace with your repo URL
    cd mime_reader
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies and the package:**
    ```bash
    pip install -e .
    ```
    This installs the package in editable mode along with its dependencies listed in `setup.py` (which should read `requirements.txt`).

4.  **Set Environment Variables:**
    You need a Google Generative AI API key. Create a `.env` file in the project root or export the variable:
    ```dotenv
    # .env
    GEMINI_API_KEY="YOUR_GOOGLE_AI_API_KEY"
    # Optional: Specify a model, defaults to gemini-1.5-flash-latest
    # GEMINI_MODEL_NAME="gemini-2.0-flash"
    ```

    For complex tasks, consider using `gemini-2.5-pro-exp-03-25`, though the quota is currently low.

## Command-Line Interface (CLI) Usage

The original CLI provides a direct way to ask questions about local files.

```bash
mime-reader -q "Describe the main subject of this image." -f /path/to/your/image.jpg
```

```bash
mime-reader --question "Summarize the first page of this document." --files /path/to/document.pdf --output summary.txt
```

**Options:**

*   `-q`, `--question`: (Required) The question to ask the AI model.
*   `-f`, `--files`: (Required) One or more file paths. Use the option multiple times for multiple files (`-f file1.pdf -f image.png`).
*   `-o`, `--output`: (Optional) Path to save the text response to a file.
*   `--model`: (Optional) Specify the Google AI model name (overrides `GEMINI_MODEL_NAME` env var).
*   `--no-cleanup`: (Optional) Prevent automatic deletion of files uploaded to Google AI backend.

*(Note: Ensure your original `mime_files_reader/cli.py` and the corresponding entry point in `setup.py` are correctly configured for this.)*

## MCP Server Interface

This package now includes an MCP server, allowing AI agents or other applications to use the multi-modal file reading capabilities as a tool.

**Prerequisites:**

*   The environment where the server runs **must** have the `GEMINI_API_KEY` environment variable set.
*   Optionally, set `GEMINI_MODEL_NAME` to use a specific model (defaults to `gemini-1.5-flash-latest`).

**Running the Server:**

After installing the package (`pip install -e .`), you can run the server using the registered console script:

```bash
# Make sure GEMINI_API_KEY is set in this shell
export GEMINI_API_KEY="YOUR_GOOGLE_AI_API_KEY"
# export GEMINI_MODEL_NAME="gemini-1.5-pro-latest" # Optional

mime-reader-mcp-server
```

By default, the server listens for MCP messages over **standard input/output (stdio)**.

**Exposed Tool:**

The server exposes a single tool:

*   **Name:** `read_files`
*   **Description:** Processes a list of local file paths with a question using a Generative AI model (like Gemini) capable of understanding various MIME types (images, PDFs, audio, etc.). Returns the generated text response.
*   **Arguments:**
    *   `question` (string, required): The question to ask the AI model about the provided files.
    *   `files` (list of strings, required): A list of file paths. **Crucially, these paths must be accessible from the environment where the `mime-reader-mcp-server` process is running.**
    *   `output` (string, optional): A local path (relative to the server's working directory, or absolute) where the AI's response should be saved. If provided, the tool returns a confirmation message instead of the full response.
    *   `auto_cleanup` (boolean, optional, default: `true`): Whether to automatically delete files uploaded to the Google AI service backend after processing.

**MCP Client Example (Python):**

```python
import asyncio
import os
import sys
import shutil
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import types as mcp_types # Import types for clarity

# --- Configuration ---
SERVER_EXECUTABLE = shutil.which("mime-reader-mcp-server")
if not SERVER_EXECUTABLE:
    print("Error: Cannot find 'mime-reader-mcp-server'. Is the package installed?", file=sys.stderr)
    sys.exit(1)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY needed for server environment.", file=sys.stderr)
    sys.exit(1)

# --- Example Usage ---
async def run_mcp_client():
    server_params = StdioServerParameters(
        command=SERVER_EXECUTABLE,
        args=[], # No extra args needed for entry point
        env={ # Pass necessary env vars to the server process
            **os.environ,
            "GEMINI_API_KEY": API_KEY,
            "PYTHONUNBUFFERED": "1"
            # Optionally pass GEMINI_MODEL_NAME here too
        }
    )

    image_path = "/path/to/your/local/image.png" # Replace with a real path accessible by server

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session Initialized.")

            # Optional: List tools
            list_resp = await session.list_tools()
            print(f"Available Tools: {list_resp.tools}")

            # Call the read_files tool
            tool_args = {
                "question": "What objects are prominent in this image?",
                "files": [image_path]
            }
            print(f"\nCalling 'read_files' with args: {tool_args}")
            call_resp: mcp_types.CallToolResponse = await session.call_tool("read_files", tool_args)

            if hasattr(call_resp, 'content') and call_resp.content:
                 if isinstance(call_resp.content[0], mcp_types.TextContent):
                     print(f"\nServer Response:\n{call_resp.content[0].text}")
                 else:
                     print(f"\nReceived non-text content: {call_resp.content[0]}")
            elif getattr(call_resp, 'isError', False):
                 print(f"\nServer returned an error flag.")
            else:
                 print(f"\nReceived unexpected response structure: {call_resp}")

if __name__ == "__main__":
    # Make sure image_path above points to a real file
    # Make sure API_KEY is set
    try:
        asyncio.run(run_mcp_client())
    except Exception as e:
        print(f"Client error: {e}")

```

## Why Use This with AI Agents (MCP Integration)?

AI agent frameworks (like LangChain, LlamaIndex, AutoGen, CrewAI, or custom implementations) often orchestrate complex tasks by allowing a central Language Model (LLM) to delegate specific capabilities to external "tools." Integrating `MimeFilesReader` as an MCP tool provides several advantages:

1.  **Offloading Powerful Multi-modal Understanding:** Instead of requiring the primary agent LLM to have potentially expensive or less optimized built-in multi-modal capabilities, the agent can delegate tasks involving vision, document understanding, etc., to `MimeFilesReader`. This tool specifically utilizes Google's powerful Gemini models, which excel at these tasks. The agent simply needs to provide the file paths and a relevant question.

2.  **Secure Access to Local Files:** Agents, especially when deployed as services, should operate with least privilege. Giving the core agent direct filesystem access can be a security risk. The MCP server acts as a secure bridge. The agent instructs the *locally running* `mime-reader-mcp-server` (which has legitimate access) to process specific files, rather than accessing them directly.

3.  **Enabling Complex Workflows:**
    *   **PDF/Document Analysis:** An agent can ask the tool to "Extract the key financial figures from `/path/to/report.pdf`" or "Summarize the main arguments in `/path/to/research_paper.pdf`." The tool handles the PDF complexity, and Gemini provides the understanding.
    *   **Web Scraping + Vision:** An agent might use one tool (e.g., a Puppeteer/Selenium based MCP tool) to navigate a website and take a screenshot, saving it locally. It can then pass the screenshot's path to the `mime-reader-mcp-server` tool and ask, "What is the price shown for the product in `/tmp/screenshot.png`?" or "Describe the layout of this webpage section based on `/tmp/screenshot.png`."
    *   **Analyzing User Uploads:** In a chatbot application, if a user uploads an image or document, the backend can save it to a temporary location and use the `read_files` tool to understand its content based on the user's query (e.g., "What kind of flower is shown in the image I uploaded?").

4.  **Modularity and Reusability:** The `mime-reader-mcp-server` can be run as a standalone service. Multiple different agents or applications can then connect to it via MCP to leverage its file-reading capabilities without each needing to implement the Google AI interaction logic themselves.

By providing this functionality via MCP, `MimeFilesReader` becomes a powerful, reusable component in the growing ecosystem of AI agents and tools.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
