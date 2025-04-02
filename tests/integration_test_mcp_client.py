# tests/integration_test_mcp_client.py
# Run with `python tests/integration_test_mcp_client.py`

import asyncio
import os
import sys
import shutil # Import shutil
import logging
from pathlib import Path
# Use pytest if available for better assertions, otherwise fallback
try:
    import pytest
except ImportError:
    pytest = None # Define pytest as None if not installed

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
# Import MCP types to understand the response structure better
from mcp import types as mcp_types

# Configure basic logging for the client
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - MCP Client - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration for the client and finding the server command ---

# Find the installed server command using shutil.which
SERVER_EXECUTABLE = shutil.which("mime-reader-mcp-server")
if not SERVER_EXECUTABLE:
    logger.error(
        "Could not find 'mime-reader-mcp-server' command in PATH. "
        "Ensure the package is installed correctly (e.g., 'pip install -e .')."
    )
    sys.exit(1) # Exit if command not found after installation attempt
else:
    SERVER_COMMAND = SERVER_EXECUTABLE # Use the found executable path
    SERVER_ARGS = []                  # No extra args needed when using the entry point
    logger.info(f"Found server executable at: {SERVER_COMMAND}")


# Get API Key and Model Name for the *server's* environment
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

if not API_KEY:
    logger.error("Client needs GEMINI_API_KEY in environment to pass to the server process.")
    sys.exit(1)

# --- Find Test Files ---
try:
    TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"
    # Use absolute paths for the files to avoid confusion about CWD
    TEST_IMAGE_PATH_1 = str((TEST_DATA_DIR / "test_image.png").resolve(strict=True))
    TEST_IMAGE_PATH_2 = str((TEST_DATA_DIR / "test_image_2.png").resolve(strict=True))
    logger.info(f"Using test file 1: {TEST_IMAGE_PATH_1}")
    logger.info(f"Using test file 2: {TEST_IMAGE_PATH_2}")
except FileNotFoundError as e:
     logger.error(f"Could not find test data files in {TEST_DATA_DIR}: {e}")
     sys.exit(1)
except Exception as e:
    logger.error(f"Error resolving test file paths: {e}")
    sys.exit(1)

# --- Helper function for failing tests ---
def fail_test(message):
    """Fail the test using pytest if available, otherwise raise Exception."""
    logger.error(f"TEST FAILED: {message}")
    if pytest:
        pytest.fail(message)
    else:
        # Raising AssertionError might be caught by the main loop's generic except
        # It's better to re-raise a specific exception or use sys.exit here if not using pytest
        raise AssertionError(message)


async def main():
    logger.info("Starting MCP client test...")

    # Define filename for cleanup *before* try block
    output_filename = "test_mcp_output.txt"
    output_file_to_remove: Path | None = None # Will hold the resolved path if needed

    server_params = StdioServerParameters(
        command=SERVER_COMMAND, # Use the executable path found by shutil.which
        args=SERVER_ARGS,       # Use the empty args list for entry point
        # Pass necessary environment variables to the server process
        env={
            **os.environ, # Inherit current env (important for PATH, etc.)
            "GEMINI_API_KEY": API_KEY,
            "GEMINI_MODEL_NAME": MODEL_NAME,
            # Ensure PYTHONUNBUFFERED is set so server logs appear immediately
            "PYTHONUNBUFFERED": "1"
        }
    )

    # Define expected text fragments (case-insensitive) for assertions
    EXPECTED_TEXT_IMG1 = "mcp server"
    EXPECTED_TEXT_IMG1_ALT = "building from scratch"
    EXPECTED_TEXT_IMG2 = "what will you build"


    test_passed = True # Flag to track overall success

    try:
        async with asyncio.timeout(90): # Increased timeout for potentially slow AI calls
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("Initializing session...")
                    await session.initialize()
                    logger.info("Session initialized.")

                    # 1. List tools
                    logger.info("Listing tools...")
                    # The list_tools method returns a ListToolsResponse object
                    tools_response: mcp_types.ListToolsResponse = await session.list_tools()
                    logger.info(f"Available tools response object: {tools_response}")

                    # Verify the tool exists in the response object's 'tools' list
                    tool_found = False
                    # Check if the response object has a 'tools' attribute which is a list
                    if hasattr(tools_response, 'tools') and isinstance(tools_response.tools, list):
                        if any(t.name == 'read_files' for t in tools_response.tools):
                            tool_found = True

                    if not tool_found:
                        fail_test("Server did not list the 'read_files' tool correctly in the response object!")
                        return # Stop the test if basic functionality fails
                    else:
                        logger.info("Verified 'read_files' tool is listed.")

                    # 2. Call the tool (example 1: single image)
                    logger.info("--- Test Case 1: Read Image 1 ---")
                    tool_args_1 = {
                        "question": "What text is in this image?",
                        "files": [TEST_IMAGE_PATH_1]
                    }
                    try:
                        # call_tool returns a CallToolResponse object
                        result1_response: mcp_types.CallToolResponse = await session.call_tool("read_files", tool_args_1)
                        logger.info(f"Raw response object for image 1: {result1_response}")

                        # Extract text from the 'content' attribute of the response object
                        result1_text = ""
                        # Check if response has 'content' which is a list, and list is not empty
                        if hasattr(result1_response, 'content') and isinstance(result1_response.content, list) and result1_response.content:
                            content_part = result1_response.content[0]
                            # Check if the first part is TextContent and has text
                            if isinstance(content_part, mcp_types.TextContent) and hasattr(content_part, 'text'):
                                result1_text = content_part.text
                                logger.info(f"Extracted text for image 1: '{result1_text[:200]}...'")
                                text_content_lower = result1_text.lower()
                                assert EXPECTED_TEXT_IMG1 in text_content_lower or EXPECTED_TEXT_IMG1_ALT in text_content_lower, \
                                    f"Result 1 did not contain expected text ('{EXPECTED_TEXT_IMG1}' or '{EXPECTED_TEXT_IMG1_ALT}')"
                                logger.info("Result 1 text verification PASSED.")
                            else:
                                fail_test(f"First content part format unexpected for image 1: {content_part}")
                        else:
                             fail_test(f"Unexpected result format or empty content for image 1: {result1_response}")

                    except Exception as e:
                        logger.exception(f"Error calling tool for image 1: {e}")
                        fail_test(f"Error calling tool for image 1: {e}")


                    # 3. Call the tool (example 2: second image, save output)
                    logger.info("--- Test Case 2: Read Image 2 (Save Output) ---")
                    # Calculate the absolute path for cleanup *before* the call
                    # Assumes server runs where client runs - might need adjustment
                    output_file_to_remove = Path.cwd() / output_filename
                    logger.info(f"Calling tool 'read_files' for image 2 (output to {output_filename} in server CWD)...")
                    tool_args_2 = {
                        "question": "Describe this image briefly.",
                        "files": [TEST_IMAGE_PATH_2],
                        "output": output_filename # Server needs write access here
                    }
                    try:
                        result2_response: mcp_types.CallToolResponse = await session.call_tool("read_files", tool_args_2)
                        logger.info(f"Raw response object for image 2: {result2_response}")

                        # Extract confirmation message from the 'content' attribute
                        result2_text = ""
                        if hasattr(result2_response, 'content') and isinstance(result2_response.content, list) and result2_response.content:
                             content_part = result2_response.content[0]
                             if isinstance(content_part, mcp_types.TextContent) and hasattr(content_part, 'text'):
                                 result2_text = content_part.text
                                 logger.info(f"Extracted text for image 2: {result2_text}")
                                 assert "successfully processed" in result2_text.lower() and output_filename in result2_text, \
                                     "Result 2 did not seem to be the expected confirmation message."
                                 logger.info("Result 2 confirmation message verification PASSED.")
                             else:
                                fail_test(f"First content part format unexpected for image 2: {content_part}")
                        else:
                            fail_test(f"Unexpected result format or empty content for image 2: {result2_response}")

                    except Exception as e:
                        logger.exception(f"Error calling tool for image 2: {e}")
                        fail_test(f"Error calling tool for image 2: {e}")


                    # 4. Call the tool (example 3: file not found)
                    logger.info("--- Test Case 3: Non-Existent File ---")
                    non_existent_file = "/path/that/surely/does/not/exist/on/server/file.xyz"
                    logger.info(f"Calling tool 'read_files' with non-existent file: {non_existent_file}")
                    tool_args_3 = {
                        "question": "What about this file?",
                        "files": [non_existent_file]
                    }
                    try:
                        result3_response: mcp_types.CallToolResponse = await session.call_tool("read_files", tool_args_3)
                        logger.info(f"Raw response object for non-existent file: {result3_response}")

                        # Extract error message from the 'content' attribute
                        result3_text = ""
                        if hasattr(result3_response, 'content') and isinstance(result3_response.content, list) and result3_response.content:
                             content_part = result3_response.content[0]
                             if isinstance(content_part, mcp_types.TextContent) and hasattr(content_part, 'text'):
                                 result3_text = content_part.text
                                 logger.info(f"Extracted text for non-existent file: {result3_text}")
                                 # Check if 'isError' flag is set on the response object (if available/used by server)
                                 is_error_flag = getattr(result3_response, 'isError', False) # Default to False if not present
                                 # Assert based on text content OR error flag
                                 assert ("error" in result3_text.lower() and ("file not found" in result3_text.lower() or "input file error" in result3_text.lower())) or is_error_flag, \
                                     f"Result 3 did not seem to be the expected file not found error message (isError={is_error_flag}): '{result3_text}'"
                                 logger.info("Result 3 error message verification PASSED.")
                             else:
                                fail_test(f"First content part format unexpected for non-existent file: {content_part}")
                        else:
                            fail_test(f"Unexpected result format or empty content for non-existent file: {result3_response}")

                    except Exception as e:
                        # It's possible the session itself might error depending on server handling
                        logger.exception(f"Error calling tool for non-existent file: {e}")
                        # Depending on expected behavior, this might be okay or a failure
                        # Let's consider an exception here a failure for now
                        fail_test(f"Error calling tool for non-existent file: {e}")


    except TimeoutError:
         logger.error("Client operation timed out.")
         test_passed = False # Mark test as failed
         # No need to call fail_test here, exception will propagate to __main__ block
         raise # Re-raise timeout error
    except Exception as e:
        logger.exception(f"An unexpected error occurred in the client test setup or execution: {e}")
        test_passed = False # Mark test as failed
        raise # Re-raise other exceptions

    finally:
        logger.info("MCP client test finished.")
        # Cleanup the output file if its path was set
        if output_file_to_remove:
            try:
                if output_file_to_remove.exists():
                    output_file_to_remove.unlink()
                    logger.info(f"Cleaned up output file: {output_file_to_remove}")
                else:
                    # This is expected if the server ran in a different CWD or failed before creating the file
                    logger.info(f"Output file for cleanup not found (may not have been created or in different CWD): {output_file_to_remove}")
            except Exception as cleanup_err:
                logger.warning(f"Could not cleanup output file '{output_file_to_remove}': {cleanup_err}")
        else:
             logger.info("No output file path was set for cleanup (as expected if test failed early).")

    return test_passed # Return True if no exceptions caused failure


if __name__ == "__main__":
    # Ensure API key is available before running
    if not API_KEY:
        print("\nError: GEMINI_API_KEY environment variable must be set to run the test client.", file=sys.stderr)
        sys.exit(1) # Exit with error code

    success = False
    exit_code = 1 # Default to failure
    try:
        success = asyncio.run(main())
        if success:
            print("\n--- Client Test Completed Successfully ---")
            exit_code = 0 # Set success code
        else:
            # Failures should have been logged by fail_test
             print("\n--- Client Test FAILED (Check logs for assertion details) ---", file=sys.stderr)
             # Keep exit_code = 1

    except KeyboardInterrupt:
        logger.info("Client interrupted by user.")
        print("\n--- Client Test Interrupted ---", file=sys.stderr)
    except Exception as main_err:
        # Catch failures from fail_test raising AssertionError or other sync exceptions
        print(f"\n--- Client Test FAILED (Unhandled Exception): {main_err} ---", file=sys.stderr)
        # Keep exit_code = 1

    sys.exit(exit_code) # Exit with appropriate code