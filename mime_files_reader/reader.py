# mime_files_reader/reader.py

import logging
from pathlib import Path
from typing import List, Optional
import re # Added import
import time

# Ensure google.genai is imported correctly
try:
    from google import genai
    from google.genai import types
    from google.api_core import exceptions as google_exceptions
except ImportError:
    logging.error("Failed to import google.genai. Please ensure it's installed ('pip install google.genai')")
    raise SystemExit("google.genai package not found.")

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MimeFilesReader:
    """
    Reads and processes various MIME type files and YouTube URLs using Google's Generative AI.

    Uploads specified local files to Google's File API. Processes YouTube URLs directly.
    Sends the prepared media along with a question to a specified GenAI model,
    retrieves the streaming text-based response, and optionally saves the response
    to a file. Handles automatic cleanup of uploaded files.
    """
    # Added regex for YouTube URL detection
    YOUTUBE_URL_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/).+")

    def __init__(self, model_name: str, google_genai_key: str, working_dir: str = "."):
        """
        Initializes the MimeFilesReader.

        Args:
            model_name: The name of the Google GenAI model to use
                        (e.g., 'gemini-1.5-flash-latest').
            google_genai_key: Your Google GenAI API key.
            working_dir: The base directory for resolving relative file paths.
                         Defaults to the current working directory (".").
        """
        self.model_name = model_name
        self.working_dir = Path(working_dir).resolve() # Ensure working_dir is absolute
        # Reverted client initialization to original working version
        self.client = self._initialize_client(google_genai_key)
        logger.info(
            f"MimeFilesReader initialized with model '{self.model_name}' "
            f"and working directory '{self.working_dir}'"
        )

    # Reverted client initialization to original working version
    def _initialize_client(self, api_key: str) -> genai.Client:
        """Initializes the GenAI Client."""
        if not api_key:
             logger.error("Google GenAI API key is missing or empty.")
             raise ValueError("Missing Google GenAI API key.")
        try:
            # Use genai.Client as in the original code
            client = genai.Client(api_key=api_key)
            logger.info("Google GenAI client initialized successfully.")
            return client
        except google_exceptions.PermissionDenied:
            logger.error("Authentication failed. Please check your Google GenAI API key.")
            raise
        except google_exceptions.DefaultCredentialsError:
             logger.error("Could not automatically find Google Cloud credentials. Ensure your environment is set up correctly or provide an API key.")
             raise
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            raise

    # Added helper method to check for YouTube URLs
    def _is_youtube_url(self, text: str) -> bool:
        """Checks if the given text is likely a YouTube URL."""
        return bool(self.YOUTUBE_URL_PATTERN.match(text))

    def _resolve_path(self, file_path: str) -> Path:
        """Resolves a file path relative to the working directory if necessary,
           ensuring it exists and is a file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path
        try:
            if not path.exists():
                 raise FileNotFoundError(f"Input file not found: {path}")
            resolved_path = path.resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Input path exists but is not a file: {resolved_path}")
            return resolved_path # Return resolved path
        except (FileNotFoundError, IsADirectoryError) as e:
            logger.error(f"Path resolution error for '{file_path}': {e}")
            raise FileNotFoundError(str(e)) # Re-raise with original message
        except Exception as e:
            logger.error(f"Unexpected error resolving path '{file_path}': {e}")
            raise


    def _upload_file(self, file_path: Path) -> Optional['types.File']: # Use string literal for type hint if types isn't fully defined here
        """
        Uploads a single file using the client's file service and waits for it to become active.

        Args:
            file_path: The absolute, validated Path object of the file to upload.

        Returns:
            A google.genai.types.File object if upload is successful and the file becomes
            ACTIVE within the timeout, None otherwise (upload failure, processing failure, or timeout).
            Note: Even if None is returned due to processing failure/timeout, the file might
            still exist on the server and require cleanup if the initial upload succeeded.
            The caller should handle tracking for cleanup based on the initial upload attempt.
        """
        logger.info(f"Uploading file: {file_path}...")
        uploaded_file: Optional['types.File'] = None
        try:
            # 1. Initial Upload Call
            # Assuming self.client exists and is initialized
            uploaded_file = self.client.files.upload(file=str(file_path))
            logger.info(
                f"Successfully initiated upload for '{file_path.name}' as '{uploaded_file.name}' "
                f"({uploaded_file.mime_type}). Current state: {uploaded_file.state}. URI: {uploaded_file.uri}"
            )

            # 2. Basic validation of the initial response object
            if not uploaded_file.name:
                logger.error(f"Uploaded file object for {file_path.name} is missing 'name' attribute.")
                return None # Cannot track or poll without a name
            # URI and mime_type are crucial for later use, check them too.
            if not uploaded_file.uri:
                logger.error(f"Uploaded file object for {file_path.name} ({uploaded_file.name}) is missing 'uri' attribute.")
                # Return None, caller should handle cleanup if needed
                return None
            if not uploaded_file.mime_type:
                logger.error(f"Uploaded file object for {file_path.name} ({uploaded_file.name}) is missing 'mime_type'. Cannot use file.")
                # Return None, caller should handle cleanup if needed
                return None

            # 3. Wait for the file to become ACTIVE
            # Use class attributes for configuration
            timeout = 300
            interval = 5
            file_name = uploaded_file.name # Store name for polling

            # Check if already active
            if uploaded_file.state == types.FileState.ACTIVE:
                logger.info(f"File '{file_name}' was already ACTIVE upon upload.")
                return uploaded_file # Already good to go

            logger.info(f"Waiting up to {timeout}s for file '{file_name}' to become active...")
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Get the current status using the file name
                    current_file_info = self.client.files.get(name=file_name)
                    state = current_file_info.state
                    logger.debug(f"Polling file '{file_name}': State is {state}")

                    if state == types.FileState.ACTIVE:
                        logger.info(f"File '{file_name}' is now ACTIVE.")
                        # Return the latest info which confirms ACTIVE state and has necessary details
                        if not current_file_info.uri or not current_file_info.mime_type:
                            logger.error(f"File '{file_name}' became ACTIVE but fetched info is missing URI/mime_type. Cannot use.")
                            return None
                        return current_file_info

                    elif state == types.File.State.FAILED:
                        logger.error(f"File '{file_name}' processing failed (State: FAILED).")
                        return None # Indicate failure

                    # If state is PROCESSING or unspecified/other non-terminal, wait and continue
                    elif state == types.File.State.PROCESSING:
                        pass # Continue loop, wait before next poll
                    else:
                        logger.warning(f"File '{file_name}' is in an unexpected state: {state}. Continuing to poll.")

                    time.sleep(interval) # Wait before the next poll

                except google_exceptions.NotFound:
                    logger.error(f"File '{file_name}' not found while polling status. Assuming failure.")
                    return None # File disappeared or invalid name somehow
                except google_exceptions.GoogleAPIError as e:
                    # Log API errors during polling but continue polling (might be transient)
                    logger.warning(f"API error while checking status for file '{file_name}': {e}. Retrying after {interval}s...")
                    time.sleep(interval) # Wait before retrying the poll
                except Exception as e:
                    # Catch unexpected errors during polling
                    logger.exception(f"Unexpected error checking status for file '{file_name}': {e}")
                    return None # Safer to assume failure

            # 4. Timeout reached if loop finishes without returning
            logger.warning(f"File '{file_name}' did not become active within {timeout} seconds.")
            return None # Indicate failure due to timeout

        # Handle exceptions during the initial upload call itself
        except TypeError as e:
            logger.error(f"TypeError during file upload call for {file_path}. Check API arguments: {e}")
            return None
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"API error during initial file upload for {file_path}: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during the upload process itself
            logger.exception(f"Unexpected error during upload process for file {file_path}: {e}")
            return None

    def _delete_file(self, file_name: str):
        """Deletes a file previously uploaded via the client's file service."""
        # Uses self.client as initialized in original code
        if not file_name:
            logger.warning("Attempted to delete a file with no name.")
            return
        logger.info(f"Deleting uploaded file reference: {file_name}...")
        try:
            self.client.files.delete(name=file_name)
            logger.info(f"Successfully deleted file reference: {file_name}")
        except google_exceptions.NotFound:
             logger.warning(f"File reference {file_name} not found during cleanup (might have been deleted already).")
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"API error deleting file reference {file_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to delete file reference {file_name}: {e}")

    def read(self, question: str, inputs: List[str], output: Optional[str] = None, auto_cleanup: bool = True) -> str:
        """
        Processes local files and YouTube URLs, asks a question to the GenAI model,
        and gets the response.

        Args:
            question: The question to ask the model about the inputs.
            inputs: A list of strings, where each string is either a local file path
                    (relative or absolute) or a YouTube URL.
            output: Optional path to save the response text. If None, the response
                    is returned directly. Relative paths are resolved against the
                    working directory.
            auto_cleanup: If True, automatically deletes uploaded file references from
                          GenAI's File API after processing. YouTube links are ignored
                          for cleanup.

        Returns:
            The text response from the model, or a confirmation message if 'output'
            is specified and writing succeeds.

        Raises:
            FileNotFoundError: If any input file path does not exist or is not a file.
            ValueError: If no inputs could be successfully prepared, or if API key is missing/invalid.
            google_exceptions.GoogleAPIError: For errors during API interaction.
            IOError: For errors writing the output file.
            Exception: For other unexpected errors during processing.
        """
        if not self.client:
            raise ValueError("GenAI Client not initialized. Check API key and initialization.")

        uploaded_file_objects: List[types.File] = [] # Track only files uploaded via API
        media_parts: List[types.Part] = []          # Parts for API call (files + youtube)
        prepared_items_count = 0
        resolved_output_path: Optional[Path] = None

        # Resolve output path early and create directories
        if output:
            try:
                output_p = Path(output)
                if not output_p.is_absolute():
                    output_p = self.working_dir / output_p
                output_p.parent.mkdir(parents=True, exist_ok=True)
                resolved_output_path = output_p.resolve()
                logger.info(f"Output will be saved to: {resolved_output_path}")
            except OSError as e:
                 logger.error(f"Cannot create output directory for '{output}': {e}")
                 raise IOError(f"Cannot create output directory for '{output}': {e}")
            except Exception as e:
                logger.error(f"Invalid output path '{output}': {e}")
                raise IOError(f"Invalid output path '{output}': {e}")

        try:
            # 1. Process inputs (files and YouTube URLs)
            for item_str in inputs:
                # Check if input is a YouTube URL
                if self._is_youtube_url(item_str):
                    logger.info(f"Processing YouTube URL: {item_str}")
                    try:
                        # Directly create a Part for the YouTube URL
                        # Use a generic video mime type, API handles specifics
                        youtube_part = types.Part.from_uri(
                            mime_type="video/*",
                            file_uri=item_str
                        )
                        media_parts.append(youtube_part)
                        prepared_items_count += 1
                        logger.info(f"Prepared YouTube URL '{item_str}' for model.")
                    except Exception as e:
                         # Log error but continue processing other inputs
                         logger.error(f"Failed to create Part for YouTube URL '{item_str}': {e}")

                else: # Assume it's a file path
                    logger.info(f"Processing potential file path: {item_str}")
                    file_path: Optional[Path] = None
                    try:
                        file_path = self._resolve_path(item_str)
                        uploaded_file = self._upload_file(file_path)

                        # Check if upload was successful and returned required info
                        if uploaded_file and uploaded_file.uri and uploaded_file.mime_type:
                            # Keep track of successfully uploaded files for cleanup
                            uploaded_file_objects.append(uploaded_file)
                            # Create Part using the uploaded file's URI and detected MIME type
                            file_part = types.Part.from_uri(
                                mime_type=uploaded_file.mime_type,
                                file_uri=uploaded_file.uri
                            )
                            media_parts.append(file_part)
                            prepared_items_count += 1
                            logger.info(f"Prepared file '{file_path.name}' ({uploaded_file.name}) for model.")
                        # Else: Upload failed or returned incomplete object. Error logged in _upload_file.
                        elif file_path: # Log skip only if path resolution succeeded
                            logger.warning(f"Skipping file '{file_path.name}' due to upload failure or missing attributes.")

                    except FileNotFoundError as e:
                         logger.error(f"Input file error for '{item_str}': {e}")
                         raise # Re-raise critical file errors
                    except Exception as e:
                        # Log exception with stack trace for unexpected errors during file processing
                        logger.exception(f"Unexpected error processing file input '{item_str}': {e}")
                        raise # Re-raise unexpected errors

            # --- End Input Processing Loop ---

            if not media_parts:
                if inputs:
                    raise ValueError(
                        "No inputs (files or URLs) were successfully prepared for processing (check logs)."
                     )
                else:
                    raise ValueError("No input files or URLs were specified.")

            logger.info(f"Successfully prepared {prepared_items_count} item(s) for the model.")

            # 2. Construct content for API call using the original structure
            prompt_parts = media_parts + [types.Part.from_text(text=question)]
            # Contents must be a list containing Content objects
            contents = [types.Content(role="user", parts=prompt_parts)]
            # Use original generation config for streaming text response
            generation_config = types.GenerateContentConfig(response_mime_type="text/plain")

            logger.info(f"Sending request with {len(media_parts)} media part(s) to model '{self.model_name}'...")

            # 3. Call GenAI API (Streaming - reverted to original method)
            response = self.client.models.generate_content(
                model=f"models/{self.model_name}",
                contents=contents,
                config=generation_config,
            )

            full_response = response.text

            if not full_response:
                 logger.warning("Received an empty or non-text response from the model.")
                 # You might want to check response_stream.prompt_feedback here if available
            else:
                logger.info("Received response from GenAI model.")

            # 5. Handle output
            if resolved_output_path:
                try:
                    with open(resolved_output_path, "w", encoding="utf-8") as f:
                        f.write(full_response)
                    logger.info(f"Response successfully written to: {resolved_output_path}")
                    return (
                        f"Inputs successfully processed. Results written to: {resolved_output_path}"
                    )
                except IOError as e:
                    logger.error(f"Failed to write output file {resolved_output_path}: {e}")
                    # Return error message but include the response text
                    return f"Error writing to file {resolved_output_path}. Response: {full_response}"
            else:
                # Return the response text directly if no output file specified
                return full_response

        except (FileNotFoundError, ValueError, google_exceptions.GoogleAPIError, IOError) as e:
            # Log expected errors and re-raise
            logger.error(f"An error occurred during processing: {e}")
            raise
        except Exception as e:
            # Log unexpected errors with traceback and re-raise
            logger.exception(f"An unexpected error occurred during processing: {e}")
            raise

        finally:
            # 6. Cleanup uploaded files if requested
            # This now correctly only targets files that were actually uploaded via the API
            if auto_cleanup and uploaded_file_objects:
                logger.info(f"Starting automatic cleanup of {len(uploaded_file_objects)} uploaded file reference(s)...")
                deleted_count = 0
                for uploaded_file in uploaded_file_objects:
                    # Check if the object and its name attribute exist before trying to delete
                    if uploaded_file and uploaded_file.name:
                        self._delete_file(uploaded_file.name)
                        deleted_count += 1
                    elif uploaded_file:
                         logger.warning("Found an uploaded file object without a 'name' in the cleanup list, skipping deletion.")
                logger.info(f"Automatic cleanup finished. Attempted to delete {deleted_count} file reference(s).")
            elif auto_cleanup:
                 logger.info("Automatic cleanup enabled, but no files were tracked for cleanup.")
