# mime_files_reader/reader.py

import logging
from pathlib import Path
from typing import List, Optional

# Ensure google.genai is imported correctly
try:
    from google import genai
    from google.genai import types
    from google.api_core import exceptions as google_exceptions
except ImportError:
    logging.error("Failed to import google.generativeai. Please ensure it's installed ('pip install google-generativeai')")
    raise SystemExit("google-generativeai package not found.")

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MimeFilesReader:
    """
    Reads and processes various MIME type files using Google's Generative AI.

    Uploads specified files to Google's File API, assuming they are ready for use
    immediately after successful upload. Sends them along with a question
    to a specified GenAI model, retrieves the streaming text-based response,
    and optionally saves the response to a file. Handles automatic cleanup.
    """

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
        self.client = self._initialize_client(google_genai_key)
        logger.info(
            f"MimeFilesReader initialized with model '{self.model_name}' "
            f"and working directory '{self.working_dir}'"
        )

    def _initialize_client(self, api_key: str) -> genai.Client:
        """Initializes the GenAI Client."""
        if not api_key:
             logger.error("Google GenAI API key is missing or empty.")
             raise ValueError("Missing Google GenAI API key.")
        try:
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
        except (FileNotFoundError, IsADirectoryError) as e:
            logger.error(f"Path resolution error for '{file_path}': {e}")
            raise FileNotFoundError(e)
        except Exception as e:
            logger.error(f"Unexpected error resolving path '{file_path}': {e}")
            raise

        return resolved_path

    def _upload_file(self, file_path: Path) -> Optional[types.File]:
        """
        Uploads a single file using the client's file service.

        Args:
            file_path: The absolute, validated Path object of the file to upload.

        Returns:
            A google.genai.types.File object if successful, None otherwise.
        """
        logger.info(f"Uploading file: {file_path}...")
        try:
            uploaded_file = self.client.files.upload(file=str(file_path))
            # Log success based on the API call returning, assuming ready.
            logger.info(
                f"Successfully uploaded '{file_path.name}' as '{uploaded_file.name}' "
                f"({uploaded_file.mime_type}). URI: {uploaded_file.uri}"
            )
            # Perform basic checks on the returned object
            if not uploaded_file.name:
                 logger.error(f"Uploaded file object for {file_path.name} is missing 'name' attribute.")
            if not uploaded_file.uri:
                 logger.error(f"Uploaded file object for {file_path.name} ({uploaded_file.name}) is missing 'uri' attribute.")
            if not uploaded_file.mime_type:
                 logger.warning(f"Uploaded file object for {file_path.name} ({uploaded_file.name}) is missing 'mime_type' attribute.")
                 # Allow proceeding but log warning
            return uploaded_file
        except TypeError as e:
             logger.error(f"TypeError during file upload for {file_path}. Check API arguments: {e}")
             return None
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            return None

    def _delete_file(self, file_name: str):
        """Deletes a file previously uploaded via the client's file service."""
        if not file_name:
            logger.warning("Attempted to delete a file with no name.")
            return
        logger.info(f"Deleting uploaded file: {file_name}...")
        try:
            self.client.files.delete(name=file_name)
            logger.info(f"Successfully deleted file: {file_name}")
        except google_exceptions.NotFound:
             logger.warning(f"File {file_name} not found during cleanup (might have been deleted already).")
        except Exception as e:
            logger.error(f"Failed to delete file {file_name}: {e}")

    def read(self, question: str, files: List[str], output: Optional[str] = None, auto_cleanup: bool = True) -> str:
        """
        Uploads files, asks a question to the GenAI model, and gets the response.

        Args:
            question: The question to ask the model about the files.
            files: A list of file paths (relative or absolute) to process.
            output: Optional path to save the response text. If None, the response
                    is returned directly. Relative paths are resolved against the
                    working directory.
            auto_cleanup: If True, automatically deletes uploaded files from
                          GenAI's File API after processing.

        Returns:
            The text response from the model, or a confirmation message if 'output'
            is specified and writing succeeds.

        Raises:
            FileNotFoundError: If any input file path does not exist or is not a file.
            ValueError: If no files could be successfully uploaded with necessary attributes, or if API key is missing.
            google_exceptions.GoogleAPIError: For errors during API interaction.
            IOError: For errors writing the output file.
            Exception: For other unexpected errors during processing.
        """
        if not self.client:
            raise ValueError("GenAI Client not initialized. Check API key and initialization.")

        uploaded_file_objects: List[types.File] = []
        file_parts: List[types.Part] = []
        prepared_files_count = 0
        resolved_output_path: Optional[Path] = None

        # Resolve output path early and create directories
        if output:
            try:
                output_p = Path(output)
                if not output_p.is_absolute():
                    output_p = self.working_dir / output_p
                output_p.parent.mkdir(parents=True, exist_ok=True)
                resolved_output_path = output_p.resolve()
            except OSError as e:
                 logger.error(f"Cannot create output directory for '{output}': {e}")
                 raise IOError(f"Cannot create output directory for '{output}': {e}")
            except Exception as e:
                logger.error(f"Invalid output path '{output}': {e}")
                raise IOError(f"Invalid output path '{output}': {e}")

        try:
            # 1. Resolve paths and upload files
            for file_path_str in files:
                file_path: Optional[Path] = None
                try:
                    file_path = self._resolve_path(file_path_str)
                    uploaded_file = self._upload_file(file_path)

                    if uploaded_file:
                        uploaded_file_objects.append(uploaded_file)
                        if uploaded_file.mime_type and uploaded_file.uri:
                            file_parts.append(types.Part.from_uri(
                                mime_type=uploaded_file.mime_type,
                                file_uri=uploaded_file.uri
                            ))
                            prepared_files_count += 1
                        else:
                            logger.error(f"Skipping file '{file_path.name}' ({uploaded_file.name}) due to missing URI or MIME type after upload.")
                    else:
                        logger.warning(f"Skipping file '{file_path.name}' due to upload failure.")

                except FileNotFoundError as e:
                     logger.error(f"Input file error for '{file_path_str}': {e}")
                     raise
                except Exception as e:
                    logger.exception(f"Unexpected error processing file '{file_path_str}': {e}")
                    raise


            if not file_parts:
                if files:
                    raise ValueError(
                        "No files were successfully prepared for processing (check upload logs)."
                     )
                else:
                    raise ValueError("No input files were specified.")

            logger.info(f"Successfully prepared {prepared_files_count} file(s) for the model.")

            # --- FIX IS HERE ---
            # 2. Construct content for API call using text= keyword argument
            prompt_parts = file_parts + [types.Part.from_text(text=question)]
            # --- END FIX ---
            contents = [types.Content(role="user", parts=prompt_parts)]
            generation_config = types.GenerateContentConfig(response_mime_type="text/plain")

            logger.info(f"Sending request with {len(file_parts)} file(s) to model '{self.model_name}'...")

            # 3. Call GenAI API (Streaming)
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generation_config,
            )

            # 4. Accumulate streamed response
            full_response_chunks = []
            for chunk in response_stream:
                try:
                    if chunk.candidates and chunk.candidates[0].content.parts:
                         text_content = chunk.candidates[0].content.parts[0].text
                         if text_content:
                             full_response_chunks.append(text_content)
                    elif hasattr(chunk, 'text') and chunk.text:
                        full_response_chunks.append(chunk.text)
                except (AttributeError, IndexError):
                    pass

            full_response = "".join(full_response_chunks)

            if not full_response:
                 logger.warning("Received an empty or non-text response from the model.")
            else:
                logger.info("Received response from GenAI model.")

            # 5. Handle output
            if resolved_output_path:
                try:
                    with open(resolved_output_path, "w", encoding="utf-8") as f:
                        f.write(full_response)
                    logger.info(f"Response successfully written to: {resolved_output_path}")
                    return (
                        f"Mime files successfully processed. Results written to: {resolved_output_path}"
                    )
                except IOError as e:
                    logger.error(f"Failed to write output file {resolved_output_path}: {e}")
                    return f"Error writing to file {resolved_output_path}. Response: {full_response}"
            else:
                return full_response

        except (FileNotFoundError, ValueError, google_exceptions.GoogleAPIError, IOError) as e:
            logger.error(f"An error occurred during processing: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during processing: {e}")
            raise

        finally:
            # 6. Cleanup uploaded files if requested
            if auto_cleanup and uploaded_file_objects:
                logger.info(f"Starting automatic cleanup of {len(uploaded_file_objects)} uploaded file reference(s)...")
                deleted_count = 0
                for uploaded_file in uploaded_file_objects:
                    if uploaded_file and uploaded_file.name:
                        self._delete_file(uploaded_file.name)
                        deleted_count += 1
                logger.info(f"Automatic cleanup finished. Attempted to delete {deleted_count} file(s).")
