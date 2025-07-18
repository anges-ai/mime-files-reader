# mime_files_reader/reader.py

import logging
import os
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

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
    def __init__(self, model_name: str, google_genai_key: Optional[str] = None, 
                 working_dir: str = ".", project: Optional[str] = None):
        """
        Initializes the MimeFilesReader.

        Args:
            model_name: The name of the Google GenAI model to use
                        (e.g., 'gemini-1.5-flash-latest').
            google_genai_key: Your Google GenAI API key. Optional if using Vertex AI
                            (when project parameter or VERTEX_GCP_PROJECT environment variable is set).
            working_dir: The base directory for resolving relative file paths.
                         Defaults to the current working directory (".").
            project: The Google Cloud project ID for Vertex AI. If provided, takes precedence
                    over VERTEX_GCP_PROJECT environment variable.

        Raises:
            ValueError: If neither google_genai_key nor project (or VERTEX_GCP_PROJECT env var) is provided.
        """
        self.model_name = model_name
        self.google_genai_key = google_genai_key
        self.project = project
        self.working_dir = Path(working_dir).resolve()  # Ensure working_dir is absolute
        self.client = self._initialize_client()
        logger.info(
            f"MimeFilesReader initialized with model '{self.model_name}' "
            f"and working directory '{self.working_dir}'"
        )

    def _initialize_client(self) -> genai.Client:
        """Initializes the GenAI Client using either Google API key or Vertex AI."""
        # Priority: passed project parameter > VERTEX_GCP_PROJECT env var > Google API key
        vertex_project = self.project or os.environ.get('VERTEX_GCP_PROJECT')
        
        if vertex_project:
            logger.info(f"Initializing Vertex AI client with project: {vertex_project}")
            try:
                return genai.Client(
                    vertexai=True,
                    project=vertex_project,
                    location="global"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI client: {e}")
                raise
        
        # Fallback to Google API key authentication
        if not self.google_genai_key:
            raise ValueError(
                "Missing Google GenAI API key. Please provide either 'google_genai_key' parameter "
                "or set 'project' parameter / 'VERTEX_GCP_PROJECT' environment variable for Vertex AI."
            )
        
        logger.info("Initializing Google GenAI client with API key")
        try:
            return genai.Client(api_key=self.google_genai_key)
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

        return resolved_path

    def _is_youtube_url(self, input_str: str) -> bool:
        """
        Check if the input string is a YouTube URL.
        
        Supports various YouTube URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        
        Args:
            input_str: The string to check
            
        Returns:
            bool: True if the string is a YouTube URL, False otherwise
        """
        if not input_str or not input_str.strip():
            return False
            
        youtube_patterns = [
            r'https?://(www\.)?youtube\.com/watch\?.*v=',
            r'https?://youtu\.be/',
            r'https?://(www\.)?youtube\.com/embed/'
        ]
        
        for pattern in youtube_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        return False
    
    def _process_youtube_url(self, url: str) -> types.Part:
        """
        Process YouTube URL directly with Gemini.
        
        Args:
            url: The YouTube URL to process
            
        Returns:
            A types.Part object for the YouTube video
            
        Raises:
            ValueError: If the URL is malformed or invalid
            Exception: For other processing errors
        """
        logger.info(f"Processing YouTube URL: {url}")
        
        # Validate URL format
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                error_msg = f"Invalid URL format: missing scheme or domain in '{url}'"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if parsed_url.scheme not in ['http', 'https']:
                error_msg = f"Invalid URL scheme '{parsed_url.scheme}': only HTTP and HTTPS are supported"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise ValueError as-is
            error_msg = f"Failed to parse URL '{url}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Additional YouTube-specific URL validation
        if not self._is_youtube_url(url):
            error_msg = f"URL does not appear to be a valid YouTube URL: '{url}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Process the YouTube URL
        try:
            logger.debug(f"Creating types.Part for YouTube URL: {url}")
            part = types.Part.from_uri(file_uri=url, mime_type="video/mp4")
            logger.info(f"Successfully processed YouTube URL: {url}")
            return part
            
        except Exception as e:
            error_msg = f"Failed to create types.Part for YouTube URL '{url}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _process_input(self, input_item: str) -> types.Part:
        """
        Process either local file or YouTube URL.
        
        Args:
            input_item: Either a local file path or YouTube URL
            
        Returns:
            A types.Part object for the input
        """
        if self._is_youtube_url(input_item):
            return self._process_youtube_url(input_item)
        else:
            # Process as local file
            file_path = self._resolve_path(input_item)
            uploaded_file = self._upload_file(file_path)
            
            if not uploaded_file:
                raise ValueError(f"Failed to upload file: {input_item}")
                
            if not (uploaded_file.mime_type and uploaded_file.uri):
                raise ValueError(f"Uploaded file missing required attributes: {input_item}")
                
            return types.Part.from_uri(
                mime_type=uploaded_file.mime_type,
                file_uri=uploaded_file.uri
            )

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
        Supports both local files and YouTube URLs.

        Args:
            question: The question to ask the model about the files.
            files: A list of file paths (relative or absolute) or YouTube URLs to process.
                  Supported YouTube URL formats: youtube.com/watch, youtu.be, youtube.com/embed
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
            # 1. Process files (local files or YouTube URLs)
            for input_str in files:
                try:
                    if self._is_youtube_url(input_str):
                        file_part = self._process_youtube_url(input_str)
                        file_parts.append(file_part)
                        prepared_files_count += 1
                    else:
                        # Process as local file
                        file_path = self._resolve_path(input_str)
                        uploaded_file = self._upload_file(file_path)
                        
                        if not uploaded_file:
                            continue  # Skip this file but don't fail immediately
                            
                        if not (uploaded_file.mime_type and uploaded_file.uri):
                            logger.warning(f"Uploaded file missing required attributes: {input_str}")
                            if uploaded_file.name:  # Still track for cleanup
                                uploaded_file_objects.append(uploaded_file)
                            continue  # Skip this file but don't fail immediately
                            
                        file_part = types.Part.from_uri(
                            mime_type=uploaded_file.mime_type,
                            file_uri=uploaded_file.uri
                        )
                        file_parts.append(file_part)
                        uploaded_file_objects.append(uploaded_file)
                        prepared_files_count += 1
                            
                except FileNotFoundError as e:
                     logger.error(f"Input file error for '{input_str}': {e}")
                     raise
                except Exception as e:
                    logger.exception(f"Unexpected error processing input '{input_str}': {e}")
                    raise

            if not file_parts:
                if files:
                    raise ValueError(
                        "No files were successfully prepared for processing (check upload logs)."
                     )
                else:
                    raise ValueError("No input files were specified.")

            logger.info(f"Successfully prepared {prepared_files_count} file(s) for the model.")

            # 2. Construct content for API call using text= keyword argument
            prompt_parts = file_parts + [types.Part.from_text(text=question)]
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
