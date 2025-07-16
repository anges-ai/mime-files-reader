# tests/test_reader.py

import os
import pytest
import time
from unittest.mock import patch, MagicMock, ANY, call
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for integration tests
load_dotenv()

# Import the class/types AFTER potentially loading .env
from google.api_core import exceptions as google_exceptions
from google.genai import types as google_types
from mime_files_reader.reader import MimeFilesReader

# --- Constants ---
TEST_WORKING_DIR_NAME = "test_run_dir"
TEST_DATA_DIR_NAME = "test_data"
TEST_WORKING_DIR = Path(__file__).parent.resolve() / TEST_WORKING_DIR_NAME
TEST_DATA_DIR = Path(__file__).parent.resolve() / TEST_DATA_DIR_NAME
MOCK_API_KEY = "mock_api_key"
REAL_API_KEY = os.environ.get("GEMINI_API_KEY")
TEST_MODEL = "gemini-2.0-flash" # Keep fast model
TEST_IMAGE1_FILENAME = "test_image.png"
TEST_IMAGE2_FILENAME = "test_image_2.png"
TEST_IMAGE1 = TEST_DATA_DIR / TEST_IMAGE1_FILENAME
TEST_IMAGE2 = TEST_DATA_DIR / TEST_IMAGE2_FILENAME

# YouTube URL test constants
YOUTUBE_URLS_VALID = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "http://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "http://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://youtube.com/embed/dQw4w9WgXcQ"
]

YOUTUBE_URLS_INVALID = [
    "https://vimeo.com/123456789",
    "https://example.com/video",
    "https://www.google.com",
    "not_a_url",
    "/local/file/path.mp4",
    "https://youtube.com/something_else",
    "https://youtu.be",
    "https://www.youtube.com"
]

EXPECTED_TEXT_IMG1 = "MCP server"
EXPECTED_TEXT_IMG1_ALT = "building from scratch"
EXPECTED_TEXT_IMG2 = "what will you build"
EXPECTED_TEXT_IMG2_ALT = "push gemini"


# --- Test Files Setup ---
def _ensure_test_file(file_path: Path):
    if not file_path.exists():
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            print(f"Warning: Created dummy file {file_path}. Integration tests may not be meaningful.")
        except OSError as e:
            print(f"Error creating dummy file {file_path}: {e}")
            pytest.fail(f"Could not create required test file: {file_path}")
    elif file_path.stat().st_size == 0:
         print(f"Warning: Test file {file_path} is empty. Integration tests results may not be meaningful.")

_ensure_test_file(TEST_IMAGE1)
_ensure_test_file(TEST_IMAGE2)

# --- YouTube URL Detection Tests ---
class TestYouTubeURLDetection:
    """Test YouTube URL detection functionality."""
    
    def test_is_youtube_url_valid_urls(self):
        """Test that valid YouTube URLs are correctly identified."""
        reader = MimeFilesReader(TEST_MODEL, MOCK_API_KEY, str(TEST_WORKING_DIR))
        
        for url in YOUTUBE_URLS_VALID:
            assert reader._is_youtube_url(url), f"Failed to detect valid YouTube URL: {url}"
    
    def test_is_youtube_url_invalid_urls(self):
        """Test that invalid URLs are correctly rejected."""
        reader = MimeFilesReader(TEST_MODEL, MOCK_API_KEY, str(TEST_WORKING_DIR))
        
        for url in YOUTUBE_URLS_INVALID:
            assert not reader._is_youtube_url(url), f"Incorrectly detected non-YouTube URL as valid: {url}"
    
    def test_is_youtube_url_edge_cases(self):
        """Test edge cases for YouTube URL detection."""
        reader = MimeFilesReader(TEST_MODEL, MOCK_API_KEY, str(TEST_WORKING_DIR))
        
        # Test empty string and None-like cases
        assert not reader._is_youtube_url(""), "Empty string should not be detected as YouTube URL"
        assert not reader._is_youtube_url("   "), "Whitespace string should not be detected as YouTube URL"
        
        # Test case sensitivity
        assert reader._is_youtube_url("https://YOUTUBE.com/watch?v=dQw4w9WgXcQ"), "Should handle uppercase domain"
        assert reader._is_youtube_url("https://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ"), "Should handle uppercase domain with www"


# --- YouTube Support Tests (Step 5) ---
class TestYouTubeSupport:
    """Comprehensive tests for YouTube URL support functionality."""
    
    def test_is_youtube_url_valid_formats(self):
        """Test YouTube URL detection for various formats."""
        reader = MimeFilesReader(TEST_MODEL, MOCK_API_KEY, str(TEST_WORKING_DIR))
        
        # Test all valid YouTube URL formats
        valid_formats = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ", 
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "http://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            # Case sensitivity tests
            "https://YOUTUBE.com/watch?v=dQw4w9WgXcQ",
            "https://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
            "https://YOUTU.BE/dQw4w9WgXcQ"
        ]
        
        for url in valid_formats:
            assert reader._is_youtube_url(url), f"Failed to detect valid YouTube URL: {url}"
    
    def test_is_youtube_url_invalid_formats(self):
        """Test that non-YouTube URLs are not detected as YouTube."""
        reader = MimeFilesReader(TEST_MODEL, MOCK_API_KEY, str(TEST_WORKING_DIR))
        
        # Test invalid URLs that should not be detected as YouTube
        invalid_formats = [
            "https://vimeo.com/123456789",
            "https://example.com/video", 
            "https://www.google.com",
            "not_a_url",
            "/local/file/path.mp4",
            "https://youtube.com/something_else",
            "https://youtu.be",  # Missing video ID
            "https://www.youtube.com",  # No watch path
            "https://youtube.fake.com/watch?v=123",  # Fake domain
            "https://notyoutube.com/watch?v=123",
            "file:///local/path/video.mp4",
            "ftp://example.com/video.mp4",
            "",  # Empty string
            "   ",  # Whitespace only
            "https://www.youtube.com/user/someuser",  # User page, not video
            "https://www.youtube.com/channel/somechannel"  # Channel page, not video
        ]
        
        for url in invalid_formats:
            assert not reader._is_youtube_url(url), f"Incorrectly detected non-YouTube URL as valid: {url}"
    
    @pytest.mark.integration
    @pytest.mark.skipif(not REAL_API_KEY, reason="GEMINI_API_KEY environment variable not set")
    def test_read_youtube_url_integration(self):
        """Integration test with real YouTube URL."""
        reader = MimeFilesReader(
            model_name=TEST_MODEL,
            google_genai_key=REAL_API_KEY,
            working_dir="."
        )
        
        # Use a stable, educational YouTube video (Khan Academy - Introduction to Programming)
        # This is a short, stable educational video that's unlikely to be removed
        youtube_url = "https://www.youtube.com/watch?v=FCMxA3m_Imc"
        question = "What is this video about? Provide a brief summary of the main topic."
        
        print(f"\nRunning YouTube integration test with URL: {youtube_url}")
        start_time = time.time()
        result = None
        
        try:
            result = reader.read(
                question=question, 
                files=[youtube_url], 
                output=None, 
                auto_cleanup=True
            )
        except Exception as e:
            pytest.fail(f"YouTube integration test failed with exception: {e}")
        finally:
            duration = time.time() - start_time
            print(f"YouTube integration test took {duration:.2f} seconds.")
            if result:
                print(f"YouTube Integration Test Response:\n---\n{result}\n---")
            else:
                print("YouTube Integration Test Warning: No result received.")
        
        # Verify response format and content
        assert isinstance(result, str), "Response should be a string"
        assert len(result) > 20, "Response should contain meaningful content"
        
        # Basic content validation - should mention programming or coding concepts
        result_lower = result.lower()
        programming_keywords = ["programming", "code", "coding", "computer", "software", "algorithm"]
        has_programming_content = any(keyword in result_lower for keyword in programming_keywords)
        
        assert has_programming_content, f"Response should contain programming-related content. Got: {result[:200]}..."


# --- Fixtures ---
@pytest.fixture(scope="module", autouse=True)
def setup_test_dirs():
    TEST_WORKING_DIR.mkdir(exist_ok=True)
    TEST_DATA_DIR.mkdir(exist_ok=True)
    yield

@pytest.fixture
def mock_genai_client():
    """Mocks the google.genai Client and its services."""
    with patch('mime_files_reader.reader.genai.Client', autospec=True) as MockClient:
        mock_client_instance = MockClient.return_value
        mock_files = mock_client_instance.files
        mock_models = mock_client_instance.models

        # Mock File object returned by upload
        mock_file_obj = MagicMock(spec=google_types.File)
        mock_file_obj.name = "uploaded_files/mock-file-123"
        mock_file_obj.uri = "gs://mock-bucket/mock-file-123"
        mock_file_obj.mime_type = "image/png"

        # Configure methods on the mock attributes
        mock_files.upload.return_value = mock_file_obj
        mock_files.delete.return_value = None

        # Mock generate_content_stream response structure
        mock_chunk1 = MagicMock(spec=google_types.GenerateContentResponse)
        mock_part1 = MagicMock(); mock_part1.text = "Mock response part 1."
        mock_content1 = MagicMock(); mock_content1.parts = [mock_part1]
        mock_candidate1 = MagicMock(); mock_candidate1.content = mock_content1
        mock_chunk1.candidates = [mock_candidate1]; mock_chunk1.text = None

        mock_chunk2 = MagicMock(spec=google_types.GenerateContentResponse)
        mock_part2 = MagicMock(); mock_part2.text = " Mock response part 2."
        mock_content2 = MagicMock(); mock_content2.parts = [mock_part2]
        mock_candidate2 = MagicMock(); mock_candidate2.content = mock_content2
        mock_chunk2.candidates = [mock_candidate2]; mock_chunk2.text = None

        mock_models.generate_content_stream.return_value = iter([mock_chunk1, mock_chunk2])

        yield {
            "MockClientClass": MockClient,
            "mock_client_instance": mock_client_instance,
            "mock_files_service": mock_files,
            "mock_models_service": mock_models,
            "mock_file_obj": mock_file_obj
        }

@pytest.fixture
def reader_instance(mock_genai_client, tmp_path):
    """Creates a MimeFilesReader instance with mocked client and temp working dir."""
    reader = MimeFilesReader(
        model_name=TEST_MODEL,
        google_genai_key=MOCK_API_KEY,
        working_dir=str(tmp_path)
    )
    reader.client = mock_genai_client["mock_client_instance"]
    yield reader


# --- Unit Tests ---

def test_init_success(mock_genai_client):
    MockClientClass = mock_genai_client["MockClientClass"]
    reader = MimeFilesReader(
        model_name=TEST_MODEL,
        google_genai_key=MOCK_API_KEY,
        working_dir=str(TEST_WORKING_DIR)
    )
    assert reader.model_name == TEST_MODEL
    assert reader.working_dir == TEST_WORKING_DIR.resolve()
    MockClientClass.assert_called_once_with(api_key=MOCK_API_KEY)
    assert reader.client is not None

@patch('mime_files_reader.reader.genai.Client', autospec=True)
def test_init_auth_failure(MockClientClass):
    MockClientClass.side_effect = google_exceptions.PermissionDenied("Auth error")
    with pytest.raises(google_exceptions.PermissionDenied):
        MimeFilesReader(model_name=TEST_MODEL, google_genai_key="bad_key")

def test_resolve_path(tmp_path):
    test_file = tmp_path / "temp_file.txt"; test_file.touch()
    test_dir = tmp_path / "temp_dir"; test_dir.mkdir()
    reader = MimeFilesReader(model_name=TEST_MODEL, google_genai_key=MOCK_API_KEY, working_dir=str(tmp_path))
    assert reader._resolve_path("temp_file.txt") == test_file.resolve()
    abs_path_str = str(test_file.resolve())
    assert reader._resolve_path(abs_path_str) == test_file.resolve()
    with pytest.raises(FileNotFoundError): reader._resolve_path("non_existent.txt")
    with pytest.raises(FileNotFoundError): reader._resolve_path("temp_dir")

def test_read_return_string(mock_genai_client, reader_instance, tmp_path):
    mock_files = mock_genai_client["mock_files_service"]
    mock_models = mock_genai_client["mock_models_service"]
    mock_file_obj = mock_genai_client["mock_file_obj"]
    test_file_rel_path = "input_image.png"
    test_file_abs_path = tmp_path / test_file_rel_path; test_file_abs_path.touch()
    question = "Describe this image."; files = [test_file_rel_path]

    # --- REMOVED patch for types.Part ---
    result = reader_instance.read(question=question, files=files, output=None, auto_cleanup=True)

    assert result == "Mock response part 1. Mock response part 2."
    mock_files.upload.assert_called_once_with(file=str(test_file_abs_path.resolve()))

    # --- MODIFIED Assertion: Check arguments passed to generate_content_stream ---
    mock_models.generate_content_stream.assert_called_once()
    call_args, call_kwargs = mock_models.generate_content_stream.call_args
    contents_list = call_kwargs.get('contents')
    assert isinstance(contents_list, list) and len(contents_list) == 1
    content_obj = contents_list[0]
    assert isinstance(content_obj, google_types.Content)
    assert len(content_obj.parts) == 2

    # Check the file part (created by the real Part.from_uri)
    file_part = content_obj.parts[0]
    assert isinstance(file_part, google_types.Part)
    # Access file_data attribute for URI and mime_type check
    assert hasattr(file_part, 'file_data'), "File part should have file_data attribute"
    assert file_part.file_data.file_uri == mock_file_obj.uri
    assert file_part.file_data.mime_type == mock_file_obj.mime_type

    # Check the text part (created by the real Part.from_text)
    text_part = content_obj.parts[1]
    assert isinstance(text_part, google_types.Part)
    assert hasattr(text_part, 'text'), "Text part should have text attribute"
    assert text_part.text == question
    # --- END MODIFIED Assertion ---

    mock_files.delete.assert_called_once_with(name=mock_file_obj.name)

def test_read_write_to_file(mock_genai_client, reader_instance, tmp_path):
    mock_files = mock_genai_client["mock_files_service"]
    mock_models = mock_genai_client["mock_models_service"]
    test_file_rel_path = "input_for_output.png"
    test_file_abs_path = tmp_path / test_file_rel_path; test_file_abs_path.touch()
    output_file_rel = "output/result.txt"
    output_file_abs = tmp_path / output_file_rel
    question = "Describe."; files = [test_file_rel_path]

    result = reader_instance.read(question=question, files=files, output=output_file_rel, auto_cleanup=False)

    expected_output_path_abs = output_file_abs.resolve()
    expected_message = f"Mime files successfully processed. Results written to: {expected_output_path_abs}"
    assert result == expected_message
    assert output_file_abs.exists()
    assert output_file_abs.read_text(encoding="utf-8") == "Mock response part 1. Mock response part 2."
    mock_files.upload.assert_called_once()
    mock_models.generate_content_stream.assert_called_once()
    mock_files.delete.assert_not_called()

def test_read_input_file_not_found(reader_instance):
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        reader_instance.read(question="Test", files=["non_existent_file.txt"])

def test_read_input_path_is_directory(reader_instance, tmp_path):
    dir_path_rel = "a_directory"; dir_path_abs = tmp_path / dir_path_rel; dir_path_abs.mkdir()
    with pytest.raises(FileNotFoundError, match="Input path exists but is not a file"):
         reader_instance.read(question="Test", files=[dir_path_rel])

def test_read_upload_fails(mock_genai_client, reader_instance, tmp_path):
    mock_files = mock_genai_client["mock_files_service"]
    mock_models = mock_genai_client["mock_models_service"]
    mock_files.upload.return_value = None
    test_file_rel = "image_upload_fail.png"; (tmp_path / test_file_rel).touch()
    with pytest.raises(ValueError, match="No files were successfully prepared"):
        reader_instance.read(question="Test", files=[test_file_rel])
    mock_files.upload.assert_called_once()
    mock_models.generate_content_stream.assert_not_called()
    mock_files.delete.assert_not_called()

def test_read_upload_returns_no_uri(mock_genai_client, reader_instance, tmp_path):
    """Test cleanup happens even if file part cannot be created due to missing URI."""
    mock_files = mock_genai_client["mock_files_service"]
    mock_models = mock_genai_client["mock_models_service"]
    # Get the standard mock file object
    mock_file_obj_no_uri = mock_genai_client["mock_file_obj"]
    # Set its uri to None *after* getting it from the fixture
    mock_file_obj_no_uri.uri = None
    # Ensure the mocked upload returns this modified object
    mock_files.upload.return_value = mock_file_obj_no_uri

    test_file_rel = "image_no_uri.png"; (tmp_path / test_file_rel).touch()

    # Expect ValueError because no parts will be created
    with pytest.raises(ValueError, match="No files were successfully prepared"):
        reader_instance.read(question="Test", files=[test_file_rel], auto_cleanup=True)

    # Verify API calls
    mock_files.upload.assert_called_once()
    mock_models.generate_content_stream.assert_not_called()
    # --- MODIFIED Assertion: Delete SHOULD now be called ---
    mock_files.delete.assert_called_once_with(name=mock_file_obj_no_uri.name)
    # --- END MODIFIED Assertion ---


# --- Integration Tests (Unchanged) ---

def get_integration_file_path(filename):
    """Helper to get absolute path to test data files."""
    return str(TEST_DATA_DIR / filename)

@pytest.mark.integration
@pytest.mark.skipif(not REAL_API_KEY, reason="GEMINI_API_KEY environment variable not set")
@pytest.mark.skipif(not TEST_IMAGE1.exists() or TEST_IMAGE1.stat().st_size == 0, reason=f"Test file {TEST_IMAGE1} not found or empty.")
@pytest.mark.skipif(not TEST_IMAGE2.exists() or TEST_IMAGE2.stat().st_size == 0, reason=f"Test file {TEST_IMAGE2} not found or empty.")
def test_integration_read_images_return_string():
    reader = MimeFilesReader(
        model_name=TEST_MODEL,
        google_genai_key=REAL_API_KEY,
        working_dir="."
    )
    question = "Output the text content visible in these images, each on a new line if possible."
    files = [
        get_integration_file_path(TEST_IMAGE1_FILENAME),
        get_integration_file_path(TEST_IMAGE2_FILENAME)
    ]
    print(f"\nRunning integration test: Reading {files} using model {TEST_MODEL}")
    start_time = time.time(); result = None
    try: result = reader.read(question=question, files=files, output=None, auto_cleanup=True)
    except Exception as e: pytest.fail(f"Integration test failed with exception: {e}")
    finally:
        duration = time.time() - start_time
        print(f"Integration test (string return, 2 images) took {duration:.2f} seconds.")
        if result: print(f"Integration Test Response (String):\n---\n{result}\n---")
        else: print("Integration Test Warning: No result received.")
    assert isinstance(result, str); assert len(result) > 10
    result_lower = result.lower()
    assert EXPECTED_TEXT_IMG1.lower() in result_lower or EXPECTED_TEXT_IMG1_ALT.lower() in result_lower, \
        f"Expected text fragment for {TEST_IMAGE1_FILENAME} not found in response."
    assert EXPECTED_TEXT_IMG2.lower() in result_lower or EXPECTED_TEXT_IMG2_ALT.lower() in result_lower, \
        f"Expected text fragment for {TEST_IMAGE2_FILENAME} not found in response."


@pytest.mark.integration
@pytest.mark.skipif(not REAL_API_KEY, reason="GEMINI_API_KEY environment variable not set")
@pytest.mark.skipif(not TEST_IMAGE1.exists() or TEST_IMAGE1.stat().st_size == 0, reason=f"Test file {TEST_IMAGE1} not found or empty.")
def test_integration_read_image1_write_file(tmp_path):
    output_file_abs = tmp_path / "integration_output_img1.txt"
    reader = MimeFilesReader(
        model_name=TEST_MODEL,
        google_genai_key=REAL_API_KEY,
        working_dir="."
    )
    question = f"Extract and output only the text visible in the image file '{TEST_IMAGE1_FILENAME}'."
    files = [get_integration_file_path(TEST_IMAGE1_FILENAME)]
    print(f"\nRunning integration test: Reading {files}, output to {output_file_abs} using model {TEST_MODEL}")
    start_time = time.time(); result_msg = None; content = ""
    try:
        result_msg = reader.read(question=question, files=files, output=str(output_file_abs), auto_cleanup=True)
        if output_file_abs.exists(): content = output_file_abs.read_text(encoding="utf-8")
    except Exception as e: pytest.fail(f"Integration test failed with exception: {e}")
    finally:
        duration = time.time() - start_time
        print(f"Integration test (file output, img1) took {duration:.2f} seconds.")
        if result_msg: print(f"Result Message: {result_msg}")
        if content: print(f"Integration Test Response (File Content):\n---\n{content}\n---")
        else: print(f"Integration Test Warning: Output file '{output_file_abs}' not found or empty.")
    expected_output_path_abs = output_file_abs.resolve()
    assert result_msg == f"Mime files successfully processed. Results written to: {expected_output_path_abs}"
    assert output_file_abs.exists(); assert isinstance(content, str); assert len(content) > 5
    content_lower = content.lower()
    assert EXPECTED_TEXT_IMG1.lower() in content_lower or EXPECTED_TEXT_IMG1_ALT.lower() in content_lower, \
        f"Expected text fragment '{EXPECTED_TEXT_IMG1}' or '{EXPECTED_TEXT_IMG1_ALT}' not found in output file content."


@pytest.mark.integration
@pytest.mark.skipif(not REAL_API_KEY, reason="GEMINI_API_KEY environment variable not set")
def test_integration_non_existent_file():
    reader = MimeFilesReader(
        model_name=TEST_MODEL,
        google_genai_key=REAL_API_KEY,
        working_dir="."
    )
    question = "This should fail."
    files = ["this_file_absolutely_does_not_exist_anywhere.kjsdfh"]
    print(f"\nRunning integration test: Attempting to read non-existent file.")
    with pytest.raises(FileNotFoundError):
        reader.read(question=question, files=files)