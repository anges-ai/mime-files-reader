# tests/test_cli.py

import pytest
import sys
from unittest.mock import patch, MagicMock
from mime_files_reader.cli import validate_inputs, main


class TestCLIYouTubeSupport:
    """Test CLI support for YouTube URLs."""
    
    def test_validate_inputs_accepts_youtube_urls(self):
        """Test that validate_inputs accepts various YouTube URL formats."""
        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://YouTube.com/watch?v=dQw4w9WgXcQ",  # Test case insensitive
            "http://youtu.be/dQw4w9WgXcQ"  # Test http vs https
        ]
        
        # Should not raise any exceptions
        try:
            validate_inputs(youtube_urls)
        except Exception as e:
            pytest.fail(f"validate_inputs should accept YouTube URLs but raised: {e}")
    
    def test_validate_inputs_rejects_invalid_files(self):
        """Test that validate_inputs rejects invalid file paths that aren't YouTube URLs."""
        invalid_inputs = [
            "/nonexistent/file.txt",
            "not_a_file.pdf",
            "https://example.com/not_youtube.mp4"
        ]
        
        with pytest.raises(FileNotFoundError):
            validate_inputs(invalid_inputs)
    
    @patch('mime_files_reader.cli.MimeFilesReader')
    @patch('sys.argv')
    def test_cli_accepts_youtube_url_arguments(self, mock_argv, mock_reader_class):
        """Test that CLI main function accepts YouTube URL arguments without errors."""
        # Mock the command line arguments
        mock_argv.__getitem__.side_effect = lambda x: [
            'cli.py',
            '--key', 'test_api_key',
            '--question', 'What is this video about?',
            '--files', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        ][x]
        mock_argv.__len__.return_value = 5
        
        # Mock the reader instance
        mock_reader = MagicMock()
        mock_reader.read.return_value = "Test response"
        mock_reader_class.return_value = mock_reader
        
        # Mock environment variable
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_api_key'}):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
                # Create a mock args object
                mock_args = MagicMock()
                mock_args.key = 'test_api_key'
                mock_args.question = 'What is this video about?'
                mock_args.files = ['https://www.youtube.com/watch?v=dQw4w9WgXcQ']
                mock_args.model = 'gemini-2.0-flash'
                mock_args.working_dir = '.'
                mock_args.output = None
                mock_args.no_cleanup = False
                mock_parse_args.return_value = mock_args
                
                # Mock print to capture output
                with patch('builtins.print') as mock_print:
                    # Should not raise any exceptions
                    try:
                        main()
                        # Verify the reader was called with YouTube URL
                        mock_reader.read.assert_called_once_with(
                            question='What is this video about?',
                            files=['https://www.youtube.com/watch?v=dQw4w9WgXcQ'],
                            output=None,
                            auto_cleanup=True
                        )
                        mock_print.assert_called_with("Test response")
                    except Exception as e:
                        pytest.fail(f"CLI should accept YouTube URLs but raised: {e}")