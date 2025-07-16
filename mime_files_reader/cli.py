# mime_files_reader/cli.py

import argparse
import os
import re
import sys
from typing import List
from dotenv import load_dotenv
from mime_files_reader.reader import MimeFilesReader


def validate_inputs(files: List[str]) -> None:
    """
    Validate that inputs are either valid file paths or YouTube URLs.
    
    Args:
        files: List of file paths or YouTube URLs to validate
        
    Raises:
        FileNotFoundError: If an input is neither a valid file nor a YouTube URL
    """
    youtube_patterns = [
        r'https?://(www\.)?youtube\.com/watch\?.*v=',
        r'https?://youtu\.be/',
        r'https?://(www\.)?youtube\.com/embed/'
    ]
    
    def is_youtube_url(input_str: str) -> bool:
        """Check if the input string is a YouTube URL."""
        if not input_str or not input_str.strip():
            return False
            
        for pattern in youtube_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        return False
    
    for file_input in files:
        if not is_youtube_url(file_input):
            # Check if it's a valid local file path
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"File not found and not a valid YouTube URL: {file_input}")


def main():
    """Main function for the CLI."""
    # Load .env file if it exists
    load_dotenv()

    parser = argparse.ArgumentParser(description="Process files using Google Generative AI.")

    parser.add_argument("-m", "--model", default="gemini-2.0-flash",
                        help="Name of the Google GenAI model to use (default: gemini-2.0-flash).")
    parser.add_argument("-k", "--key", default=os.environ.get("GEMINI_API_KEY"),
                        help="Google GenAI API key (defaults to GEMINI_API_KEY environment variable).")
    parser.add_argument("-w", "--working-dir", default=".",
                        help="Working directory for relative paths (default: current directory).")
    parser.add_argument("-q", "--question", required=True,
                        help="The question to ask about the files.")
    parser.add_argument("-f", "--files", required=True, nargs='+',
                        help="List of file paths or YouTube URLs to process. Supports local files (images, PDFs, audio, video) and YouTube video URLs.")
    parser.add_argument("-o", "--output", default=None,
                        help="Optional output file path to save the response.")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Disable automatic cleanup of uploaded files on GenAI.")

    args = parser.parse_args()

    if not args.key:
        print("Error: Google GenAI API key not found. "
              "Please set the GEMINI_API_KEY environment variable or use the --key option.", file=sys.stderr)
        sys.exit(1)

    # Validate inputs before processing
    try:
        validate_inputs(args.files)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        reader = MimeFilesReader(
            model_name=args.model,
            google_genai_key=args.key,
            working_dir=args.working_dir
        )

        result = reader.read(
            question=args.question,
            files=args.files,
            output=args.output,
            auto_cleanup=not args.no_cleanup
        )

        print(result)  # Print the result (either response or confirmation message)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()