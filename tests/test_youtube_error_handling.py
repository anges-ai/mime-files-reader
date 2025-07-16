#!/usr/bin/env python3
"""
Test script to verify YouTube URL error handling improvements in Step 8.

This script tests the enhanced _process_youtube_url() method with various
malformed URLs to ensure proper error handling and user-friendly error messages.
"""

import os
import sys
from pathlib import Path

# Add the mime_files_reader directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mime_files_reader.reader import MimeFilesReader

def test_youtube_url_error_handling():
    """
    Test the enhanced YouTube URL error handling functionality.
    """
    # Use a dummy API key for testing (we won't actually call the API)
    api_key = os.environ.get('GOOGLE_GENAI_API_KEY', 'dummy_key_for_testing')
    
    try:
        reader = MimeFilesReader(
            model_name='gemini-1.5-flash-latest',
            google_genai_key=api_key,
            working_dir='.'
        )
        
        # Test cases for malformed URLs
        test_cases = [
            {
                'url': 'not_a_url_at_all',
                'description': 'Invalid URL format - no scheme'
            },
            {
                'url': 'ftp://youtube.com/watch?v=test',
                'description': 'Invalid scheme - FTP instead of HTTP/HTTPS'
            },
            {
                'url': 'https://example.com/video',
                'description': 'Valid URL format but not YouTube'
            },
            {
                'url': 'http://',
                'description': 'Incomplete URL - missing netloc'
            },
            {
                'url': '',
                'description': 'Empty URL'
            },
            {
                'url': 'https://youtube.com/invalid_path',
                'description': 'YouTube domain but invalid path'
            }
        ]
        
        print("Testing YouTube URL error handling...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            url = test_case['url']
            description = test_case['description']
            
            print(f"Test {i}: {description}")
            print(f"URL: '{url}'")
            
            try:
                # This should raise a ValueError with a descriptive message
                reader._process_youtube_url(url)
                print("❌ UNEXPECTED: No error was raised!")
            except ValueError as e:
                print(f"✅ EXPECTED: ValueError raised with message: {e}")
            except Exception as e:
                print(f"⚠️  UNEXPECTED: Different exception type: {type(e).__name__}: {e}")
            
            print("-" * 60)
        
        # Test a valid YouTube URL (should work)
        print("\nTesting valid YouTube URL...")
        valid_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        print(f"URL: {valid_url}")
        
        try:
            result = reader._process_youtube_url(valid_url)
            print(f"✅ SUCCESS: Valid URL processed successfully")
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR with valid URL: {e}")
        
        print("\n" + "=" * 60)
        print("YouTube URL error handling test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = test_youtube_url_error_handling()
    sys.exit(0 if success else 1)