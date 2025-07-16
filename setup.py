# setup.py

from setuptools import setup, find_packages
import os

# Function to read requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Read README.md for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mime-files-reader",
    version="0.2.0", # Match version in __init__.py
    author="Anges",
    author_email="me@anges.ai", # Replace with your email
    description="A tool to process various file types (images, PDFs, audio) using Google Generative AI",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/anges-ai/mime-files-reader", # Replace with your repo URL
    packages=find_packages(exclude=["tests*", "docs", "examples"]),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.8', # Specify your minimum Python version
    classifiers=[
        "Development Status :: 3 - Alpha", # Adjust as appropriate
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    keywords="generative ai, gemini, google genai, file processing, multimodal",
    entry_points={
        "console_scripts": [
            "mime-reader=mime_files_reader.cli:main",
            'mime-reader-mcp-server=mime_files_reader.mcp_server:main',
        ],
    },
    # Optional: Include package data like default configs, templates etc.
    # package_data={
    #     'your_package': ['data/*.json'],
    # },
    # include_package_data=True,
)
