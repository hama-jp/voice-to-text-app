# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese voice-to-text web application that combines OpenAI Whisper (large-v3) for speech recognition with Qwen3-8B for advanced Japanese text correction. The application provides a web interface for uploading audio files and receiving high-quality transcriptions with optional AI-powered text correction.

## Architecture

### Core Components
- **Backend**: FastAPI server (`backend/main.py`) providing REST API endpoints
- **Frontend**: HTML5/JavaScript web interface (`frontend/index.html`) with drag-and-drop functionality
- **Text Correction**: Modular correction system (`text_corrector.py`) with basic and LLM-based correction
- **Model Management**: Whisper model loading and Qwen3-8B integration

### Data Flow
1. Audio upload via web interface → FastAPI endpoint (`/transcribe`)
2. Whisper model processes audio → raw transcription
3. Optional text correction (basic regex-based or LLM-based)
4. Structured JSON response with transcription, corrections, and metadata

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Quick start (recommended)
python start_server.py

# Manual backend start
python backend/main.py

# Manual frontend access
open frontend/index.html  # macOS
xdg-open frontend/index.html  # Linux
```

### Testing
```bash
# Test individual components
python test_whisper.py          # Whisper functionality
python test_simple_corrector.py # Basic text correction
python test_qwen_integration.py # LLM integration
python test_llm.py             # LLM model loading

# API health check
curl http://localhost:8000/health
```

## Key Configuration

### Model Settings
- **Whisper Model**: `large-v3` (configurable in `backend/main.py`)
- **Language**: Japanese (`ja`) with optimized parameters
- **Qwen Model**: `Qwen/Qwen3-8B` with 4-bit quantization
- **GPU Support**: CUDA-enabled with fallback to CPU

### API Endpoints
- `POST /transcribe` - Main transcription endpoint
- `GET /health` - System status
- `GET /supported_formats` - Audio format support
- `GET /docs` - FastAPI auto-generated documentation

### Directory Structure
- `uploads/` - Temporary audio file storage
- `outputs/` - Generated transcription files
- `models/` - Model cache directory
- `venv/` - Python virtual environment

## Development Notes

### Text Correction System
The `JapaneseTextCorrector` class supports two modes:
- **Basic**: Regex-based correction for common Japanese errors
- **Advanced**: LLM-powered correction using Qwen3-8B

### Error Handling
- File upload validation (format, size limits)
- GPU/CPU fallback for models
- Graceful degradation if LLM unavailable
- Comprehensive error reporting in API responses

### Performance Considerations
- GPU processing significantly faster than CPU
- LLM correction adds processing time but improves quality
- Models loaded once at startup for efficiency
- Temporary file cleanup after processing

## Supported Audio Formats
MP3, WAV, M4A, FLAC, AAC (up to 100MB)