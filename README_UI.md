# Speaker Diarization Web UI

A web-based interface for speaker diarization with speaker identification capabilities.

## Features

- **Speaker Reference Management**: Upload and manage speaker reference audio files with name labels
- **Audio Processing**: Upload audio files for diarization and speaker identification
- **Real-time Results**: View diarization results directly in the browser
- **Configurable Parameters**:
  - Whisper model selection (tiny to large-v3)
  - Language detection or manual selection
  - Diarizer choice (MSDD or Sortformer)
  - Similarity threshold (default: 0.4)
  - Vocal stem separation toggle

## Workflow

1. **Upload Speaker References**
   - Add speaker name (e.g., "Phong", "An")
   - Upload reference audio file
   - System automatically creates embeddings for matching

2. **Process Audio**
   - Upload audio file to analyze
   - Configure processing parameters:
     - Whisper Model: Speech-to-text model
     - Language: Auto-detect or manual selection
     - Diarizer: MSDD or Sortformer
     - Threshold: 0.4 (cosine similarity for speaker matching)
   - Start processing

3. **View Results**
   - Transcript with speaker labels
   - Output files (.txt, .srt) saved to `data/processed/`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch (if not already installed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Quick Start

```bash
# Make script executable
chmod +x start_server.sh

# Run the server
./start_server.sh
```

The server will start at `http://localhost:8000`

### Manual Start

```bash
python app_api.py
```

Then open `http://localhost:8000` in your browser.

## Directory Structure

```
speaker_diarization/
├── data/
│   ├── speaker_references/    # Reference audio files for known speakers
│   ├── uploads/               # Uploaded audio files for processing
│   └── processed/             # Output files (.txt, .srt)
├── static/                    # Web UI files
├── app_api.py                 # FastAPI backend
├── ui.html                    # Frontend UI
├── diarize.py                 # Main diarization logic
├── speaker_identification.py  # Speaker embedding and matching
└── start_server.sh           # Startup script
```

## Configuration

### Threshold Setting

The threshold (default: 0.4) controls speaker matching sensitivity:
- **Lower (0.1-0.3)**: More likely to match speakers (may produce false positives)
- **Higher (0.5-0.7)**: Stricter matching (may miss valid matches)
- **Recommended**: 0.4 for Vietnamese audio

### Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 75 MB | Fastest | Lower |
| base | 142 MB | Fast | Low |
| small | 466 MB | Medium | Medium |
| medium | 1.5 GB | Slow | High |
| large-v3 | 3.1 GB | Slowest | Highest |

### Diarizers

- **MSDD**: Multi-scale diarization model, good for most cases
- **Sortformer**: Transformer-based diarization, may perform better on some audio types

## API Endpoints

### Speaker Management

- `GET /api/speakers` - List all registered speakers
- `POST /api/speakers` - Upload a new speaker reference
- `DELETE /api/speakers/{id}` - Delete a speaker
- `GET /api/speakers/{id}/audio` - Stream speaker audio

### Audio Processing

- `POST /api/audio/upload` - Upload audio file
- `POST /api/process` - Process audio with diarization
- `GET /api/results/{filename}` - Download result files

### Configuration

- `GET /api/config` - Get available models and settings

## Notes

- Speaker embeddings are automatically cached in `data/speaker_references/.speaker_embeddings_cache.npz`
- Reference audio files should be at least 1-2 seconds long for best results
- Supported audio formats: .wav, .mp3, .m4a, .flac, .ogg, .opus
- Processing time depends on audio length and selected model size

## Troubleshooting

**Issue**: Speaker identification not working
- **Solution**: Ensure you have uploaded speaker references first

**Issue**: Out of memory error
- **Solution**: Use a smaller Whisper model (e.g., "small" instead of "large-v3")
- **Solution**: Reduce batch size in the code

**Issue**: Poor diarization quality
- **Solution**: Enable vocal stem separation
- **Solution**: Adjust the threshold value
- **Solution**: Try a different diarizer (MSDD vs Sortformer)
