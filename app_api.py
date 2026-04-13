"""
Speaker Diarization Web UI - Backend API
Handles speaker reference management, audio uploads, and diarization processing.
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("app-api")

app = FastAPI(title="Speaker Diarization API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SPEAKER_REFS_DIR = DATA_DIR / "speaker_references"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_SAMPLES_DIR = DATA_DIR / "upload_samples"

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}

# Ensure directories exist
for d in [SPEAKER_REFS_DIR, UPLOADS_DIR, PROCESSED_DIR, STATIC_DIR, UPLOAD_SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


class ProcessRequest(BaseModel):
    audio_filepath: str
    whisper_model: str = "large-v3"
    language: Optional[str] = None
    diarizer: str = "sortformer"
    threshold: float = 0.45
    batch_size: int = 64


@app.get("/api/speakers")
async def list_speakers():
    """List all registered speakers with their audio files."""
    speakers = []
    if SPEAKER_REFS_DIR.exists():
        for audio_file in sorted(SPEAKER_REFS_DIR.iterdir()):
            if audio_file.is_file() and audio_file.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                stat = audio_file.stat()
                size_kb = stat.st_size / 1024
                speakers.append(
                    {
                        "id": audio_file.stem,
                        "name": audio_file.stem,
                        "filename": audio_file.name,
                        "size_kb": round(size_kb, 1),
                        "has_embedding": True,
                    }
                )
    return {"speakers": speakers}


@app.post("/api/speakers", status_code=201)
async def upload_speaker(name: str = Form(...), audio: UploadFile = File(...)):
    """Upload a speaker reference audio with a name label."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Speaker name is required")
    
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file selected")
    
    if not allowed_file(audio.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    speaker_name = name.strip()
    original_ext = Path(audio.filename).suffix
    safe_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '-', '_')).strip()
    unique_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}{original_ext}"
    filepath = SPEAKER_REFS_DIR / unique_filename
    
    try:
        content = await audio.read()
        filepath.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Rename to just the speaker name + extension if no conflict
    final_path = SPEAKER_REFS_DIR / f"{safe_name}{original_ext}"
    if final_path.exists() and final_path != filepath:
        pass  # Keep unique filename
    else:
        filepath.rename(final_path)
        filepath = final_path
    
    logger.info(f"Saved speaker reference: {filepath.name}")
    
    return {
        "id": filepath.stem,
        "name": filepath.stem,
        "filename": filepath.name,
        "size_kb": round(filepath.stat().st_size / 1024, 1),
        "has_embedding": True,
    }


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete a speaker reference."""
    deleted = False
    if SPEAKER_REFS_DIR.exists():
        for audio_file in SPEAKER_REFS_DIR.iterdir():
            if audio_file.stem.startswith(speaker_id) or speaker_id.startswith(audio_file.stem):
                audio_file.unlink()
                logger.info(f"Deleted speaker: {audio_file.name}")
                deleted = True
                break
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    return {"success": True}


@app.get("/api/speakers/{speaker_id}/audio")
async def get_speaker_audio(speaker_id: str):
    """Stream speaker reference audio."""
    if SPEAKER_REFS_DIR.exists():
        for audio_file in SPEAKER_REFS_DIR.iterdir():
            if audio_file.stem.startswith(speaker_id):
                return FileResponse(str(audio_file), media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio not found")


@app.post("/api/audio/upload", status_code=201)
async def upload_audio(audio: UploadFile = File(...)):
    """Upload an audio file for processing."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file selected")
    
    if not allowed_file(audio.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    # Save with unique name
    unique_filename = f"{uuid.uuid4().hex}_{audio.filename}"
    filepath = UPLOADS_DIR / unique_filename
    
    try:
        content = await audio.read()
        filepath.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    logger.info(f"Uploaded audio: {filepath.name}")
    
    return {
        "id": filepath.stem,
        "filename": filepath.name,
        "filepath": str(filepath),
        "size_kb": round(filepath.stat().st_size / 1024, 1),
    }


@app.get("/api/audio/{audio_id}/stream")
async def stream_audio(audio_id: str):
    """Stream uploaded audio."""
    if UPLOADS_DIR.exists():
        for audio_file in UPLOADS_DIR.iterdir():
            if audio_id in audio_file.stem:
                return FileResponse(str(audio_file), media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio not found")


@app.get("/api/samples/{sample_id}/stream")
async def stream_sample_audio(sample_id: str):
    """Stream sample audio file."""
    # Map sample ID to actual file
    sample_files = {
        'phong_van': 'phỏng vấn.mp3',
        'tro_chuyen': 'trò chuyện.wav'
    }
    
    filename = sample_files.get(sample_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    filepath = UPLOAD_SAMPLES_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    
    media_type = "audio/mpeg" if filepath.suffix.lower() == '.mp3' else "audio/wav"
    return FileResponse(str(filepath), media_type=media_type)


@app.post("/api/samples/{sample_id}/prepare")
async def prepare_sample(sample_id: str):
    """Copy sample file to uploads directory and return filepath for processing."""
    # Map sample ID to actual filename
    sample_mapping = {
        'phong_van': 'phỏng vấn.mp3',
        'tro_chuyen': 'trò chuyện.wav',
    }
    
    filename = sample_mapping.get(sample_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    source_path = UPLOAD_SAMPLES_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    
    # Copy to uploads directory with unique name
    unique_filename = f"sample_{sample_id}_{uuid.uuid4().hex[:8]}{source_path.suffix}"
    dest_path = UPLOADS_DIR / unique_filename
    shutil.copy(str(source_path), str(dest_path))
    
    logger.info(f"Prepared sample: {dest_path.name}")
    
    return {
        "id": dest_path.stem,
        "filename": dest_path.name,
        "filepath": str(dest_path),
        "size_kb": round(dest_path.stat().st_size / 1024, 1),
    }


@app.get("/api/samples")
async def list_samples():
    """List available sample audio files."""
    # Map actual filenames to URL-safe IDs
    sample_mapping = {
        'phong_van': {
            'filename': 'phỏng vấn.mp3',
            'display_name': 'Phỏng vấn',
            'description': 'Cuộc phỏng vấn với 3 nhân vật: Cường Nguyễn, Nhân sự 1 và Nhân sự 2',
            'speakers': ['Cường Nguyễn', 'Nhân sự 1', 'Nhân sự 2'],
        },
        'tro_chuyen': {
            'filename': 'trò chuyện.wav',
            'display_name': 'Trò chuyện',
            'description': 'Cuộc trò chuyện giữa anh Khánh, chị Hoàng Anh và anh Toàn',
            'speakers': ['anh Khánh', 'chị Hoàng Anh', 'anh Toàn'],
        },
    }
    
    samples = []
    if UPLOAD_SAMPLES_DIR.exists():
        for sample_id, info in sample_mapping.items():
            filepath = UPLOAD_SAMPLES_DIR / info['filename']
            if filepath.exists():
                stat = filepath.stat()
                size_kb = stat.st_size / 1024
                size_mb = size_kb / 1024
                samples.append({
                    "id": sample_id,
                    "display_name": info['display_name'],
                    "filename": info['filename'],
                    "description": info['description'],
                    "speakers": info['speakers'],
                    "size_kb": round(size_kb, 1),
                    "size_mb": round(size_mb, 2),
                })
    return {"samples": samples}


@app.post("/api/process")
async def process_audio(request: ProcessRequest):
    """Process audio through diarization and speaker identification."""
    audio_path = Path(request.audio_filepath)
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Build command line:
    # python diarize.py -a "file.mp3" --no-stem --device cuda --batch-size 16 --diarizer sortformer --identify-threshold 0.45 --language vi --whisper-model large-v3 --speakers-dir "./data/speaker_references"
    cmd = [
        "python",
        str(BASE_DIR / "diarize.py"),
        "-a", str(audio_path),
        "--no-stem",
        "--device", "cuda",
        "--batch-size", str(request.batch_size),
        "--diarizer", request.diarizer,
        "--identify-threshold", str(request.threshold),
    ]
    
    # Add language if specified
    if request.language:
        cmd.extend(["--language", request.language])
    
    # Add whisper model
    cmd.extend(["--whisper-model", request.whisper_model])
    
    # Add speakers dir if we have speaker references
    if SPEAKER_REFS_DIR.exists() and any(
        f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
        for f in SPEAKER_REFS_DIR.iterdir()
        if f.is_file()
    ):
        cmd.extend(["--speakers-dir", str(SPEAKER_REFS_DIR)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown error"
            logger.error(f"Diarization failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Diarization failed: {error_msg}")
        
        # Read results
        output_prefix = audio_path.stem
        output_dir = audio_path.parent
        txt_file = output_dir / f"{output_prefix}.txt"
        srt_file = output_dir / f"{output_prefix}.srt"
        
        transcript = ""
        if txt_file.exists():
            transcript = txt_file.read_text(encoding="utf-8-sig")
        
        # Move output to processed directory
        output_files = {"txt": None, "srt": None}
        if txt_file.exists():
            shutil.move(str(txt_file), str(PROCESSED_DIR / txt_file.name))
            output_files["txt"] = str(PROCESSED_DIR / txt_file.name)
            if srt_file.exists():
                shutil.move(str(srt_file), str(PROCESSED_DIR / srt_file.name))
                output_files["srt"] = str(PROCESSED_DIR / srt_file.name)
        
        return {
            "success": True,
            "transcript": transcript,
            "output_files": output_files,
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Processing timeout")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{filename}")
async def download_result(filename: str):
    """Download processed result file."""
    safe_filename = "".join(c for c in filename if c.isalnum() or c in ('.', '-', '_'))
    filepath = PROCESSED_DIR / safe_filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(filepath), filename=filename)


@app.get("/api/config")
async def get_config():
    """Get application configuration."""
    return {
        "threshold": 0.45,
        "device": "cuda",
        "batch_size": 64,
        "whisper_models": [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
        ],
        "languages": [
            {"code": None, "name": "Auto-detect"},
            {"code": "vi", "name": "Vietnamese"},
            {"code": "en", "name": "English"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "es", "name": "Spanish"},
            {"code": "ru", "name": "Russian"},
        ],
        "diarizers": [
            {"id": "msdd", "name": "MSDD"},
            {"id": "sortformer", "name": "Sortformer"},
        ],
    }


@app.get("/")
async def serve_ui():
    """Serve the UI HTML file."""
    ui_file = STATIC_DIR / "index.html"
    if not ui_file.exists():
        ui_file = BASE_DIR / "ui.html"
    if ui_file.exists():
        return FileResponse(str(ui_file), media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "UI not found"})


if __name__ == "__main__":
    import uvicorn
    
    # Copy ui.html to static directory
    ui_source = BASE_DIR / "ui.html"
    if ui_source.exists():
        shutil.copy(str(ui_source), str(STATIC_DIR / "index.html"))
    
    uvicorn.run(app, host="0.0.0.0", port=5001)
