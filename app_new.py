#!/usr/bin/env python3
"""
Speaker Diarization Web UI - FastAPI Backend with Claude speaker naming
Flow: upload audio -> call Deepgram -> infer speaker names with Claude ->
return final transcript with named speakers.
"""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("app_new")

# Global task storage
tasks = {}

PUBLIC_UPLOAD_ERROR_MESSAGE = "Có lỗi xảy ra. Vui lòng thử lại sau."
PUBLIC_PROCESSING_ERROR_MESSAGE = "Xử lý thất bại."

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest")
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Constraints
MAX_AUDIO_DURATION_SECONDS = 45 * 60  # 45 minutes
RECOMMENDED_MAX_SPEAKERS = 8  # Best performance with fewer speakers

CLAUDE_SYSTEM_PROMPT = """Bạn là một hệ thống chuyên xác định danh tính người nói (speaker identification) dựa trên transcript hội thoại.

Nhiệm vụ của bạn:
- Gán tên chính xác nhất có thể cho từng speaker dựa trên ngữ cảnh và cách các nhân vật giao tiếp với nhau.
- Input đã được diarization (ví dụ: speaker 0, speaker 1, ...).

## NGUYÊN TẮC SUY LUẬN (RẤT QUAN TRỌNG)

1. Ưu tiên cao nhất: SUY LUẬN TÊN
   - Phân tích cách các nhân vật gọi nhau trong hội thoại:
     - Ví dụ: "Anh Long ơi", "Bình nói đúng", "Em gửi cho chị Hoa rồi"
     → Đây là tín hiệu mạnh để xác định tên của người KHÁC
   - Dựa vào nhiều lượt hội thoại để suy luận ngược lại ai là ai
   - Có thể suy luận gián tiếp (không cần bằng chứng tuyệt đối)

2. Phân tích CÁCH GIAO TIẾP (interaction pattern):
   - Ai đang hỏi / ai đang trả lời
   - Ai chủ động dẫn dắt cuộc hội thoại (có thể là host, sếp, interviewer)
   - Quan hệ xưng hô:
     - "anh - em", "chị - em", "sếp - nhân viên"
   → Dùng các pattern này để suy luận danh tính chính xác hơn

3. Phân biệt rõ:
   - Người nói ≠ người được gọi tên
   - Ví dụ:
     - "Anh Long ơi" → người nói KHÔNG phải Long

4. Xử lý tiếng Việt:
   - "anh", "chị", "em", "sếp", "bạn"... chỉ là đại từ
   → KHÔNG phải tên
   → Chỉ dùng để hỗ trợ suy luận

5. Nếu không thể xác định chắc chắn tên:
   → Suy luận ROLE dựa trên hành vi:
   - Ví dụ:
     - "Khách hàng", "Nhân viên", "Interviewer", "Host", "Sếp"
   - Role phải hợp lý với cách họ nói chuyện

6. CHỈ dùng "Unknown_X" khi:
   - Không thể suy luận được cả tên lẫn role
   - Không có đủ ngữ cảnh để đưa ra giả định hợp lý

7. Tính nhất quán:
   - Mỗi speaker chỉ có 1 label duy nhất
   - Không thay đổi tên giữa các đoạn hội thoại

8. Ưu tiên phương án hợp lý nhất:
   - Ngay cả khi không chắc chắn 100%, hãy chọn phương án có khả năng cao nhất dựa trên toàn bộ ngữ cảnh

## THỨ TỰ ƯU TIÊN

1. Tên cụ thể (ví dụ: Long, Bình, Hoa)
2. Role hợp lý (ví dụ: Khách hàng, Nhân viên)
3. Unknown_X (chỉ khi bất khả kháng)

## OUTPUT FORMAT

- Chỉ output JSON hợp lệ
- Key = speaker index (string)
- Value = tên hoặc role

Ví dụ:
{
  "0": "Long",
  "1": "Khách hàng",
  "2": "Nhân viên"
}

## LƯU Ý

- Không giải thích
- Không thêm text ngoài JSON
- Không sửa nội dung transcript
- Chỉ map speaker → tên hoặc role"""


def call_deepgram_api(
    audio_path: str,
    api_key: str,
    model: str = "nova-3",
    language: str = "vi",
    diarize: bool = True,
    punctuate: bool = True,
    utterances: bool = True,
) -> tuple[dict, float]:
    """Call Deepgram API with optimal configuration for Vietnamese."""
    url = "https://api.deepgram.com/v1/listen"

    params = {
        "model": model,
        "language": language,
        "diarize": "true" if diarize else "false",
        "punctuate": "true" if punctuate else "false",
        "utterances": "true" if utterances else "false",
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mpeg" if audio_path.lower().endswith(".mp3") else "audio/wav",
    }

    logger.info("Calling Deepgram for %s with model=%s language=%s", audio_path, model, language)
    started_at = time.perf_counter()

    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(url, params=params, headers=headers, data=audio_file, timeout=300)

        response.raise_for_status()
        elapsed_seconds = time.perf_counter() - started_at
        logger.info("Deepgram request completed successfully in %.2fs", elapsed_seconds)
        return response.json(), elapsed_seconds

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {audio_path}")
    except requests.exceptions.RequestException as exc:
        error_msg = f"Deepgram API request failed: {exc}"
        if hasattr(exc, "response") and exc.response is not None:
            error_msg += f"\nResponse: {exc.response.text}"
        raise Exception(error_msg) from exc


def format_utterances(data: dict, min_confidence: float = 0.0) -> list[str]:
    """Format utterances from JSON response into [Speaker:X] Text format."""
    output_lines: list[str] = []

    try:
        results = data.get("results", {})
        utterances = results.get("utterances", [])

        if not utterances:
            channels = results.get("channels", [])
            if channels and channels[0].get("alternatives"):
                words = channels[0]["alternatives"][0].get("words", [])
                return format_from_words(words, min_confidence)

        for utt in utterances:
            confidence = utt.get("confidence", 0)
            if confidence < min_confidence:
                continue

            speaker = utt.get("speaker", "unknown")
            transcript = utt.get("transcript", "").strip()

            if transcript:
                output_lines.append(f"[Speaker:{speaker}] {transcript}")

    except (KeyError, TypeError, IndexError) as exc:
        logger.warning("Error parsing Deepgram utterances: %s", exc)

    return output_lines


def format_from_words(words: list[dict], min_confidence: float = 0.0) -> list[str]:
    """Fallback: format from words array when utterances are not available."""
    if not words:
        return []

    output_lines: list[str] = []
    current_speaker = None
    current_sentence: list[str] = []

    for word_data in words:
        confidence = word_data.get("confidence", 0)
        if confidence < min_confidence:
            continue

        speaker = word_data.get("speaker", "unknown")
        word = word_data.get("punctuated_word") or word_data.get("word", "")

        if speaker != current_speaker and current_sentence:
            text = " ".join(current_sentence).strip()
            if text:
                output_lines.append(f"[Speaker:{current_speaker}] {text}")
            current_sentence = []

        current_speaker = speaker
        current_sentence.append(word)

    if current_sentence:
        text = " ".join(current_sentence).strip()
        if text:
            output_lines.append(f"[Speaker:{current_speaker}] {text}")

    return output_lines


def extract_speaker_ids(lines: list[str]) -> list[str]:
    """Get ordered unique speaker ids from transcript lines."""
    seen = set()
    speaker_ids = []

    for line in lines:
        match = re.match(r"^\[Speaker:(.+?)\]\s*", line)
        if not match:
            continue
        speaker_id = match.group(1)
        if speaker_id not in seen:
            seen.add(speaker_id)
            speaker_ids.append(speaker_id)

    return speaker_ids


def count_recognized_words(lines: list[str]) -> int:
    """Count recognized transcript words, excluding speaker labels."""
    word_count = 0

    for line in lines:
        text = re.sub(r"^\[Speaker:.+?\]\s*", "", line).strip()
        if text:
            word_count += len(text.split())

    return word_count


def build_claude_user_prompt(numbered_transcript: str, speaker_ids: list[str]) -> str:
    """Build user prompt for Claude speaker naming."""
    speaker_lines = "\n".join(f'- "{speaker_id}"' for speaker_id in speaker_ids)
    return (
        "Determine the speaker names from this transcript.\n\n"
        "Speaker indexes present:\n"
        f"{speaker_lines}\n\n"
        "Transcript:\n"
        f"{numbered_transcript}\n"
    )


def extract_json_object(text: str) -> str:
    """Extract a JSON object from Claude output, including fenced output."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Claude response")

    return cleaned[start : end + 1]


def normalize_speaker_mapping(raw_mapping: dict, speaker_ids: list[str]) -> dict[str, str]:
    """Ensure every speaker id has a safe output label."""
    normalized: dict[str, str] = {}

    for speaker_id in speaker_ids:
        raw_value = raw_mapping.get(speaker_id, raw_mapping.get(str(speaker_id)))
        label = str(raw_value).strip() if raw_value is not None else ""

        if not label:
            label = f"Unknown_{speaker_id}"

        normalized[speaker_id] = label

    return normalized


def get_claude_model_candidates() -> list[str]:
    """Return ordered Haiku-only Claude model candidates to try."""
    candidates = [
        CLAUDE_MODEL,
        "claude-3-5-haiku-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
    ]

    deduped = []
    seen = set()
    for model_name in candidates:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


def _call_claude_once(model_name: str, numbered_transcript: str, speaker_ids: list[str]) -> tuple[dict[str, str], dict, float]:
    """Call Claude API once with a specific model."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model_name,
        "max_tokens": 512,
        "system": CLAUDE_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": build_claude_user_prompt(numbered_transcript, speaker_ids),
            }
        ],
    }
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    logger.info("Calling Claude speaker naming with model=%s for speakers=%s", model_name, speaker_ids)
    started_at = time.perf_counter()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        error_msg = f"Claude API request failed: {exc}"
        if hasattr(exc, "response") and exc.response is not None:
            error_msg += f"\nResponse: {exc.response.text}"
        raise Exception(error_msg) from exc

    content_blocks = data.get("content", [])
    text_parts = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    response_text = "\n".join(text_parts).strip()
    elapsed_seconds = time.perf_counter() - started_at
    usage = data.get("usage", {}) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    logger.info(
        "Claude speaker naming response received in %.2fs | input_tokens=%s output_tokens=%s total_tokens=%s",
        elapsed_seconds,
        input_tokens,
        output_tokens,
        total_tokens,
    )

    try:
        mapping = json.loads(extract_json_object(response_text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise Exception(f"Claude returned invalid JSON speaker mapping: {response_text}") from exc

    if not isinstance(mapping, dict):
        raise Exception(f"Claude returned non-object speaker mapping: {mapping}")

    return normalize_speaker_mapping(mapping, speaker_ids), usage, elapsed_seconds


def call_claude_speaker_naming(numbered_transcript: str, speaker_ids: list[str]) -> tuple[dict[str, str], str, dict, float]:
    """Call Claude API to infer speaker names from transcript with model fallback."""
    last_error = None

    for model_name in get_claude_model_candidates():
        try:
            mapping, usage, elapsed_seconds = _call_claude_once(model_name, numbered_transcript, speaker_ids)
            return mapping, model_name, usage, elapsed_seconds
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            is_missing_model = (
                "not_found_error" in error_text
                or '"message":"model:' in error_text
                or "model:" in error_text
            )

            if is_missing_model:
                logger.warning("Claude model unavailable, trying fallback: %s", model_name)
                continue

            raise

    raise Exception(f"All Claude model candidates failed. Last error: {last_error}") from last_error


def remap_transcript_lines(lines: list[str], speaker_mapping: dict[str, str]) -> list[str]:
    """Replace [Speaker:X] labels with inferred names."""
    remapped_lines = []

    for line in lines:
        match = re.match(r"^\[Speaker:(.+?)\]\s*(.*)$", line)
        if not match:
            remapped_lines.append(line)
            continue

        speaker_id = match.group(1)
        transcript = match.group(2).strip()
        speaker_name = speaker_mapping.get(speaker_id, f"Unknown_{speaker_id}")
        remapped_lines.append(f"{speaker_name}: {transcript}")

    return remapped_lines


async def process_audio_task(task_id: str, file_path: str, model: str, language: str):
    """Background task to process audio, infer speaker names, and save outputs."""
    logger.info("Task %s started for file=%s", task_id, file_path)
    task_started_at = time.perf_counter()

    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10

        try:
            from mutagen.flac import FLAC
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            from mutagen.oggvorbis import OggVorbis
            from mutagen.wave import WAVE

            audio = None
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".mp3":
                audio = MP3(file_path)
            elif file_ext == ".m4a":
                audio = MP4(file_path)
            elif file_ext == ".flac":
                audio = FLAC(file_path)
            elif file_ext == ".ogg":
                audio = OggVorbis(file_path)
            elif file_ext == ".wav":
                audio = WAVE(file_path)

            if audio is not None and hasattr(audio, "info") and hasattr(audio.info, "length"):
                duration = audio.info.length
                tasks[task_id]["duration_seconds"] = duration

                if duration > MAX_AUDIO_DURATION_SECONDS:
                    raise Exception(
                        f"Audio duration ({duration / 60:.1f} minutes) exceeds the maximum limit of "
                        f"{MAX_AUDIO_DURATION_SECONDS / 60:.0f} minutes. Please use a shorter audio file."
                    )

                tasks[task_id]["duration_formatted"] = f"{duration / 60:.1f} minutes"
                logger.info("Task %s duration detected: %s", task_id, tasks[task_id]["duration_formatted"])
        except Exception as exc:
            if "exceeds the maximum limit" in str(exc):
                raise exc
            logger.warning("Task %s could not check duration: %s", task_id, exc)

        tasks[task_id]["progress"] = 25

        response_data, deepgram_elapsed_seconds = call_deepgram_api(
            audio_path=file_path,
            api_key=DEEPGRAM_API_KEY,
            model=model,
            language=language,
        )

        tasks[task_id]["progress"] = 60
        numbered_lines = format_utterances(response_data, min_confidence=0.0)
        numbered_output_text = "\n".join(numbered_lines)
        deepgram_word_count = count_recognized_words(numbered_lines)

        speakers = extract_speaker_ids(numbered_lines)
        num_speakers = len(speakers)
        tasks[task_id]["num_speakers"] = num_speakers
        logger.info("Task %s Deepgram found %s speakers: %s", task_id, num_speakers, speakers)
        logger.info("Task %s Deepgram recognized %s words", task_id, deepgram_word_count)

        if num_speakers > RECOMMENDED_MAX_SPEAKERS:
            tasks[task_id]["warning"] = (
                f"Detected {num_speakers} speakers. For best accuracy, "
                f"we recommend audio with {RECOMMENDED_MAX_SPEAKERS} or fewer speakers."
            )

        tasks[task_id]["progress"] = 75

        speaker_mapping = normalize_speaker_mapping({}, speakers)
        claude_model_used = None
        claude_usage = {}
        claude_elapsed_seconds = 0.0
        if speakers:
            if not CLAUDE_API_KEY:
                raise Exception("CLAUDE_API_KEY not configured")
            (
                speaker_mapping,
                claude_model_used,
                claude_usage,
                claude_elapsed_seconds,
            ) = call_claude_speaker_naming(numbered_output_text, speakers)
        logger.info("Task %s speaker mapping: %s", task_id, speaker_mapping)
        if claude_model_used:
            logger.info("Task %s Claude model used: %s", task_id, claude_model_used)

        total_elapsed_seconds = time.perf_counter() - task_started_at
        logger.info(
            "Task %s summary | deepgram_words=%s deepgram=%.2fs claude=%.2fs total=%.2fs",
            task_id,
            deepgram_word_count,
            deepgram_elapsed_seconds,
            claude_elapsed_seconds,
            total_elapsed_seconds,
        )

        named_lines = remap_transcript_lines(numbered_lines, speaker_mapping)
        named_output_text = "\n".join(named_lines)

        tasks[task_id]["progress"] = 95

        output_dir = Path(tempfile.gettempdir())
        named_file = output_dir / f"{task_id}_named.txt"
        numbered_file = output_dir / f"{task_id}_numbered.txt"
        json_file = output_dir / f"{task_id}.json"

        named_file.write_text(named_output_text, encoding="utf-8")
        numbered_file.write_text(numbered_output_text, encoding="utf-8")
        json_file.write_text(json.dumps(response_data, ensure_ascii=False, indent=2), encoding="utf-8")

        tasks[task_id]["progress"] = 100
        tasks[task_id]["status"] = "done"
        tasks[task_id]["result"] = {
            "text_file": str(named_file),
            "named_text_file": str(named_file),
            "numbered_text_file": str(numbered_file),
            "json_file": str(json_file),
            "output_text": named_output_text,
            "named_output_text": named_output_text,
            "numbered_output_text": numbered_output_text,
            "speaker_mapping": speaker_mapping,
            "claude_model_used": claude_model_used,
            "claude_usage": {
                "input_tokens": claude_usage.get("input_tokens", 0),
                "output_tokens": claude_usage.get("output_tokens", 0),
                "total_tokens": claude_usage.get("input_tokens", 0) + claude_usage.get("output_tokens", 0),
            },
            "processing_time": {
                "total_seconds": round(total_elapsed_seconds, 2),
                "deepgram_seconds": round(deepgram_elapsed_seconds, 2),
                "claude_seconds": round(claude_elapsed_seconds, 2),
            },
            "duration": response_data.get("metadata", {}).get("duration", 0),
            "duration_formatted": tasks[task_id].get("duration_formatted"),
            "language": language,
            "num_speakers": num_speakers,
            "warning": tasks[task_id].get("warning"),
        }
        logger.info("Task %s completed successfully", task_id)

    except Exception as exc:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = PUBLIC_PROCESSING_ERROR_MESSAGE
        tasks[task_id]["internal_error"] = str(exc)
        logger.exception("Task %s failed: %s", task_id, exc)

    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.info("Task %s cleaned up uploaded file %s", task_id, file_path)
        except Exception as exc:
            logger.warning("Task %s failed to clean up upload %s: %s", task_id, file_path, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for task_data in tasks.values():
        if "file_path" in task_data:
            try:
                Path(task_data["file_path"]).unlink(missing_ok=True)
            except Exception:
                pass


app = FastAPI(title="Hệ thống ghi chép tự động - Named Speakers", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = "nova-3",
    language: str = "vi",
    background_tasks: BackgroundTasks = None,
):
    """Upload audio file and start processing."""
    if not DEEPGRAM_API_KEY:
        logger.error("Upload rejected: DEEPGRAM_API_KEY not configured")
        raise HTTPException(status_code=500, detail=PUBLIC_UPLOAD_ERROR_MESSAGE)

    if not CLAUDE_API_KEY:
        logger.error("Upload rejected: CLAUDE_API_KEY not configured")
        raise HTTPException(status_code=500, detail=PUBLIC_UPLOAD_ERROR_MESSAGE)

    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        logger.warning("Upload rejected: invalid file type filename=%s ext=%s", file.filename, file_ext)
        raise HTTPException(status_code=400, detail=PUBLIC_UPLOAD_ERROR_MESSAGE)

    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    logger.info("Received upload task_id=%s filename=%s language=%s model=%s", task_id, file.filename, language, model)

    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as exc:
        logger.exception("Failed to save upload %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=PUBLIC_UPLOAD_ERROR_MESSAGE) from exc

    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "filename": file.filename,
        "language": language,
        "model": model,
        "file_path": str(file_path),
    }

    background_tasks.add_task(process_audio_task, task_id, str(file_path), model, language)

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "File uploaded successfully. Processing started.",
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task processing status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_data["status"],
        "progress": task_data.get("progress", 0),
        "filename": task_data.get("filename"),
        "result": task_data.get("result"),
        "error": task_data.get("error"),
    }


@app.get("/api/download/{task_id}/{file_type}")
async def download_result(task_id: str, file_type: str):
    """Download processed result file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]
    if task_data["status"] != "done":
        raise HTTPException(status_code=400, detail="Task not completed yet")

    result = task_data["result"]

    if file_type == "txt":
        return FileResponse(
            result["named_text_file"],
            filename=f"{task_data['filename']}_named_output.txt",
            media_type="text/plain",
        )
    if file_type == "numbered_txt":
        return FileResponse(
            result["numbered_text_file"],
            filename=f"{task_data['filename']}_numbered_output.txt",
            media_type="text/plain",
        )
    if file_type == "json":
        return FileResponse(
            result["json_file"],
            filename=f"{task_data['filename']}_response.json",
            media_type="application/json",
        )

    raise HTTPException(status_code=400, detail="Invalid file type. Use 'txt', 'numbered_txt', or 'json'")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api_key_configured": bool(DEEPGRAM_API_KEY),
        "deepgram_api_key_configured": bool(DEEPGRAM_API_KEY),
        "claude_api_key_configured": bool(CLAUDE_API_KEY),
        "claude_model": CLAUDE_MODEL,
        "claude_model_candidates": get_claude_model_candidates(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)
