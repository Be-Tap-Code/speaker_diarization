# Báo cáo — Speaker Diarization UI

## Tổng quan quy trình

Hệ thống nhận diện và phân loại người nói (speaker diarization) từ file audio đầu vào,
xuất ra file `.txt` (transcript theo người nói) và `.srt` (phụ đề có timestamp).

### Pipeline chi tiết

| Bước | Mô hình / Thư viện | Chức năng |
|---|---|---|
| **1. Tách giọng nói** | [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`) | Tách vocal ra khỏi nhạc/nền (có thể tắt bằng `--no-stem`) |
| **2. Transcription** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — **OpenAI Whisper large-v3** | Chuyển giọng nói thành văn bản + timestamp |
| **3. Forced Alignment** | [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | Căn chỉnh word-level timestamp chính xác hơn từ transcript thô |
| **4. Speaker Diarization** | NeMo **Sortformer** (hoặc MSDD) | Phân đoạn audio theo người nói |
| **5. Speaker Identification** (tùy chọn) | NeMo **Titanet Large** | So khớp đoạn âm thanh với reference audio có sẵn để gán tên người nói |
| **6. Punctuation Restoration** | [deepmultilingualpunctuation](https://github.com/kredor/deepmultilingualpunctuation) (`kredor/punctuate-all`) | Khôi phục dấu câu (`.?!`) để cải thiện phân tách câu/người nói |

### Công nghệ nền tảng

- **Framework:** PyTorch (CUDA)
- **Backend API:** FastAPI + Uvicorn
- **Frontend:** HTML/CSS/JS vanilla, gọi API qua `fetch`
- **Batch size:** 64 (mặc định)

---

## Hạn chế

### 1. Whisper nhận diện tiếng Việt chưa tốt

- Mô hình Whisper hiện tại (**large-v3**) được train đa ngôn ngữ nhưng **không tối ưu cho tiếng Việt**.
- Accuracy tiếng Việt ở mức trung bình — đặc biệt với âm thanh có nhiễu, nói nhanh, hoặc giọng địa phương.
- **Đã thử nghiệm PhoWhisper** (`vinai/PhoWhisper-large` / `vinai/PhoWhisper-medium`) — mô hình Whisper fine-tune riêng cho tiếng Việt của VinAI — nhưng **không đủ dung lượng bộ nhớ** để tải và chạy:
  - `PhoWhisper-large`: ~6.2GB → vượt disk space
  - `PhoWhisper-medium`: ~3.06GB → yêu cầu PyTorch >= 2.6 (hiện tại < 2.6)
- **Hiện tại:** Quay lại dùng `whisper large-v3` làm mặc định.

### 2. Độ chính xác phân tách speaker ~80%

- Mô hình diarization (Sortformer / MSDD) đạt độ chính xác khoảng **80%** trong điều kiện lý tưởng.
- Đây là **mô hình mã nguồn mở tốt nhất hiện có** — khó cải thiện thêm vì đây là giới hạn vốn có của model.
- **Phục vụ tốt nhất với 2–4 người.** Đây là **giới hạn kiến trúc** của mô hình:
  - Số lượng speaker tăng → accuracy giảm đáng kể
  - Các speaker nói chồng lấn, chen ngang → model dễ nhầm lẫn
  - Không xử lý tốt trường hợp >5 người hoặc hội thoại nhóm lớn

### 3. Yêu cầu âm thanh rõ ràng, đủ dài để nhận diện

- Để nhận diện chính xác giọng của một người nói, đoạn âm thanh cần **ít nhất ~10 giây nói rõ ràng, liên tục**.
- Lý do: mô hình **Titanet Large** cần trích xuất **speaker embedding** — vector đặc trưng giọng nói — từ audio.
  - Nếu đoạn nói quá ngắn → embedding không đủ thông tin → similarity score không chính xác → nhận diện sai hoặc không nhận diện được.
- Audio đầu vào cần:
  - **Rõ tiếng**, ít nhiễu nền
  - **Không nói chồng lấn** quá nhiều giữa các speaker
  - Mỗi speaker có **đủ nội dung liên tục** để trích xuất đặc trưng giọng

---

## Cấu hình hiện tại

| Tham số | Giá trị |
|---|---|
| Whisper model | `large-v3` |
| Diarizer | `Sortformer` |
| Language | `Vietnamese` (mặc định) |
| Batch size | `64` |
| Identify threshold | `0.45` |
| Device | `CUDA` |

---

## Đề xuất cải thiện (tương lai)

| Vấn đề | Hướng giải quyết |
|---|---|
| Transcription tiếng Việt | Nâng cấp PyTorch >= 2.6 để chạy PhoWhisper, hoặc thử [WhisperX](https://github.com/m-bain/whisperX) với model fine-tune tiếng Việt |
| Diarization accuracy >80% | Thử model thương mại (Deepgram, AssemblyAI) hoặc fine-tune NeMo trên dataset tiếng Việt |
| Speaker identification với audio ngắn | Thu thập reference audio dài hơn (>10s), chất lượng cao cho mỗi speaker |
