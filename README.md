---
title: AIO2025M12 Project RAG Learning System
emoji: 📉
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# Simple NotebookLM — RAG với TensorFlow

Ứng dụng Gradio giúp học từ tài liệu PDF: hỏi đáp có nguồn trích dẫn, tóm tắt, tạo quiz, flashcards và xuất kết quả. PDF được lưu và index vào Qdrant local; Gemini sinh câu trả lời từ context đã truy xuất.

## Thành phần chính

| Thành phần | Implementation |
|---|---|
| UI | Gradio, entry point `app.py` |
| PDF | `PyPDFLoader` / PyPDF |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, vector 384 chiều, chuẩn hóa L2 |
| Backend embedding | TensorFlow mặc định; có thể chọn Sentence Transformers bằng `local` |
| Chunking | Semantic bằng TensorFlow mặc định, hoặc recursive |
| Vector store | Qdrant local trên đĩa, không cần chạy Qdrant server |
| Reranking tùy chọn | TensorFlow Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Sinh nội dung | Gemini qua `google-genai`; prompt Jinja2 |

```text
PDF → đọc từng trang → semantic/recursive chunks → embedding → Qdrant

Câu hỏi → embedding → Qdrant candidates
        → Cross-Encoder nếu bật → final context → prompt → Gemini → answer + citations
```

Semantic chunking tính cosine giữa câu liền kề bằng phép toán TensorFlow vector hóa, cắt theo ngưỡng `mean − k × std` hoặc max size, sau đó gộp tiny chunks khi có thể. Cross-Encoder nhận query và chunk cùng lúc, dùng relevance logits để đổi thứ tự; không ghi đè score Qdrant gốc.

## 1. Cài đặt trên Windows / PowerShell

Dùng **Python 3.11** cho bộ dependency hiện tại. Chạy các lệnh từ thư mục chứa `app.py`:

```powershell
cd C:\Users\LEGION\Desktop\AIO\Mini_note_tensor\simple_notebooklm
py -3.11 -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tf\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-tf\Scripts\python.exe -m pip check
```

Nếu `.venv-tf` đã tồn tại thì bỏ qua bước tạo môi trường. `requirements-tensorflow.txt` tham chiếu cùng `requirements.txt`, không phải bộ thư viện riêng.

TensorFlow được giữ ở 2.15.1 và Transformers ở 4.49.0 để phù hợp code `TFAutoModel`/Keras hiện tại. PyTorch vẫn cần cho backend `local` và chuyển checkpoint sang TensorFlow. NumPy dùng nhánh 1.x; không nâng riêng lên 2.x trong môi trường này. TensorFlow 2.15 trên Windows native chạy CPU.

Lần đầu cần Internet để tải checkpoint Hugging Face. Những lần sau dùng cache model; Gemini vẫn cần kết nối Internet. Chỉ định đường dẫn Python như trên giúp tránh dùng nhầm môi trường đang activate trong terminal.

## 2. Cấu hình `.env`

Nếu chưa có `.env`, sao chép mẫu **một lần**:

```powershell
Copy-Item .env.example .env
```

Điền key vào `.env`, không commit key thật:

```dotenv
GEMINI_API_KEY=your_api_key
```

Ứng dụng đọc `.env` từ thư mục làm việc thông qua Pydantic Settings; biến môi trường có thể ghi đè giá trị trong file. `.env.example` chỉ là mẫu. Có thể nhập key trong UI; key nhập ở đó được ưu tiên cho yêu cầu hiện tại, nếu để trống thì dùng cấu hình sẵn có.

Các cấu hình chính và mặc định:

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `RAG_DATA_DIR` | `data` | Thư mục PDF |
| `RAG_STORAGE_DIR` | `storage/qdrant_tensorflow` | Kho Qdrant local |
| `RAG_QDRANT_COLLECTION` | `rag_chunks_tensorflow` | Collection sử dụng |
| `RAG_EMBEDDING_PROVIDER` | `tensorflow` | `tensorflow` hoặc `local` |
| `RAG_EMBEDDING_MAX_LENGTH` | `512` | Giới hạn token embedding |
| `RAG_EMBEDDING_BATCH_SIZE` | `16` | Batch embedding TensorFlow |
| `RAG_CHUNKING_STRATEGY` | `semantic` | `semantic` hoặc `recursive` |
| `RAG_CHUNK_SIZE` | `1500` | Kích thước chunk theo ký tự |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap cho recursive |
| `RAG_SEMANTIC_THRESHOLD_K` | `1.0` | Hệ số trong adaptive threshold |
| `RAG_SEMANTIC_MIN_CHUNK_SIZE` | `100` | Ngưỡng gộp tiny chunks |
| `RAG_RERANKER_ENABLED` | `false` | Bật Cross-Encoder trong `answer()` |
| `RAG_RERANKER_CANDIDATE_K` | `10` | Số candidates trước reranking |
| `RAG_RERANKER_TOP_K` | `5` | Số context cuối nếu không truyền `k` |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Model relevance |
| `RAG_RERANKER_BATCH_SIZE` | `8` | Batch Cross-Encoder |
| `RAG_LLM_MODEL` | `gemini-flash-lite-latest` | Gemini model hiện tại |

`reranker_candidate_k` phải lớn hơn hoặc bằng `reranker_top_k`; semantic min size phải nằm trong khoảng từ 0 đến max size. Semantic chunker giữ nguyên câu quá dài và có thể giữ tiny chunk khi không thể gộp trong giới hạn. Giới hạn ký tự chunk không bảo đảm văn bản nằm trong giới hạn token của encoder.

## 3. Chạy UI

```powershell
.\.venv-tf\Scripts\python.exe .\app.py
```

Mở URL Gradio được in trong terminal, thường là `http://127.0.0.1:7860`.

- Nếu đã ingest: refresh danh sách tài liệu, chọn PDF rồi dùng tab hỏi đáp.
- Nếu chưa có dữ liệu: upload PDF và bấm **Nạp & Index**; sau đó chọn tài liệu để hỏi đáp hoặc tạo nội dung học tập.
- Có thể lọc theo trang khi chọn một tài liệu.

Để bật Cross-Encoder mà không sửa code:

```powershell
$env:RAG_RERANKER_ENABLED = "true"
.\.venv-tf\Scripts\python.exe .\app.py
```

Tab hỏi đáp gọi `_ask_chat()` → `_ask()` → `src.rag.answer()` → `prepare_context()`. Khi bật reranker, slider **Top-k retrieval** là số context cuối cùng. Slider hiện mặc định **6**; đặt **5** để chạy flow 10 candidates → 5 final chunks. Nếu truyền `k` lớn hơn candidate count cấu hình, retrieval yêu cầu `max(candidate_k, k)` candidates.

Khi tắt reranker, flow retrieval cũ được giữ nguyên và Cross-Encoder không được tải. Khi bật, model được lazy load và cache trong tiến trình. Reranking chỉ tích hợp vào đường `answer()`; các chức năng học tập gọi retrieval riêng không tự động được rerank. Khởi động lại app sau khi đổi `.env`.

## 4. Ingest và rebuild

UI sử dụng `save_and_ingest_pdf()`. Hàm `src.indexing.ingest()` dùng các PDF nằm trực tiếp trong thư mục `data` (không quét thư mục con), rồi gọi pipeline hiện có:

```text
discover_pdfs → build_chunks → embedding → index_chunks → Qdrant
```

Để ingest từ `data` theo chế độ mặc định, không recreate collection:

```powershell
.\.venv-tf\Scripts\python.exe -c "from src.indexing import ingest; from src.store import close_client; print('Number of chunks ingested:', ingest()); close_client()"
```

Mặc định là **upsert theo ID**, không tự xóa chunks cũ khi đổi cách chia chunk. Nếu cần rebuild sạch sau thay đổi chunking, dùng script có sẵn:

```powershell
.\.venv-tf\Scripts\python.exe .\test_semantic_ingest.py
```

**Lệnh này xóa và tạo lại toàn bộ collection đang cấu hình**, rồi ingest PDF trong `data`. Script gọi `ingest(recreate=True)`, kiểm tra số point thực tế bằng số chunks ingest và có xử lý handle SQLite của Qdrant local trên Windows. Không dùng script rebuild chỉ để mở UI hoặc kiểm tra retrieval.

Dừng app và các tiến trình dùng cùng kho Qdrant local trước khi chạy script ingest/test độc lập để tránh khóa storage. Đổi reranker không đòi hỏi ingest lại; đổi embedding hoặc chunking thì cần cập nhật dữ liệu tương ứng.

## 5. Chạy kiểm tra

Các test là script Python chạy trực tiếp, không cần pytest. Các test đọc Qdrant cần dữ liệu đã ingest trong collection hiện tại.

| Script | Kiểm tra | Gọi Gemini / ghi dữ liệu |
|---|---|---|
| `check_tensorflow.py` | Embedding thật, pooling, padding, L2, kiểu dữ liệu | Không |
| `test_tensorflow_embedding.py` | Tensor embedding và cosine của 3 câu | Không |
| `test_semantic_similarity.py` | Boundary, adaptive threshold, max size | Không |
| `test_semantic_chunker.py` | Class chunker và metadata | Không |
| `test_semantic_indexing.py` | PDF → chunks, thống kê min/max/tiny chunks | Không ghi Qdrant |
| `test_semantic_ingest.py` | Rebuild và đếm points | **Rebuild Qdrant** |
| `test_semantic_retrieval.py` | Retrieval trên dữ liệu đã ingest | Chỉ đọc Qdrant |
| `test_tensorflow_reranking.py` | Thí nghiệm cosine TensorFlow, không phải cross-encoder | Chỉ đọc Qdrant |
| `test_tensorflow_cross_encoder.py` | Rerank 10 candidates, giữ 5 | Chỉ đọc Qdrant |
| `test_rag_reranker_integration.py` | Context production, cache và citations wiring | Không gọi Gemini |
| `test_semantic_rag.py` | `answer()` theo config hiện tại | **Gọi Gemini** |
| `test_rag_reranker_e2e.py` | Public `answer()` với reranker bật | **Gọi Gemini** |

Kiểm tra embedding và integration không gọi Gemini:

```powershell
$env:RAG_EMBEDDING_PROVIDER = "tensorflow"
$env:RAG_CHUNKING_STRATEGY = "semantic"
.\.venv-tf\Scripts\python.exe .\check_tensorflow.py
.\.venv-tf\Scripts\python.exe .\test_rag_reranker_integration.py
```

Kiểm tra end-to-end, sử dụng Gemini API key hiện có:

```powershell
.\.venv-tf\Scripts\python.exe .\test_rag_reranker_e2e.py
```

Hai test reranker integration/e2e tự bật cấu hình reranker trong tiến trình, không sửa `.env`. E2E đã được chạy thành công với 10 candidates → 5 final chunks → Gemini → citations. Đây là kiểm tra chức năng, chưa phải benchmark chất lượng.

## 6. Cấu trúc project

```text
app.py                         # Gradio UI
src/config.py                  # Pydantic Settings / .env
src/embeddings.py              # Chọn và cache embedding backend
src/tensorflow_embeddings.py   # TF encoder, masked mean pooling, L2
src/semantic_chunker.py        # Semantic boundaries và gộp tiny chunks
src/indexing.py                # PDF loading, metadata, ingest
src/store.py                   # Qdrant local và vector store
src/tensorflow_reranker.py     # Cross-Encoder relevance ranking
src/rag.py                    # Retrieval, context, answer và citations
src/llm.py                    # Gemini API
src/learning.py               # Tóm tắt, quiz, flashcards
src/prompts/                  # Jinja2 prompts
src/export.py                 # Xuất kết quả
static/                       # CSS và hình ảnh UI
data/                         # PDF local (gitignored)
storage/                      # Qdrant local (gitignored)
exports/                      # File xuất (gitignored)
```

Chi tiết các phép toán tensor, kết quả kiểm tra và lịch sử tích hợp: [TENSORFLOW_CHANGES.md](TENSORFLOW_CHANGES.md).
