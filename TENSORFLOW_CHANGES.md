# Tích hợp TensorFlow vào Simple NotebookLM

## 1. TensorFlow áp dụng được ở đâu?

| Phần | Khả năng áp dụng | Trạng thái lần sửa này |
|---|---|---|
| Embedding tài liệu và câu hỏi | Chạy encoder bằng TensorFlow để tạo vector tìm kiếm | Đã triển khai |
| Mean pooling, chuẩn hóa vector | Tính vector câu từ token, bỏ padding, chuẩn hóa L2 | Đã triển khai bằng TensorFlow |
| Semantic chunking | So sánh embedding câu liền kề và xác định ranh giới bằng ngưỡng thích nghi | Đã tích hợp vào `build_chunks()` |
| Reranking | Cross-Encoder chấm điểm trực tiếp cặp câu hỏi–chunk bằng TensorFlow | Đã tích hợp tùy chọn vào `answer()`; mặc định tắt |
| Phân loại chủ đề | Có thể huấn luyện mạng Keras trên embedding | Chưa triển khai; cần dữ liệu gán nhãn |
| LLM sinh câu trả lời | Có thể thay bằng mô hình tương thích TensorFlow, nhưng cần trọng số và tài nguyên tương ứng | Vẫn dùng Gemini API |
| Đọc PDF, Qdrant, Jinja2, giao diện | Không cần TensorFlow; tiếp tục dùng các thư viện chuyên dụng | Giữ pipeline hiện có |

Đây là inference với model đã huấn luyện, không phải huấn luyện model mới. Cả bốn chức năng hỏi đáp, tóm tắt, quiz và flashcards sử dụng dữ liệu được index bằng backend mới. Với tác vụ lấy toàn bộ chunk, bước đó chỉ đọc Qdrant; TensorFlow đã được dùng khi index.

## 2. Thay đổi cụ thể

Các đường dẫn dưới đây nằm trong `simple_notebooklm/`.

| File | Đã sửa/thêm gì | Mục đích |
|---|---|---|
| `src/tensorflow_embeddings.py` | Thêm `TensorFlowEmbeddings`, cache model, xử lý batch, masked mean pooling, L2 normalization | Thực thi encoder và phép toán embedding bằng TensorFlow |
| `src/embeddings.py` | Chọn backend theo cấu hình; chỉ import SentenceTransformer khi dùng backend local | Kết nối backend mới với giao diện LangChain `Embeddings` mà Qdrant đang dùng |
| `src/config.py` | Mặc định `embedding_provider=tensorflow`, batch 16, giới hạn 512 token; đổi storage và collection mặc định | Bật TensorFlow và tách kho vector mới khỏi kho cũ |
| `app.py` | Hiển thị embedding backend trong khu vực thông tin model | Giúp xác định backend đang chạy |
| `requirements.txt` | Thêm TensorFlow 2.15.1, Transformers 4.49.0, torch; giới hạn Sentence Transformers dưới 5.2 | Dùng Transformers 4.x có hỗ trợ TFAutoModel và Keras 2 đi cùng TF 2.15 |
| `requirements-tensorflow.txt` | File cài đặt tham chiếu requirements chính | Cung cấp lệnh cài đặt riêng rõ ràng |
| `.env.example` | Mẫu cấu hình TensorFlow, đường dẫn kho vector, API key | Dễ thiết lập môi trường |
| `.gitignore` | Bỏ qua `.env` và môi trường ảo | Tránh đưa khóa API và thư viện cài đặt vào Git |
| `check_tensorflow.py` | Kiểm tra inference thật, mask padding, chiều vector, chuẩn hóa, batch/query consistency, tìm vector gần nhất | Xác minh chức năng embedding không cần gọi Gemini |

Không sửa notebook solution gốc. Không sửa nội dung prompt trong `src/prompts/`.

## 3. Model và đường đi dữ liệu

Vẫn dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, đầu ra 384 chiều. Cả hai backend dùng `embedding_max_length=512`, phù hợp với `max_position_embeddings=512` của encoder; Sentence Transformers được đặt lại `max_seq_length` để vượt giới hạn mặc định 128. Văn bản vượt 512 token vẫn bị cắt; nội dung chunk lưu trong Qdrant giữ nguyên. Sau khi đổi giới hạn, cần index lại tài liệu để cập nhật vector đã lưu. Pooling và chuẩn hóa L2 dùng toán tử TensorFlow trong `tf.function`; wrapper và model được cache trong tiến trình.

```text
PDF → chunk → tokenizer → TFAutoModel → masked mean pooling → L2 → Qdrant
Câu hỏi → cùng encoder TensorFlow → Qdrant tìm chunk → Jinja2 → Gemini → kết quả
```

`TFAutoModel.from_pretrained(..., use_safetensors=True)` đọc checkpoint `.safetensors`, nhận diện định dạng trọng số PyTorch từ metadata và chuyển sang TensorFlow. Không dùng `from_pt=True` vì đường tải đó tìm file `.bin` không có trong cache hiện tại. PyTorch vẫn là dependency của backend Sentence Transformers cũ; suy luận của backend mới chạy bằng TensorFlow. Model được cache trong bộ nhớ tiến trình; lần khởi động mới sẽ tải/chuyển lại từ cache Hugging Face, chưa xuất một checkpoint TensorFlow riêng ra đĩa.

Checkpoint gốc tiếp tục nằm trong cache Hugging Face, mặc định trên máy này là `C:\Users\LEGION\.cache\huggingface\hub`. Không tải một LLM local mới.

## 4. Cài đặt và chạy trên Windows

Dùng Python 3.11 và môi trường mới để không thay đổi `.venv` đang có:

```powershell
cd C:\Users\LEGION\Desktop\AIO\Mini_note_tensor\simple_notebooklm
py -3.11 -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install -r requirements-tensorflow.txt
$env:GEMINI_API_KEY = "API_KEY_CUA_BAN"
.\.venv-tf\Scripts\python.exe check_tensorflow.py
.\.venv-tf\Scripts\python.exe app.py
```
Có thể nhập Gemini API key qua giao diện thay vì biến môi trường. TensorFlow 2.15 trên Windows native dùng CPU; không mặc định tận dụng GPU NVIDIA như PyTorch CUDA. Để chạy TensorFlow GPU với phiên bản này cần môi trường phù hợp như WSL2.

Khi chưa có dữ liệu, upload/index PDF để tạo vector trong `storage/qdrant_tensorflow`, collection `rag_chunks_tensorflow`. Nếu đã ingest, chỉ refresh danh sách tài liệu và hỏi đáp, không cần ingest lại mỗi lần mở app. Kho `storage/qdrant` cũ không bị xóa hoặc chuyển đổi. `.env` hoặc biến môi trường đã đặt trước có thể ghi đè các mặc định.

Chỉ backend mặc định và mean pooling của model nêu trên được hỗ trợ theo cấu hình này; không tùy ý thay model cần pooling hoặc prompt riêng. Khi đổi model, max length hay cách pooling, hãy dùng kho/collection mới và index lại, kể cả khi số chiều không đổi.

Để quay về backend và kho cũ, khởi động app với:

```powershell
$env:RAG_EMBEDDING_PROVIDER = "local"
$env:RAG_STORAGE_DIR = "storage/qdrant"
$env:RAG_QDRANT_COLLECTION = "rag_chunks"
.\.venv-tf\Scripts\python.exe app.py
```

## 5. Kiểm tra và giới hạn

Đã chạy `check_tensorflow.py` thành công với checkpoint thật trong cache, ở chế độ offline trên Windows CPU, TensorFlow 2.15.1 và Transformers 4.49.0:

- Đầu ra `(2, 384)`, các giá trị hữu hạn và norm L2 bằng 1.
- Padding không ảnh hưởng mean pooling.
- Cùng văn bản cho vector nhất quán khi chạy riêng và theo batch.
- Tìm vector gần nhất bằng tích vô hướng trả đúng văn bản trùng với truy vấn.
- Đầu vào danh sách rỗng trả về danh sách rỗng.
- Kiểm tra cú pháp 14 file Python và `git diff --check` đều đạt.

Môi trường kiểm thử riêng là `.tf-validation/` ở thư mục workspace, dùng `--system-site-packages` để tận dụng TensorFlow đã có và cài Transformers 4.49 vào riêng môi trường đó. Không thay `.venv` của ứng dụng và không đổi thư viện Python toàn cục. Hướng dẫn ở trên dùng môi trường sạch cho người chạy app.

Các kiểm tra embedding ở trên không gọi Gemini. Sau đó đã kiểm tra Qdrant, Cross-Encoder và public flow `answer()` với Gemini thật thành công (xem mục 10). UI đã được kiểm tra đường gọi trong source; chưa đánh giá chất lượng bằng Ragas hoặc benchmark để kết luận reranker cải thiện chất lượng tổng thể.

## 6. Tài liệu tham khảo

- [Model card MiniLM: mean pooling, 384 chiều, giới hạn 128 token](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Transformers 4.49: tải và chuyển trọng số với from_pt](https://huggingface.co/docs/transformers/v4.49.0/main_classes/model)
- [TensorFlow: cài đặt, giới hạn GPU trên Windows native](https://www.tensorflow.org/install/pip)

## 7. Semantic chunking bằng TensorFlow

`src/semantic_chunker.py` bổ sung `TensorFlowSemanticChunker`. Constructor nhận embedding object từ bên ngoài; không tự tải thêm embedding model. `src/indexing.py` chọn recursive hoặc semantic trong `build_chunks()`, giữ nguyên metadata, document ID, chunk ID và bước ghi Qdrant.

Luồng xử lý cho từng `Document` (thường là một trang PDF):

```text
Text → regex tách câu → embedding các câu → tensor float32 [N, 384]
     → cosine giữa các câu liền kề → adaptive threshold → boundary_mask
     → ghép chunk theo boundary hoặc max size → gộp tiny chunks → Document
```

Phần tính toán tensor trong `_semantic_boundaries()`:

```python
left_embeddings = tensor[:-1]
right_embeddings = tensor[1:]
similarities = tf.reduce_sum(left_embeddings * right_embeddings, axis=1)
mean_similarity = tf.reduce_mean(similarities)
std_similarity = tf.math.reduce_std(similarities)
threshold = mean_similarity - tf.cast(self.threshold_k, tf.float32) * std_similarity
boundary_mask = similarities < threshold
```

Embedding đã chuẩn hóa L2 nên tích vô hướng là cosine similarity. Với N câu, hai tensor liền kề có shape `[N-1, 384]`, similarities và mask có shape `[N-1]`. Không dùng vòng lặp Python để tính similarity. `tf.where()` dùng trong test minh họa để in boundary index; class production đọc trực tiếp `boundary_mask` khi ghép câu.

Regex tách theo khoảng trắng sau `.`, `!`, `?`. Ghép chuỗi, kiểm tra độ dài và gộp tiny chunks dùng Python:

- Cắt trước câu mới khi có semantic boundary **hoặc** candidate chunk vượt `max_chunk_size`, tính cả khoảng trắng nối câu.
- Giữ nguyên câu đơn lẻ vượt max size; không cắt giữa câu.
- Chunk có `len(chunk.strip()) < min_chunk_size` được ưu tiên gộp vào chunk trước, nếu không được thì thử chunk sau. Chunk đầu chỉ có thể gộp về sau; chunk cuối chỉ có thể gộp về trước.
- Chỉ gộp khi kết quả không vượt max size; nếu không thể gộp thì giữ nguyên, không xóa text. Không gộp xuyên qua các `Document`/trang PDF.
- Document rỗng được bỏ qua; document chỉ có một câu không cần gọi embedding để xác định boundary.
- Mỗi chunk copy `metadata=dict(document.metadata)`; indexing tiếp tục chịu trách nhiệm thêm `chunk_id`.

`min_chunk_size` và `max_chunk_size` tính bằng **ký tự**, khác với `embedding_max_length` tính bằng **token**. Min size không phải bảo đảm mọi chunk đều đạt mức tối thiểu. `chunk_overlap` vẫn dùng cho recursive, không được áp dụng trong semantic chunker hiện tại.

## 8. Cross-Encoder reranking bằng TensorFlow

`src/tensorflow_reranker.py` bổ sung `TensorFlowReranker`, sử dụng model `cross-encoder/ms-marco-MiniLM-L6-v2`. Đây là model chấm relevance cho cặp query–passage; nó khác embedding model đa ngôn ngữ dùng cho retrieval. Tham khảo [model card Cross-Encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

```text
Query + chunk text → tokenizer theo cặp → TFAutoModelForSequenceClassification
                  → relevance logits → tf.concat → tf.math.top_k
                  → danh sách RetrievedChunk theo thứ tự mới
```

- Tải tokenizer bằng `AutoTokenizer`; chuyển checkpoint PyTorch sang TensorFlow bằng `from_pt=True`. Inference dùng `training=False`, batch mặc định 8, giới hạn tổng chuỗi token của cặp là 512.
- Model một output dùng scalar logit làm relevance score, điểm cao hơn được xếp trước. Đây không phải xác suất và không phải cosine similarity.
- Model nhiều output phải có đúng một class tên `relevant` hoặc `relevance` trong `config.id2label`; dùng logit tương ứng. Config mơ hồ như `LABEL_0`/`LABEL_1` bị từ chối bằng `ValueError`.
- TensorFlow kiểm tra shape và score hữu hạn; `tf.math.top_k` chỉ giữ tối đa số candidates thực tế.
- Trả lại chính các object `RetrievedChunk` ban đầu, giữ text, metadata và **score Qdrant gốc**. Relevance score chỉ dùng sắp thứ tự, không ghi đè field `score`.

`test_tensorflow_reranking.py` là thí nghiệm cũ tính lại cosine bằng `tf.linalg.matvec` với cùng embedding model. Nó phục vụ học tensor, không phải Cross-Encoder production. `test_tensorflow_cross_encoder.py` mới kiểm tra reranker theo cặp thật.

## 9. Tích hợp vào RAG và cấu hình

`src/rag.py` giữ `retrieve()` chỉ thực hiện embedding và Qdrant search. `answer()` gọi helper `prepare_context()`:

```text
Reranker OFF:
Question → retrieve(k) → context → prompt hiện tại → Gemini → RagAnswer + citations

Reranker ON (mặc định kích thước 10/5):
Question → retrieve(10) → Cross-Encoder → top 5 context
         → prompt hiện tại → Gemini → RagAnswer + citations
```

Khi ON, `k` truyền vào `answer()` là số chunks cuối cùng, ghi đè `reranker_top_k`. Số candidates yêu cầu từ Qdrant là `max(reranker_candidate_k, final_k)`. Retrieval rỗng trả về câu trả lời thiếu ngữ cảnh hiện có, không tải reranker hay gọi Gemini. Citations được tạo theo context sau reranking. Các chức năng gọi `retrieve()` trực tiếp không tự động được rerank; thay đổi này nằm ở đường hỏi đáp `answer()`.

`_get_reranker()` dùng lazy import và `@lru_cache(maxsize=1)`: khi tắt reranker không tải Cross-Encoder, khi cần thì tải và tái sử dụng trong tiến trình. Embedding vẫn dùng cache `get_embeddings()` và cache model riêng. Khởi động lại app sau khi đổi cấu hình để nạp setting/model mới.

Các setting có thể đặt trong `.env` hoặc biến môi trường (giữ prefix `RAG_`):

```dotenv
RAG_EMBEDDING_PROVIDER=tensorflow
RAG_EMBEDDING_MAX_LENGTH=512
RAG_CHUNKING_STRATEGY=semantic
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=200
RAG_SEMANTIC_THRESHOLD_K=1.0
RAG_SEMANTIC_MIN_CHUNK_SIZE=100
RAG_RERANKER_ENABLED=false
RAG_RERANKER_CANDIDATE_K=10
RAG_RERANKER_TOP_K=5
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RAG_RERANKER_BATCH_SIZE=8
```

Đổi `RAG_RERANKER_ENABLED=true` để bật reranking. Settings kiểm tra `reranker_candidate_k >= reranker_top_k`; constructor semantic chunker kiểm tra `0 <= min_chunk_size <= max_chunk_size`. `.env.example` là mẫu, file `.env` mới được Settings đọc. API key Gemini tiếp tục dùng cơ chế hiện có, không đưa key thật vào tài liệu.

App hiện tại là **Gradio**, entry point `app.py`. Tab hỏi đáp đi qua `_ask_chat()` → `_ask()` → `answer()`. Slider `Top-k retrieval` mặc định 6: cần đặt **5** trên UI nếu muốn flow 10 candidates → 5 final chunks.

```powershell
cd C:\Users\LEGION\Desktop\AIO\Mini_note_tensor\simple_notebooklm
$env:RAG_RERANKER_ENABLED = "true"
.\.venv-tf\Scripts\python.exe .\app.py
```

Mở URL in trong terminal, refresh/chọn tài liệu đã index rồi hỏi đáp. Không cần upload hoặc rebuild dữ liệu chỉ để bật reranker.

## 10. Các test bổ sung và kết quả đã kiểm tra

Các lần kiểm tra bổ sung dưới đây chạy bằng `.venv-tf`, dùng source/pipeline thật ở những bước được mô tả. Đây là kết quả của PDF và cấu hình tại thời điểm kiểm tra, không phải số lượng được hard-code vào production.

| File test | Nội dung | Tác động |
|---|---|---|
| `test_tensorflow_embedding.py` | Tensor embedding 3 câu, shape, cosine TensorFlow | Không gọi Qdrant/Gemini |
| `test_semantic_similarity.py` | Similarity liền kề, ngưỡng thích nghi, `tf.where`, boundary và max size | Minh họa độc lập |
| `test_semantic_chunker.py` | Class chunker thật, hai chủ đề, bảo toàn metadata | Không ingest |
| `test_semantic_indexing.py` | `build_chunks()` trên PDF thật, metadata và thống kê tiny chunks | Không ghi Qdrant |
| `test_semantic_ingest.py` | `ingest(recreate=True)` và đếm point chính xác | **Rebuild collection đang cấu hình** |
| `test_semantic_retrieval.py` | `retrieve()` và schema kết quả hiện tại | Đọc Qdrant, không Gemini |
| `test_semantic_rag.py` | `answer()` theo config hiện tại | Gọi Gemini thật |
| `test_tensorflow_reranking.py` | Cosine bằng `tf.linalg.matvec`, xếp hạng bằng `tf.math.top_k` | Đọc Qdrant, không Gemini |
| `test_tensorflow_cross_encoder.py` | Top 10 candidates → Cross-Encoder → top 5 | Đọc Qdrant, không Gemini |
| `test_rag_reranker_integration.py` | Đường `prepare_context()` thật, cache, OFF/empty, metadata, citations | Không gọi Gemini; kiểm tra wiring LLM dùng stub |
| `test_rag_reranker_e2e.py` | Public `answer()` với reranker bật và Gemini thật | Không ingest; đóng Qdrant client sau test |

Với `[Description]-LLMs-Fine-tuning.pdf`, sau khi bổ sung min size 100:

```text
Number of chunks: 57
Minimum chunk length: 42
Maximum chunk length: 1491
Average chunk length: 735.877193
Tiny chunks (<100 chars): 1
```

Rebuild đã xác nhận **57 chunks ingest và 57 points thực tế**. `ingest()` mặc định chỉ upsert, không dọn point cũ khi cách chia chunk thay đổi. File `test_semantic_ingest.py` dùng `recreate=True` và workaround đóng handle collection local trước recreate để tránh khóa SQLite trên Windows; workaround truy cập API nội bộ Qdrant chỉ nằm trong test. Không chạy file này nếu chỉ muốn kiểm tra retrieval/UI.

Test cross-encoder và integration đã PASS. Test end-to-end gọi Gemini thật cũng PASS: lấy 10 candidates, giữ 5 context theo thứ tự Qdrant cũ **2 → 1 → 7 → 8 → 5**. Text, metadata, score Qdrant giữ nguyên; citations S1–S5 tương ứng context mới. Gemini trả lời định nghĩa fine-tuning và dẫn `[S3]`, là chunk trang 4 chứa định nghĩa. Kết quả này xác nhận flow hoạt động, chưa chứng minh mức cải thiện chất lượng trên tập câu hỏi rộng.

Chạy kiểm tra không gọi Gemini:

```powershell
cd C:\Users\LEGION\Desktop\AIO\Mini_note_tensor\simple_notebooklm
.\.venv-tf\Scripts\python.exe .\test_rag_reranker_integration.py
```

Chạy end-to-end với Gemini API key qua `.env`/environment hiện có:

```powershell
.\.venv-tf\Scripts\python.exe .\test_rag_reranker_e2e.py
```

Test tự bật reranker trong tiến trình, không sửa `.env`. Dòng kết thúc khi đạt:

```text
TensorFlow reranked RAG end-to-end PASSED
```
