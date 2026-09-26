from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import gradio as gr

from src.config import settings
from src.export import export
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import save_and_ingest_pdf
from src.learning import generate_flashcards, generate_quiz, summarize
from src.llm import set_runtime_gemini_api_key
from src.rag import answer
from src.store import list_documents

_INFO_NOTE_HTML = """
<div class="info-card">
  <div class="info-card-title">⚠️ Lưu ý sử dụng</div>
  <ul class="info-card-list">
    <li>Đây là ứng dụng demo phục vụ mục đích học tập và minh hoạ cho bài toán RAG.</li>
    <li>Ứng dụng dùng <b>Gemini API</b>. Hãy nhập <b>Gemini API Key</b> trước khi chạy (key chỉ dùng trong phiên hiện tại, không lưu).</li>
    <li>Hãy tải PDF trước, đợi hệ thống index xong, rồi mới dùng các tab Hỏi đáp, Tóm tắt, Quiz và Flashcards.</li>
  </ul>
</div>
"""

_USAGE_MARKDOWN = """
1. **Tải PDF** ở khối bên trái rồi bấm **Nạp & Index**.  
2. **Chọn tài liệu** muốn học trong danh sách đã index.  
3. Dùng các tab để **hỏi đáp**, **tóm tắt**, **tạo quiz** hoặc **flashcards**.  
4. Nếu chỉ chọn đúng 1 tài liệu, bạn có thể lọc thêm theo **trang**.

**Mẹo:** Khi đặt câu hỏi rõ ràng theo chủ đề, kết quả RAG thường sát và dễ học hơn.
"""

_CSS = Path("static/style.css").read_text(encoding="utf-8")


def _img_b64(path: str) -> str:
    with open(path, "rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def _status_html(message: str) -> str:
    return f'<div class="status-bar">{message}</div>'

def _result_markdown() -> gr.Markdown:
    # Gradio versions differ in how they gate HTML in Markdown; try in priority order.
    for kwargs in ({"sanitize_html": False}, {"sanitize": False}, {"unsafe_allow_html": True}, {}):
        try:
            return gr.Markdown(elem_classes="result-markdown", **kwargs)
        except TypeError:
            continue
    return gr.Markdown(elem_classes="result-markdown")

def _read_uploaded_pdf(file_obj: object) -> tuple[bytes, str]:
    """Normalize Gradio file payload into (bytes, filename).

    Gradio may pass:
    - `str` / `NamedString`: a local filepath
    - `FileData`: object with `.path` and optional `.orig_name`
    - `dict`: with keys like `path` / `orig_name`
    """
    if isinstance(file_obj, str):
        p = Path(file_obj)
        return p.read_bytes(), p.name

    path = getattr(file_obj, "path", None)
    orig = getattr(file_obj, "orig_name", None)
    if isinstance(path, str) and path:
        p = Path(path)
        name = str(orig).strip() if isinstance(orig, str) and orig.strip() else p.name
        return p.read_bytes(), name

    if isinstance(file_obj, dict):
        raw_path = file_obj.get("path")
        raw_name = file_obj.get("orig_name") or file_obj.get("name")
        if isinstance(raw_path, str) and raw_path:
            p = Path(raw_path)
            name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else p.name
            return p.read_bytes(), name

    raise TypeError(f"Unsupported uploaded file type: {type(file_obj).__name__}")

def _read_uploaded_pdfs(file_obj: object) -> list[tuple[bytes, str]]:
    """Normalize Gradio file payload into a list of (bytes, filename)."""
    if file_obj is None:
        return []
    if isinstance(file_obj, (list, tuple)):
        return [_read_uploaded_pdf(x) for x in file_obj]
    return [_read_uploaded_pdf(file_obj)]


def _filters(filenames: list[str] | None, page: int | None) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    if filenames:
        payload["filenames"] = filenames
    if page is not None:
        payload["page"] = page
    return filters_to_dict(MetadataFilter.model_validate(payload)) if payload else None


def _refresh_docs() -> tuple[object, dict[str, Any], object, str, str]:
    docs = list_documents()
    choices = [d["filename"] for d in docs]
    doc_map = {d["filename"]: d for d in docs}
    if docs:
        summary = (
            f"**{len(docs)}** tài liệu đã index · "
            f"**{sum(int(d['chunk_count']) for d in docs)}** đoạn văn"
        )
    else:
        summary = "Chưa có tài liệu nào được index."
    filenames_text = "\n".join(f"- `{name}`" for name in choices) if choices else "_Danh sách trống_"
    return (
        gr.update(choices=choices, value=[]),
        doc_map,
        gr.update(choices=["(Tất cả trang)"], value="(Tất cả trang)", interactive=True),
        summary,
        filenames_text,
    )


def _pages_for_selection(doc_map: dict[str, Any], selected: list[str]) -> gr.Dropdown:
    if len(selected) != 1:
        return gr.update(choices=["(Tất cả trang)"], value="(Tất cả trang)", interactive=False)
    doc = doc_map.get(selected[0]) or {}
    pages = doc.get("pages") or []
    page_choices = ["(Tất cả trang)", *[str(p) for p in pages]]
    return gr.update(choices=page_choices, value="(Tất cả trang)", interactive=True)


def _upload_pdf(
    file: object | None,
) -> tuple[str, object, dict[str, Any], object, str, str]:
    payloads = _read_uploaded_pdfs(file)
    if not payloads:
        choices, doc_map, page_dropdown, summary, filenames_text = _refresh_docs()
        return (
            _status_html("⚠️ Vui lòng chọn file PDF."),
            choices,
            doc_map,
            page_dropdown,
            summary,
            filenames_text,
        )

    successes: list[str] = []
    failures: list[str] = []
    chunks_total = 0
    for file_bytes, filename in payloads:
        try:
            info = save_and_ingest_pdf(file_bytes, filename)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{filename}: {e}")
            continue
        successes.append(str(info["filename"]))
        chunks_total += int(info.get("chunks_indexed") or 0)

    parts: list[str] = []
    if successes:
        parts.append(f"✅ Đã nạp {len(successes)} file · {chunks_total} đoạn")
    if failures:
        parts.append(f"⚠️ Lỗi **{len(failures)}** file")
    details = ""
    if failures:
        items = "".join(f"<li><code>{html.escape(x, quote=False)}</code></li>" for x in failures)
        details = f"<details><summary>Xem lỗi</summary><ul>{items}</ul></details>"
    message_body = (" · ".join(parts) if parts else "⚠️ Không có file hợp lệ.") + details
    message = _status_html(message_body)

    choices, doc_map, page_dropdown, summary, filenames_text = _refresh_docs()
    return message, choices, doc_map, page_dropdown, summary, filenames_text


def _ask(question: str, k: int, selected_docs: list[str], page: str, gemini_key: str) -> tuple[str, str]:
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", ""
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    res = answer(question.strip(), k=int(k), filters=_filters(selected_docs, page_num))
    return export(res, fmt="md"), json.dumps(res.model_dump(), ensure_ascii=False, indent=2)

def _ask_chat(
    message: str,
    history: list[dict[str, str]] | None,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
) -> tuple[list[dict[str, str]], str, str]:
    """Append a Q&A turn to the chatbot history."""
    user_text = (message or "").strip()
    hist: list[dict[str, str]] = list(history or [])
    if not user_text:
        return hist, "", ""
    answer_md, raw = _ask(user_text, k, selected_docs, page, gemini_key)
    hist.append({"role": "user", "content": user_text})
    hist.append({"role": "assistant", "content": answer_md})
    return hist, raw, ""


def _summarize(
    query: str,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    return _run_feature(
        summarize, selected_docs, page, gemini_key, progress,
        query=query.strip() or None, k=int(k),
    )


def _quiz(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    return _run_feature(
        generate_quiz, selected_docs, page, gemini_key, progress,
        query=query.strip() or None, count=int(count), k=int(k),
    )


def _flashcards(
    query: str,
    count: int,
    k: int,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    return _run_feature(
        generate_flashcards, selected_docs, page, gemini_key, progress,
        query=query.strip() or None, count=int(count), k=int(k),
    )


def _write_export(md_text: str, filename: str) -> str | None:
    if not md_text or md_text.startswith("Lỗi:"):
        return None
    output_path = Path("exports") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding="utf-8")
    return str(output_path)


def _run_feature(
    fn,
    selected_docs: list[str],
    page: str,
    gemini_key: str,
    progress: gr.Progress,
    **kwargs: object,
) -> tuple[str, str]:
    page_num = None if page == "(Tất cả trang)" else int(page)
    set_runtime_gemini_api_key(gemini_key)
    progress(0.3, desc="Đang truy xuất ngữ cảnh…")
    res = fn(**kwargs, filters=_filters(selected_docs, page_num))
    progress(0.9, desc="Đang định dạng kết quả…")
    return export(res, fmt="md"), res.model_dump_json(indent=2)


_theme = gr.themes.Base().set(
    background_fill_primary="#eef1fb",
    background_fill_secondary="#e4e9f7",
    block_background_fill="transparent",
    block_border_color="transparent",
    block_border_width="0px",
    input_background_fill="#ffffff",
)

with gr.Blocks(title="Building a Simple NotebookLM", fill_width=True, fill_height=True) as demo:
    with gr.Row(elem_classes="header-row"):
        logo_b64 = _img_b64("static/aivn_logo.png")
        gr.HTML(f'<img src="data:image/png;base64,{logo_b64}" alt="AIVN">', elem_classes="header-logo")
        gr.HTML(
            '<div class="header-meta">'
            '<p class="header-title">📚 Building a Simple NotebookLM</p>'
            '<p class="header-sub">AIO2025 — Hỏi đáp · Tóm tắt · Quiz · Flashcards có trích dẫn nguồn</p>'
            '</div>'
            ,
            elem_classes="header-text",
        )

    gr.HTML(_INFO_NOTE_HTML, elem_classes="site-info-note")

    doc_map_state = gr.State({})

    with gr.Row(equal_height=False, elem_classes="main-layout"):
        with gr.Column(scale=4, min_width=340, elem_classes="control-stack"):
            gr.Markdown("### 📥 Nạp tài liệu PDF")
            upload = gr.File(
                label="Chọn PDF",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
            )
            upload_btn = gr.Button("Nạp & Index", elem_classes="gen-btn")
            upload_status = gr.HTML(_status_html("Sẵn sàng."))

            with gr.Accordion("🔑 Gemini API Key", open=False):
                gr.Markdown(
                    "Nhập API key để chạy Gemini. Lấy key tại: "
                    "[Google AI Studio](https://aistudio.google.com/app/api-keys). ",
                    elem_classes="help-markdown",
                )
                gemini_key_input = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    lines=1,
                    max_lines=1,
                )

            with gr.Accordion("❓ Hướng dẫn sử dụng", open=False):
                gr.Markdown(_USAGE_MARKDOWN, elem_classes="help-markdown")

            with gr.Accordion("🔐 Cấu hình chạy trên Space", open=False):
                gr.Markdown(
                    f"""
                    - LLM model: `{settings.llm_model}`
                    - Embedding backend: `{settings.embedding_provider}`
                    - Embedding model: `{settings.embedding_model}`
                    - Collection: `{settings.qdrant_collection}`
                    - Data dir: `{settings.data_dir}`
                    - Storage dir: `{settings.storage_dir}`
                    """,
                    elem_classes="help-markdown",
                )

        with gr.Column(scale=7, min_width=560, elem_classes="preview-col"):
            gr.HTML(
                '<div class="preview-header"><div>'
                '<p class="preview-title">🗂️ Tài liệu đã index</p>'
                '<p class="preview-sub">Làm mới danh sách sau khi tải PDF, rồi chọn phạm vi học tập trước khi truy vấn.</p>'
                '</div></div>'
            )

            refresh_btn = gr.Button("Làm mới danh sách tài liệu")
            doc_summary = gr.Markdown("Chưa có tài liệu nào được index.", elem_classes="doc-summary")
            docs = gr.CheckboxGroup(label="Chọn tài liệu", choices=[], value=[])
            page = gr.Dropdown(
                label="Trang (chỉ áp dụng khi chọn đúng 1 tài liệu)",
                choices=["(Tất cả trang)"],
                value="(Tất cả trang)",
            )
            doc_list_md = gr.Markdown("_Danh sách trống_")

    with gr.Tabs():
        with gr.Tab("💬 Hỏi đáp"):
            chatbot = gr.Chatbot(elem_classes="qa-chat", height=520)
            gr.Markdown(
                "Nhập câu hỏi ở ô bên dưới và nhấn **Enter** để chat theo các tài liệu bạn đã chọn.",
                elem_classes="feature-sub",
            )
            q = gr.Textbox(
                label="",
                lines=1,
                placeholder="Nhập câu hỏi và nhấn Enter…",
                elem_classes="qa-input",
            )
            with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                k_ask = gr.Slider(1, 32, value=6, step=1, label="Top-k retrieval")
            with gr.Accordion("JSON debug", open=False):
                ask_raw = gr.Code(label="", language="json")

        with gr.Tab("📝 Tóm tắt"):
            gr.Markdown(
                "Tạo tóm tắt theo phạm vi tài liệu đã chọn (và theo trang nếu bạn chỉ chọn 1 tài liệu).",
                elem_classes="feature-sub",
            )
            with gr.Row(equal_height=False, elem_classes="feature-layout"):
                with gr.Column(scale=4, min_width=320, elem_classes="feature-controls"):
                    s_query = gr.Textbox(label="Chủ đề (tuỳ chọn)", lines=1)
                    s_btn = gr.Button("Tạo tóm tắt", elem_classes="gen-btn")
                    with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                        s_k = gr.Slider(1, 64, value=10, step=1, label="Số đoạn truy xuất (k)")
                    with gr.Accordion("JSON debug", open=False):
                        s_raw = gr.Code(label="", language="json")

                with gr.Column(scale=8, min_width=420, elem_classes="feature-output"):
                    s_md = _result_markdown()
                    s_download = gr.File(label="Tải Markdown", interactive=False)

        with gr.Tab("📋 Quiz"):
            gr.Markdown(
                "Tạo bộ câu hỏi trắc nghiệm từ các tài liệu bạn đã chọn.",
                elem_classes="feature-sub",
            )
            with gr.Row(equal_height=False, elem_classes="feature-layout"):
                with gr.Column(scale=4, min_width=320, elem_classes="feature-controls"):
                    z_query = gr.Textbox(label="Chủ đề (tuỳ chọn)", lines=1)
                    z_btn = gr.Button("Tạo quiz", elem_classes="gen-btn")
                    with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                        z_count = gr.Slider(1, 30, value=3, step=1, label="Số câu hỏi")
                        z_k = gr.Slider(1, 64, value=10, step=1, label="Số đoạn truy xuất (k)")
                    with gr.Accordion("JSON debug", open=False):
                        z_raw = gr.Code(label="", language="json")

                with gr.Column(scale=8, min_width=420, elem_classes="feature-output"):
                    z_md = _result_markdown()
                    z_download = gr.File(label="Tải Markdown", interactive=False)

        with gr.Tab("🃏 Flashcards"):
            gr.Markdown(
                "Tạo flashcards từ các tài liệu bạn đã chọn để ôn tập nhanh.",
                elem_classes="feature-sub",
            )
            with gr.Row(equal_height=False, elem_classes="feature-layout"):
                with gr.Column(scale=4, min_width=320, elem_classes="feature-controls"):
                    f_query = gr.Textbox(label="Chủ đề (tuỳ chọn)", lines=1)
                    f_btn = gr.Button("Tạo flashcards", elem_classes="gen-btn")
                    with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                        f_count = gr.Slider(1, 40, value=15, step=1, label="Số thẻ")
                        f_k = gr.Slider(1, 64, value=16, step=1, label="Số đoạn truy xuất (k)")
                    with gr.Accordion("JSON debug", open=False):
                        f_raw = gr.Code(label="", language="json")

                with gr.Column(scale=8, min_width=420, elem_classes="feature-output"):
                    f_md = _result_markdown()
                    f_download = gr.File(label="Tải Markdown", interactive=False)

    gr.HTML(
        '<div class="footer-text">'
        'Created by <a href="https://vlai.aivietnam.edu.vn/" target="_blank">VLAI</a>'
        ' &bull; <a href="https://aivietnam.edu.vn/" target="_blank">AI VIETNAM</a>'
        '</div>'
    )

    refresh_btn.click(
        fn=_refresh_docs,
        inputs=[],
        outputs=[docs, doc_map_state, page, doc_summary, doc_list_md],
    )
    docs.change(fn=_pages_for_selection, inputs=[doc_map_state, docs], outputs=[page])
    upload_btn.click(
        fn=_upload_pdf,
        inputs=[upload],
        outputs=[upload_status, docs, doc_map_state, page, doc_summary, doc_list_md],
    )

    q.submit(
        fn=_ask_chat,
        inputs=[q, chatbot, k_ask, docs, page, gemini_key_input],
        outputs=[chatbot, ask_raw, q],
    )
    s_btn.click(fn=_summarize, inputs=[s_query, s_k, docs, page, gemini_key_input], outputs=[s_md, s_raw]).then(
        fn=lambda text: _write_export(text, "summary.md"),
        inputs=[s_md],
        outputs=[s_download],
    )
    z_btn.click(fn=_quiz, inputs=[z_query, z_count, z_k, docs, page, gemini_key_input], outputs=[z_md, z_raw]).then(
        fn=lambda text: _write_export(text, "quiz.md"),
        inputs=[z_md],
        outputs=[z_download],
    )
    f_btn.click(
        fn=_flashcards,
        inputs=[f_query, f_count, f_k, docs, page, gemini_key_input],
        outputs=[f_md, f_raw],
    ).then(
        fn=lambda text: _write_export(text, "flashcards.md"),
        inputs=[f_md],
        outputs=[f_download],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        allowed_paths=["static/aivn_logo.png"],
        css=_CSS, theme=_theme
    )

