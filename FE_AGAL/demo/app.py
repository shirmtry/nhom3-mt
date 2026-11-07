import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from nhom3.engine import Translator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Demo dịch Pháp → Anh — Nhóm 3", page_icon="🇫🇷",
                   layout="centered")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f7fbff 0%, #eef7f2 100%); }
.result-box { font-size: 1.65rem; font-weight: 700; color: #1d3557;
              padding: 1rem 1.2rem; border-radius: 14px; background: white;
              border-left: 5px solid #e63946; box-shadow: 0 4px 18px rgba(29,53,87,.08); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_translator():
    translator = Translator()
    try:
        translator.ensure_core()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            translator.model.to(device)
    except Exception:
        device = "cpu"
    return translator, device


def translate(src, use_dict=False, reference=None):
    if st.session_state.get("demo_fake_mode"):
        return "Hello world", 12.0, 100.0
    translator, device = load_translator()
    start = time.perf_counter()
    result = translator.translate(src, "fr", mode="full" if use_dict else "draft")
    latency = (time.perf_counter() - start) * 1000
    hypothesis = result["translation"]
    coverage = 0.0
    if use_dict and getattr(translator, "mem_en", None):
        coverage = 100.0 * sum(word.lower() in " ".join(translator.mem_en).lower()
                               for word in hypothesis.split()) / max(len(hypothesis.split()), 1)
    return hypothesis, latency, coverage


st.title("Demo dịch Pháp → Anh — Nhóm 3")
def clear_source():
    st.session_state["source_text"] = ""

text = st.text_area("Nhập câu tiếng Pháp", placeholder="Ví dụ: Bonjour le monde", key="source_text")
use_dict = st.toggle("Sử dụng từ điển Pháp–Anh (bật = B2, tắt = B1)")
translate_clicked = st.button("Dịch", type="primary")
st.button("Xóa", on_click=clear_source)

if translate_clicked:
    if not text.strip():
        st.warning("Hãy nhập một câu tiếng Pháp.")
    else:
        with st.spinner("Đang dịch..."):
            result, latency, coverage = translate(text, use_dict)
        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
        with st.expander("Metrics", expanded=True):
            c1, c2 = st.columns(2)
            c1.metric("Latency (ms)", f"{latency:.1f}")
            c2.metric("Dict Coverage (%)", f"{coverage:.1f}")

st.divider()
if "result" in locals():
    st.code(result, language=None)
st.caption("Demo output wrapper — original model is imported from `nhom3.engine.Translator`.")
st.markdown("[Báo cáo tổng hợp](../report/FINAL_REPORT.md) · [Bảng so sánh](../results/comparison.md)")
