"""Demo Gradio cho Phần 4: nhập câu Việt hoặc Pháp, xem từng bước của kiến trúc.

  python app.py --config configs/default.yaml --exp C [--share]
"""
import argparse

import gradio as gr

from mtm.evaluate import get_rarity
from mtm.pipeline import load_system
from mtm.utils import load_config, wpath


def build_demo(tr):
    def run(text, lang_label, mode_label):
        lang = "vi" if lang_label.startswith("Tiếng Việt") else "fr"
        mode = {"Có cổng (đề xuất)": "full", "Chỉ dịch nháp": "draft", "Luôn tra kho": "always"}[mode_label]
        if not text.strip():
            return "", "", "", []
        r = tr.translate([text], [lang], mode=mode)[0]
        th = tr.thresholds(lang)
        gate = (f"C = {r['C']:.3f} (τ = {th['tau']:.2f}) · R = {r['R']:.2f} (ρ = {th['rho']:.2f})\n"
                f"→ {'KHÓ/HIẾM: tra kho Pháp–Anh' if r['hard'] else 'DỄ: dùng luôn bản nháp, không tra kho'}")
        if r["hard"]:
            if r["evidence"] is None:
                gate += "\n→ không tìm được bằng chứng"
            elif r["refined"] is None:
                gate += f"\n→ bằng chứng tốt nhất {r['ev_score']:.2f} < δ = {th['delta']:.2f}: giữ bản nháp"
            else:
                gate += f"\n→ e* (rerank {r['ev_score']:.2f}): {r['evidence']}"
        table = [[c["fr"], c["en"], round(c["cosine"], 3), round(c["rerank"], 3)] for c in r["candidates"]]
        return r["output"], r["draft"], gate, table

    with gr.Blocks(title="Multilingual Transformer VI/FR → EN") as demo:
        gr.Markdown("## Multilingual Transformer: Việt / Pháp → Anh\nCâu dễ dịch ngay; câu khó/hiếm mới tra kho Pháp–Anh.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Câu nguồn", lines=3, value="Nếu dữ liệu huấn luyện quá ít, mô hình khó tổng quát hóa trên miền mới.")
                lang = gr.Radio(["Tiếng Việt (vi_VN)", "Tiếng Pháp (fr_XX)"], value="Tiếng Việt (vi_VN)", label="Ngôn ngữ nguồn")
                mode = gr.Radio(["Có cổng (đề xuất)", "Chỉ dịch nháp", "Luôn tra kho"], value="Có cổng (đề xuất)", label="Chế độ")
                btn = gr.Button("Dịch", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="Bản dịch cuối y", lines=3)
                draft = gr.Textbox(label="② Bản nháp ŷ", lines=2)
                gate = gr.Textbox(label="③ Cổng quyết định", lines=4)
        cands = gr.Dataframe(headers=["④ Câu Pháp (khóa)", "Câu Anh đi kèm", "cosine", "⑤ rerank"], label="Top-5 trong kho Pháp–Anh")
        btn.click(run, [text, lang, mode], [out, draft, gate, cands])
        gr.Examples([["Tôi là sinh viên.", "Tiếng Việt (vi_VN)", "Có cổng (đề xuất)"],
                     ["Mô hình RAG giúp giảm ảo giác của mô hình ngôn ngữ lớn.", "Tiếng Việt (vi_VN)", "Có cổng (đề xuất)"],
                     ["Avec peu de données, le modèle généralise mal aux nouveaux domaines.", "Tiếng Pháp (fr_XX)", "Có cổng (đề xuất)"]],
                    [text, lang, mode])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", default="C")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tr, _ = load_system(cfg, wpath(cfg, "checkpoints", a.exp, "best"))
    tr.rarity = get_rarity(cfg)
    build_demo(tr).launch(share=a.share)
