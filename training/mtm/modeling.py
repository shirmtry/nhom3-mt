"""Bước 3.2 + 3.3 — Bộ tách từ chung và Multilingual Transformer.

- Bộ tách từ: SentencePiece 250K của mBART-50, mã ngôn ngữ ở đầu câu: [vi_VN] x </s>
- Mô hình: facebook/mbart-large-50 (đã tiền huấn luyện, chưa học dịch) + LoRA r=16 trên q_proj, v_proj
- Đầu ra luôn ép token đầu = en_XX
- Bước dịch lại (3.7) dùng cùng mô hình, đầu vào: [lang] x </s> e* </s>
- A0: Transformer nhỏ huấn luyện từ đầu, SentencePiece 16K riêng (để đo đóng góp của tiền huấn luyện)
"""
import os

import torch

from .utils import LANG_CODE, load_json, save_json


# ---------------------------------------------------------------- tokenizer
def load_tokenizer(path, kind="fast"):
    from transformers import MBart50Tokenizer, MBart50TokenizerFast
    cls = MBart50TokenizerFast if kind == "fast" else MBart50Tokenizer
    return cls.from_pretrained(path, src_lang="vi_VN", tgt_lang="en_XX")


def train_scratch_tokenizer(texts, out_dir, vocab_size=16000):
    """SentencePiece riêng cho A0, bọc trong MBart50Tokenizer để dùng chung mã ngôn ngữ."""
    import sentencepiece as spm
    from transformers import MBart50Tokenizer
    os.makedirs(out_dir, exist_ok=True)
    corpus = os.path.join(out_dir, "spm_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    prefix = os.path.join(out_dir, "spm")
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=vocab_size,
                                   character_coverage=1.0, model_type="unigram", hard_vocab_limit=False)
    tok = MBart50Tokenizer(vocab_file=prefix + ".model", src_lang="vi_VN", tgt_lang="en_XX")
    tok.save_pretrained(out_dir)
    return tok


def lang_id(tok, lang):
    return tok.convert_tokens_to_ids(LANG_CODE[lang])


def encode_sources(tok, texts, langs, evidences=None, max_src=128, max_ev=96):
    """Mã hóa câu nguồn (có thể nối bằng chứng e*). Trả về list[list[int]]."""
    evidences = evidences or [None] * len(texts)
    out = [None] * len(texts)
    for lang in set(langs):  # gom theo ngôn ngữ để gắn đúng mã [vi_VN]/[fr_XX]
        idx = [i for i, l in enumerate(langs) if l == lang]
        tok.src_lang = LANG_CODE[lang]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_src)["input_ids"]
        for i, ids in zip(idx, enc):
            out[i] = list(ids)
    ev_idx = [i for i, e in enumerate(evidences) if e]
    if ev_idx:
        ev = tok([evidences[i] for i in ev_idx], add_special_tokens=False, truncation=True, max_length=max_ev)["input_ids"]
        for i, ids in zip(ev_idx, ev):
            out[i] = out[i] + list(ids) + [tok.eos_token_id]  # [lang] x </s> e* </s>
    return out


def encode_targets(tok, texts, max_tgt=128):
    tok.tgt_lang = "en_XX"
    return [list(x) for x in tok(text_target=texts, truncation=True, max_length=max_tgt)["input_ids"]]


def pad_batch(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


# ---------------------------------------------------------------- mô hình
def build_lora_model(cfg):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import MBartForConditionalGeneration
    m = cfg["model"]
    base = MBartForConditionalGeneration.from_pretrained(m["base"])
    lcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=m["lora"]["r"], lora_alpha=m["lora"]["alpha"],
                      lora_dropout=m["lora"]["dropout"], target_modules=m["lora"]["target_modules"])
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()  # ≈ 2,36M tham số (0,39%)
    return model


def build_scratch_model(cfg, tok):
    from transformers import MBartConfig, MBartForConditionalGeneration
    s = cfg["scratch"]
    conf = MBartConfig(
        vocab_size=max(len(tok), tok.vocab_size), d_model=s["d_model"],
        encoder_layers=s["layers"], decoder_layers=s["layers"],
        encoder_attention_heads=s["heads"], decoder_attention_heads=s["heads"],
        encoder_ffn_dim=s["ffn"], decoder_ffn_dim=s["ffn"], dropout=s["dropout"],
        attention_dropout=0.1, activation_dropout=0.1, max_position_embeddings=512,
        pad_token_id=tok.pad_token_id, bos_token_id=0, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.eos_token_id, forced_eos_token_id=tok.eos_token_id,
        scale_embedding=True)
    return MBartForConditionalGeneration(conf)


def save_checkpoint(model, tok, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)  # LoRA: chỉ lưu adapter (~10 MB)
    tok.save_pretrained(out_dir)
    save_json(os.path.join(out_dir, "meta.json"), meta)


def load_checkpoint(cfg, ckpt_dir, merge=True):
    """Nạp mô hình đã huấn luyện (LoRA hoặc A0) để suy luận."""
    from transformers import MBartForConditionalGeneration
    meta = load_json(os.path.join(ckpt_dir, "meta.json"))
    tok = load_tokenizer(ckpt_dir, kind="slow" if meta["type"] == "scratch" else "fast")
    if meta["type"] == "lora":
        from peft import PeftModel
        base = MBartForConditionalGeneration.from_pretrained(cfg["model"]["base"])
        model = PeftModel.from_pretrained(base, ckpt_dir)
        if merge:
            model = model.merge_and_unload()
    else:
        model = MBartForConditionalGeneration.from_pretrained(ckpt_dir)
    from .utils import device
    model.to(device()).eval()
    return model, tok, meta


# ---------------------------------------------------------------- dịch + độ tin cậy C
@torch.no_grad()
def sequence_confidence(model, ids, mask, seqs, pad, chunk=4):
    # C = exp(trung bình log p của các token đã sinh), bỏ token en_XX bị ép (Fomicheva et al., 2020 — slide 3.5).
    # Tính lại bằng 1 lượt forward, không dùng output_scores (lưu điểm 250K từ vựng mỗi bước → tràn VRAM).
    dec_in, tgt = seqs[:, :-1], seqs[:, 1:]
    valid = tgt.ne(pad)
    valid[:, 0] = False
    conf = []
    for s in range(0, seqs.shape[0], chunk):
        logits = model(input_ids=ids[s:s + chunk], attention_mask=mask[s:s + chunk],
                       decoder_input_ids=dec_in[s:s + chunk]).logits.float()
        lp = logits.gather(-1, tgt[s:s + chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        v = valid[s:s + chunk]
        conf += torch.exp(lp.masked_fill(~v, 0).sum(1) / v.sum(1).clamp(min=1)).tolist()
        del logits, lp
    return conf


@torch.no_grad()
def generate(model, tok, texts, langs, evidences=None, num_beams=5, max_new_tokens=200, batch_size=16,
             max_src=128, max_ev=96, with_confidence=True):
    # Dịch danh sách câu sang tiếng Anh. Trả về (bản dịch, C).
    model.eval()
    dev = next(model.parameters()).device
    enc = encode_sources(tok, texts, langs, evidences, max_src, max_ev)
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))  # gom câu dài gần nhau cho nhanh
    hyps, confs = [None] * len(enc), [None] * len(enc)
    en_id, pad = lang_id(tok, "en"), tok.pad_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        ids, mask = pad_batch([enc[i] for i in idx], pad)
        ids, mask = ids.to(dev), mask.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            seqs = model.generate(input_ids=ids, attention_mask=mask, num_beams=num_beams,
                                  max_new_tokens=max_new_tokens, forced_bos_token_id=en_id,
                                  decoder_start_token_id=tok.eos_token_id)
            if not torch.is_tensor(seqs):
                seqs = seqs.sequences
            c = sequence_confidence(model, ids, mask, seqs, pad) if with_confidence else [None] * len(idx)
        texts_out = tok.batch_decode(seqs, skip_special_tokens=True)
        for j, i in enumerate(idx):
            hyps[i], confs[i] = texts_out[j].strip(), c[j]
        del seqs
    return hyps, confs
