# ==== reader_baseline.py ====
import torch, json, pathlib
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL = "distilbert-base-cased-distilled-squad"
MODEL = "deepset/xlm-roberta-base-squad2"
class QAPipeline:
    def __init__(self, model_name=MODEL):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, question, context, max_len=384):
        enc = self.tok(question, context, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16 if DEVICE=="cuda" else torch.float32):
            out = self.model(**enc)
        start = int(out.start_logits.argmax())
        end   = int(out.end_logits.argmax())
        ans = self.tok.convert_tokens_to_string(self.tok.convert_ids_to_tokens(enc["input_ids"][0][start:end+1]))
        score = float(out.start_logits.max() + out.end_logits.max())
        return ans.strip(), score