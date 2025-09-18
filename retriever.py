# retriever.py
import re
import os
import pickle
from typing import List, Dict, Iterable, Optional, Tuple, Union

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25: pip install rank-bm25")

# --------- utils ---------
_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[^\w\s]")

def _normalize_text(s: str) -> str:
    # lower + strip punctuation + collapse spaces
    s = s.lower().strip()
    s = _PUNC.sub(" ", s)
    s = _WS.sub(" ", s)
    return s

def _simple_tokenize(s: str) -> List[str]:
    return _normalize_text(s).split()

# --------- retriever core ---------
class BM25Retriever:
    """
    Simple BM25 retriever:
      - build() from corpus (list[str] or list[dict{title,text}])
      - search(question, k) -> List[(idx, score)]
      - get_passages(idxs) -> List[str]
      - save()/load() cache the tokenized corpus + titles + texts
    """
    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: Optional[List[List[str]]] = None
        self._titles: Optional[List[str]] = None
        self._texts: Optional[List[str]] = None

    @staticmethod
    def _to_texts(corpus: Iterable[Union[str, Dict]]) -> Tuple[List[str], List[str]]:
        titles, texts = [], []
        for item in corpus:
            if isinstance(item, str):
                titles.append("")
                texts.append(item)
            elif isinstance(item, dict):
                title = item.get("title", "") or ""
                text  = item.get("text", "") or ""
                if not text:
                    # hotpot style: context = list[sentences] or {"title":..., "sentences":[...]}
                    sents = item.get("sentences") or item.get("sents") or []
                    if isinstance(sents, (list, tuple)):
                        text = " ".join(sents)
                titles.append(str(title))
                texts.append(str(text))
            else:
                raise TypeError(f"Unsupported corpus item type: {type(item)}")
        return titles, texts

    def build(self, corpus: Iterable[Union[str, Dict]], show_stats: bool = True) -> None:
        titles, texts = self._to_texts(corpus)
        tokenized = [_simple_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._tokenized = tokenized
        self._titles = titles
        self._texts = texts
        if show_stats:
            print(f"[BM25] Built index: {len(texts)} passages")

    def search(self, question: str, k: int = 4) -> List[Tuple[int, float]]:
        assert self._bm25 is not None and self._tokenized is not None, "Call build() or load() first."
        q_tok = _simple_tokenize(question)
        scores = self._bm25.get_scores(q_tok)
        # top-k indices
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in idxs]

    def get_passages(self, idxs: List[int]) -> List[str]:
        assert self._texts is not None
        return [self._texts[i] for i in idxs]

    def get_titles(self, idxs: List[int]) -> List[str]:
        assert self._titles is not None
        return [self._titles[i] for i in idxs]

    # ----- (de)serialization -----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "titles": self._titles,
                    "texts": self._texts,
                    "tokenized": self._tokenized,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[BM25] Saved corpus cache to {path}")

    def load(self, path: str, show_stats: True) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._titles = data["titles"]
        self._texts = data["texts"]
        self._tokenized = data["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized)
        if show_stats:
            print(f"[BM25] Loaded index from {path}: {len(self._texts)} passages")

# --------- adapters ---------
def build_corpus_from_hotpot_context_item(ex: Dict) -> List[Dict]:
    """
    HotpotQA 'fullwiki' training instances have fields:
      - 'context': List[ [title: str, sentences: List[str]] ]
    This helper flattens them into paragraph-like docs (title + joined sentences).
    """
    corpus = []
    ctx = ex.get("context") or []
    # ctx is a list of [title, sentences]
    for item in ctx:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            title, sents = item
            if isinstance(sents, (list, tuple)):
                corpus.append({"title": title, "text": " ".join(map(str, sents))})
        elif isinstance(item, dict):
            title = item.get("title", "")
            sents = item.get("sentences") or item.get("sents") or []
            corpus.append({"title": title, "text": " ".join(map(str, sents))})
    return corpus
