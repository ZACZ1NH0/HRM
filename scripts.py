from datasets import load_dataset
from retriever import BM25Retriever

ds = load_dataset("hotpot_qa", "fullwiki")["train"]
corpus = []
for ex in ds:
    for ctx in ex["context"]:
        title, sents = ctx[0], ctx[1]
        corpus.append({"title": title, "text": " ".join(sents)})

retr = BM25Retriever()
retr.build(corpus, show_stats=True)
retr.save("artifacts/bm25_fullwiki.pkl")
