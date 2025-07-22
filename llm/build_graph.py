import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("query_templates.json", "r") as f:
    query_templates = json.load(f)

template_texts = [qt["cypher"] for qt in query_templates]

template_embeddings = embed_model.encode(template_texts)
dim = template_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(template_embeddings))

faiss.write_index(index, "template_embeddings.index")
with open("templates_only.json", "w") as f:
    json.dump(template_texts, f)
