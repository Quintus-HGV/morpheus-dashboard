import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class SemanticQueryMatcher:
    def __init__(self, template_text_path="llm/templates_only.json",template_path="llm/query_templates.json", index_path="llm/template_embeddings.index"):

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(template_text_path, "r") as f:
            self.template_texts = json.load(f)

        with open(template_path, "r") as f:
            self.query_templates = json.load(f)

        self.index = faiss.read_index(index_path)

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def match_template(self, user_query, top_k=1):
        emb = self.embed_model.encode([user_query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)
        results = []
        for j in range(top_k):
            idx = I[0][j]
            score = float(D[0][j])
            results.append({
                "template_text": self.template_texts[idx],
                "query": self.query_templates[idx]["cypher"],
                "params": self.query_templates[idx].get("params", {}),
                "score": score
            })
        return results

    def graph_rag(self, user_query, top_k=1):
        matched = self.match_template(user_query, top_k=top_k)[0]
        print(f"✅ Matched Template: {matched['template_text']} (score={matched['score']:.4f})")

        query_params = matched.get("params", {})

        with self.driver.session() as session:
            print(matched)
            result = session.run(matched["template_text"], query_params)
            data = [r.data() for r in result]

        return {
            "matched_template": matched["template_text"],
            "query_run": matched["query"],
            "graph_results": data
        }

if __name__ == "__main__":
    matcher = SemanticQueryMatcher()
    q = input("Ask your question: ")
    match = matcher.graph_rag(q)
    print("\nBest matched template:")
    print("Template:", match["matched_template"])
    print("Query Run:", match["query_run"])
    print("Graph Results:", match["graph_results"])
