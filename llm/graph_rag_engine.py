import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import re

load_dotenv()

class SemanticQueryMatcher:
    def __init__(self, template_text_path="llm/templates_only.json",
                 template_path="llm/query_templates.json",
                 index_path="llm/template_embeddings.index",
                 raw_tenant_logs_path="llm/raw_tenant_logs.index",
                 raw_chunks_path="llm/raw_chunks.json"):

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

        self.log_index = faiss.read_index(raw_tenant_logs_path)
        with open(raw_chunks_path, "r") as f:
            self.raw_chunks = json.load(f)

    def get_all_values(self, label, field):
        query = f"MATCH (n:{label}) RETURN DISTINCT n.{field} AS value"
        with self.driver.session() as session:
            result = session.run(query)
            return [r["value"] for r in result if r["value"]]

    def extract_from_logs_semantically(self, user_query):
        emb = self.embed_model.encode([user_query], normalize_embeddings=True)
        D, I = self.log_index.search(np.array(emb), k=1)
        chunk = self.raw_chunks[I[0][0]]
        result = {}

        tenant_match = re.search(r"Tenant:\s*(.+)", chunk)
        if tenant_match:
            result["tenant_name"] = tenant_match.group(1).strip()

        plan_match = re.search(r"Instance Plan:\s*(.+)", chunk)
        if plan_match:
            result["plan_name"] = plan_match.group(1).strip()

        inst_type_match = re.search(r"Instance Type:\s*(.+)", chunk)
        if inst_type_match:
            result["instance_type"] = inst_type_match.group(1).strip()

        return result

    def match_template(self, user_query, top_k=3, score_threshold=0.3):
        emb = self.embed_model.encode([user_query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)

        results = []
        for j in range(top_k):
            idx = I[0][j]
            score = float(D[0][j])
            if score < score_threshold:
                continue
            results.append({
                "template_text": self.template_texts[idx],
                "query": self.query_templates[idx]["cypher"],
                "params": self.query_templates[idx].get("params", {}),
                "score": score
            })

        if not results:
            return [{
                "template_text": "No strong match found",
                "query": "",
                "params": {},
                "score": 0
            }]
        return results

    def extract_query_parameters(self, user_query, cypher_query, existing_params):
        params = existing_params.copy()
        required_params = list(set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", cypher_query)))

        semantic_values = self.extract_from_logs_semantically(user_query)

        for param in required_params:
            if param in params:
                continue
            if param in semantic_values:
                print(f"Log-matched `{param}` = '{semantic_values[param]}'")
                params[param] = semantic_values[param]
                continue
        return params

    def get_missing_parameters(self, cypher_query, existing_params):
        params = existing_params.copy()
        required_params = list(set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", cypher_query)))

        missing_params = []
        for param in required_params:
            if param not in params:
                missing_params.append(param)
        return missing_params

    def graph_rag(self, user_query, top_k=3):
        matches = self.match_template(user_query, top_k=top_k)
        matched = matches[0]

        print(f"Matched Template: {matched['template_text']} (score={matched['score']:.4f})")

        if not matched["query"]:
            return {
                "matched_template": matched["template_text"],
                "query_run": "",
                "graph_results": [],
                "missing_params": []
            }

        query_params = self.extract_query_parameters(
            user_query, matched["query"], matched.get("params", {})
        )
        missing = self.get_missing_parameters(matched["query"], query_params)

        if not missing:
            with self.driver.session() as session:
                result = session.run(matched["query"], query_params)
                data = [r.data() for r in result]
            return {
                "matched_template": matched["template_text"],
                "query_run": matched["query"],
                "graph_results": data,
                "missing_params": [],
                "base_params": query_params
            }

        return {
            "matched_template": matched["template_text"],
            "query_run": matched["query"],
            "graph_results": [],
            "missing_params": missing,
            "base_params": query_params
        }
# Only for stand-alone execution (testing locally)
# if __name__ == "__main__":
#     matcher = SemanticQueryMatcher()
#     q = input("Ask your question: ")
#     match = matcher.graph_rag(q)
#     print("\nBest matched template:")
#     print("Template:", match["matched_template"])
#     print("Query Run:", match["query_run"])
#     print("Graph Results:", json.dumps(match["graph_results"], indent=2))
