import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

with open("neo4j_query_table_data.json", "r") as f:
    neo4j_data = json.load(f)

raw_chunks = []
for record in neo4j_data:
    text = f"""
Tenant: {record['tenant']}
User: {record['user']}
Action Timestamp: {record.get('action_ts')}
Action Type : {record.get('action_type')}
Message: {record.get('message')}
Execution Start: {record.get('exec_start')}
Duration: {record.get('exec_duration')}
Status: {record.get('exec_status')}
Run Start: {record.get('run_start')}
Run End: {record.get('run_end')}
Avg CPU: {record.get('run_avg_cpu')}
Instance: {record.get('instance_name')} (ID: {record.get('instance_id')})
Instance Type: {record.get('instance_type')}
Instance Plan: {record.get('instance_plan')}
"""
    raw_chunks.append(text.strip())
    print(text)
print(f"✅ Processed {len(raw_chunks)} raw log chunks")

with open("data.txt", "r", encoding="utf-8") as f:
    summary_text = f.read()

def chunk_text(text, chunk_size=800):
    sentences = text.split("\n")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + "\n"
        else:
            chunks.append(current.strip())
            current = s + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

summary_chunks = chunk_text(summary_text)
print(f"Processed {len(summary_chunks)} summary chunks")

def build_faiss(chunks, index_file):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)

build_faiss(raw_chunks, "raw_tenant_logs.index")
build_faiss(summary_chunks, "summary_logs.index")

with open("raw_chunks.json", "w") as f: json.dump(raw_chunks, f)
with open("summary_chunks.json", "w") as f: json.dump(summary_chunks, f)

print("Saved")
