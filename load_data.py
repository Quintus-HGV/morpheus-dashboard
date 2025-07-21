import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

@st.cache_data
def load_activity_data():
    
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_data(tx):
        query = """
        MATCH (u:User)-[:BELONGS_TO]->(t:Tenant)
        MATCH (u)-[:PERFORMED]->(a:Action)
        OPTIONAL MATCH (a)-[:PROVISIONS]->(i:Instance)
        OPTIONAL MATCH (a)-[:DELETES]->(i:Instance)
        RETURN 
            t.name AS tenant, 
            u.username AS username, 
            a.type AS type, 
            a.success AS status,
            a.ts AS ts, 
            i.name AS instance_name,
            i.id AS instance_id,
            a.message AS message, 
            a.objecttype AS objecttype, 
            a.id AS id
        ORDER BY a.ts DESC
        """

        result = tx.run(query)
        records = []

        for record in result:
            row = dict(record)
            if "ts" in row and hasattr(row["ts"], "to_native"):
                row["ts"] = row["ts"].to_native()
            records.append(row)

        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(fetch_data)

    driver.close()
    return df

@st.cache_data
def fetch_run_data():

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_runs(tx):
        query = """
        MATCH (t:Tenant)<-[:BELONGS_TO]-(:User)-[:PERFORMED]->(:Action)-[:PROVISIONS]->(i:Instance)-[:HAS_RUN]->(r:Run)
        RETURN 
            t.name AS tenant,
            r.start_date AS start,
            r.end_date AS end,
            r.avg_cpu_usage_percent AS avg_cpu,
            i.id AS instance_id
        """
        result = tx.run(query)
        records = []
        for record in result:
            if record["start"] and record["end"]:
                records.append({
                    "tenant": record["tenant"],
                    "start": record["start"].to_native(),
                    "end": record["end"].to_native(),
                    "avg_cpu": float(record["avg_cpu"]) if record["avg_cpu"] is not None else 0.0,
                    "instance_id": record["instance_id"]
                })
        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(query_runs)
    driver.close()
    return df

@st.cache_data
def fetch_instance_counts():
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_instances(tx):
        query = """
        MATCH (t:Tenant)<-[:BELONGS_TO]-(u:User)
        OPTIONAL MATCH (u)-[:PERFORMED]->(a:Action)-[r]->(i:Instance)
        WHERE type(r) IN ['PROVISIONS', 'DELETES']
        RETURN 
            t.name AS tenant,
            u.username AS username,
            i.id AS instance_id,
            i.instance_type AS instance_type,
            i.plan AS instance_plan,
            a.ts AS action_ts,
            type(r) AS action_type,
            i.curr_status AS curr_status
        """
        result = tx.run(query)
        records = []
        for record in result:
            row = dict(record)
            # Convert Neo4j DateTime to Python datetime
            if "action_ts" in row and hasattr(row["action_ts"], "to_native"):
                row["action_ts"] = row["action_ts"].to_native()
            records.append(row)

        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(query_instances)

    driver.close()
    return df


@st.cache_data
def fetch_execution_data():

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_executions(tx):
        query = """
        MATCH (u:User)-[:BELONGS_TO]->(t:Tenant)
        MATCH (u)-[:CREATED]->(j:Job)-[:HAS_EXECUTION]->(e:Execution)
        RETURN 
            t.name AS tenant,
            u.username AS username,
            j.name AS job_name,
            e.startDate AS ts,
            e.status AS status,
            e.`duration (in seconds)` AS duration_in_seconds,
            e.type AS process_type
        ORDER BY e.startDate DESC
        """
        result = tx.run(query)
        records = []
        for record in result:
            row = dict(record)
            if "ts" in row and hasattr(row["ts"], "to_native"):
                row["ts"] = row["ts"].to_native()
            records.append(row)
        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(query_executions)
    driver.close()
    return df

@st.cache_data
def fetch_temporal_activity_data(selected_tenants=None):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(tx):
        query = """
        // 1. ACTIONS
        MATCH (t:Tenant)<-[:BELONGS_TO]-(u:User)
        OPTIONAL MATCH (u)-[:PERFORMED]->(a:Action)
        OPTIONAL MATCH (a)-[:PROVISIONS|DELETES]->(i:Instance)
        WHERE $tenants IS NULL OR t.name IN $tenants
        RETURN 
        t.name AS tenant,
        u.username AS user,
        a.ts AS action_ts,
        a.type AS action_type,
        a.message AS message,
        NULL AS exec_start,
        NULL AS exec_duration_in_seconds,
        NULL AS exec_status,
        NULL AS exec_type,
        NULL AS run_start,
        NULL AS run_end,
        NULL AS run_avg_cpu,
        i.name AS instance_name,
        i.id AS instance_id,
        i.instance_type AS instance_type,
        i.plan AS instance_plan

        UNION

        // 2. EXECUTIONS
        MATCH (t:Tenant)<-[:BELONGS_TO]-(u:User)
        OPTIONAL MATCH (u)-[:CREATED]->(j:Job)-[:HAS_EXECUTION]->(e:Execution)
        WHERE $tenants IS NULL OR t.name IN $tenants
        RETURN 
        t.name AS tenant,
        u.username AS user,
        NULL AS action_ts,
        NULL AS action_type,
        NULL AS message,
        e.startDate AS exec_start,
        e.`duration (in seconds)` AS exec_duration_in_seconds,
        e.status AS exec_status,
        e.type AS exec_type,
        NULL AS run_start,
        NULL AS run_end,
        NULL AS run_avg_cpu,
        NULL AS instance_name,
        NULL AS instance_id,
        NULL AS instance_type,
        NULL AS instance_plan

        UNION

        // 3. RUNS
        MATCH (t:Tenant)<-[:BELONGS_TO]-(u:User)
        OPTIONAL MATCH (u)-[:PERFORMED]->(:Action)-[:PROVISIONS]->(i:Instance)-[:HAS_RUN]->(r:Run)
        WHERE $tenants IS NULL OR t.name IN $tenants
        RETURN 
        t.name AS tenant,
        u.username AS user,
        NULL AS action_ts,
        NULL AS action_type,
        NULL AS message,
        NULL AS exec_start,
        NULL AS exec_duration_in_seconds,
        NULL AS exec_status,
        NULL AS exec_type,
        r.start_date AS run_start,
        r.end_date AS run_end,
        r.avg_cpu_usage_percent AS run_avg_cpu,
        i.name AS instance_name,
        i.id AS instance_id,
        i.instance_type AS instance_type,
        i.plan AS instance_plan

        ORDER BY tenant, user, action_ts, exec_start, run_start
        """
        result = tx.run(query, tenants=selected_tenants)
        rows = []

        for r in result:
            record = dict(r)
            for field in ["action_ts", "exec_start", "run_start", "run_end"]:
                if field in record and hasattr(record[field], "to_native"):
                    record[field] = record[field].to_native()
            rows.append(record)

        return rows

    with driver.session() as session:
        data = session.execute_read(run_query)

    driver.close()
    return pd.DataFrame(data)

@st.cache_data
def prepare_llm_friendly_json(df):
    grouped = df.groupby(["tenant", "user"])
    tenant_map = {}

    for (tenant, user), group in grouped:
        entry = tenant_map.setdefault(tenant, {"tenant": tenant, "users": []})
        activity = []

        for _, row in group.iterrows():
            if pd.notna(row.get("action_ts")):
                activity.append({
                    "type": "action",
                    "ts": row["action_ts"].isoformat(),
                    "action_type": row.get("action_type"),
                    "message": row.get("message"),
                    "instance_name": row.get("instance_name"),
                    "instance_id": row.get("instance_id"),
                    "instance_type": row.get("instance_type"),
                    "instance_plan": row.get("instance_plan"),
                })
            if pd.notna(row.get("exec_start")):
                activity.append({
                    "type": "execution",
                    "start": row["exec_start"].isoformat(),
                    "duration_in_seconds": row.get("exec_duration_in_seconds"),
                    "status": row.get("exec_status"),
                    "exec_type": row.get("exec_type")
                })
            if pd.notna(row.get("run_start")):
                run_event = {
                    "type": "run",
                    "start": row["run_start"].isoformat(),
                    "end": row["run_end"].isoformat() if pd.notna(row.get("run_end")) else None,
                    "avg_cpu": row.get("run_avg_cpu"),
                    "instance_name": row.get("instance_name"),
                    "instance_id": row.get("instance_id"),
                    "instance_type": row.get("instance_type"),
                    "instance_plan": row.get("instance_plan")
                }
                activity.append(run_event)

        entry["users"].append({
            "user": user,
            "activity": activity
        })

    return tenant_map

@st.cache_data
def get_temporal_insights_from_ai(payload_json):
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        prompt = (
            "Analyze the following tenant activity logs which include timestamped actions and job executions.\n"
            "Look for hidden patterns, anomalies (e.g., failed jobs, usage gaps), night-time activity, or inefficiencies.\n\n"
            f"{payload_json}"
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Failed to fetch insights: {e}"

@st.cache_data
def load_combined_data():
    df = load_activity_data()
    exec_df = fetch_execution_data()

    # Normalize both to have similar fields for merging
    df["source"] = "Action"
    exec_df["source"] = "Execution"
    exec_df["type"] = exec_df["process_type"]
    exec_df["message"] = "Job: " + exec_df["job_name"] + " executed | Status: " + exec_df["status"]
    exec_df = exec_df[["tenant", "username", "type", "ts", "message", "source", "status"]]

    df = df[["tenant", "username", "type", "ts", "message", "source", "status"]]

    combined_df = pd.concat([df, exec_df], ignore_index=True)
    combined_df["date"] = combined_df["ts"].dt.date
    combined_df["status"] = combined_df["status"].apply(lambda x: "success" if x != "failed" and x else "failed")
    return combined_df