
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_FILE = PROJECT_ROOT / "reports" / "influencer_network.json"


def import_network():
    if not all([URI, USERNAME, PASSWORD]):
        raise ValueError("Neo4j connection details are missing from .env")

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}")

    with open(GRAPH_FILE, "r", encoding="utf-8") as file:
        graph_data = json.load(file)

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    node_rows = [
        {
            "channel_id": node["id"],
            "node_type": node.get("node_type", "unknown")
        }
        for node in nodes
    ]

    edge_rows = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "weight": edge.get("weight", 1)
        }
        for edge in edges
    ]

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.")

        with driver.session(database="neo4j") as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (c:Channel {channel_id: row.channel_id})
                SET c.node_type = row.node_type
            """, rows=node_rows).consume()

            session.run("""
                UNWIND $rows AS row
                MATCH (source:Channel {channel_id: row.source})
                MATCH (target:Channel {channel_id: row.target})
                MERGE (source)-[r:COMMENTED_ON]->(target)
                SET r.weight = row.weight
            """, rows=edge_rows).consume()

            node_count = session.run(
                "MATCH (c:Channel) RETURN count(c) AS total"
            ).single()["total"]

            edge_count = session.run(
                "MATCH ()-[r:COMMENTED_ON]->() RETURN count(r) AS total"
            ).single()["total"]

        print("\nNEO4J IMPORT COMPLETED")
        print("---------------------")
        print("Nodes imported:", node_count)
        print("Relationships imported:", edge_count)

    finally:
        driver.close()


if __name__ == "__main__":
    import_network()
