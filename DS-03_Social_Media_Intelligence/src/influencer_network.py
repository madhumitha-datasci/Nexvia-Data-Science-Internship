
import json
import logging
from pathlib import Path

import pandas as pd
import networkx as nx

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "all_cleaned_records.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def build_network():
    logging.info("Loading cleaned social media data...")

    df = pd.read_csv(DATA_FILE)

    videos = df[df["record_type"] == "video"].copy()
    comments = df[df["record_type"] == "comment"].copy()

    # Map each video ID to its creator's channel ID
    video_creators = (
        videos.dropna(subset=["video_id", "channel_id"])
        .drop_duplicates(subset=["video_id"])
        .set_index("video_id")["channel_id"]
        .to_dict()
    )

    graph = nx.DiGraph()

    # Add creator nodes
    for channel_id in video_creators.values():
        graph.add_node(channel_id, node_type="creator")

    logging.info("Building commenter-to-creator interactions...")

    valid_interactions = 0

    for _, comment in comments.iterrows():
        video_id = comment.get("video_id")
        commenter = comment.get("author_channel_id")

        creator = video_creators.get(video_id)

        if pd.isna(commenter) or not creator:
            continue

        commenter = str(commenter).strip()
        creator = str(creator).strip()

        if not commenter or not creator:
            continue

        if commenter == creator:
            continue

        if not graph.has_node(commenter):
            graph.add_node(commenter, node_type="commenter")

        # Multiple comments by the same user increase interaction weight
        if graph.has_edge(commenter, creator):
            graph[commenter][creator]["weight"] += 1
        else:
            graph.add_edge(commenter, creator, weight=1)

        valid_interactions += 1

    logging.info("Valid interactions: %s", valid_interactions)
    logging.info("Nodes: %s", graph.number_of_nodes())
    logging.info("Edges: %s", graph.number_of_edges())

    return graph, valid_interactions


def analyze_network(graph):
    logging.info("Calculating PageRank...")

    if graph.number_of_nodes() == 0:
        raise ValueError("No valid network nodes were found.")

    pagerank = nx.pagerank(graph, weight="weight")

    # Identify the most influential nodes
    ranked_nodes = sorted(
        pagerank.items(),
        key=lambda item: item[1],
        reverse=True
    )

    pagerank_df = pd.DataFrame(
        ranked_nodes,
        columns=["channel_id", "pagerank_score"]
    )

    pagerank_df["node_type"] = pagerank_df["channel_id"].map(
        nx.get_node_attributes(graph, "node_type")
    )

    pagerank_df.to_csv(
        REPORT_DIR / "influencer_rankings.csv",
        index=False
    )

    logging.info("Detecting communities...")

    # Community detection on an undirected version of the graph
    undirected_graph = graph.to_undirected()

    if undirected_graph.number_of_edges() > 0:
        communities = nx.community.greedy_modularity_communities(
            undirected_graph,
            weight="weight"
        )

        community_map = {}

        for community_number, members in enumerate(communities):
            for member in members:
                community_map[member] = community_number

    else:
        community_map = {
            node: index
            for index, node in enumerate(undirected_graph.nodes())
        }

    community_df = pd.DataFrame(
        [
            {
                "channel_id": node,
                "community": community_map.get(node, -1),
                "node_type": graph.nodes[node].get("node_type")
            }
            for node in graph.nodes()
        ]
    )

    community_df.to_csv(
        REPORT_DIR / "network_communities.csv",
        index=False
    )

    # Save graph structure
    graph_data = nx.node_link_data(graph)

    with open(REPORT_DIR / "influencer_network.json", "w", encoding="utf-8") as file:
        json.dump(graph_data, file, indent=2)

    # Network summary
    summary = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "total_communities": len(set(community_map.values())),
        "valid_interactions": valid_interactions_global,
        "top_10_influencers": [
            {
                "channel_id": channel_id,
                "pagerank_score": round(score, 6)
            }
            for channel_id, score in ranked_nodes[:10]
        ]
    }

    with open(REPORT_DIR / "network_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


if __name__ == "__main__":
    graph, valid_interactions_global = build_network()

    summary = analyze_network(graph)

    print("\nINFLUENCER NETWORK ANALYSIS COMPLETED")
    print("-------------------------------------")
    print("Total nodes:", summary["total_nodes"])
    print("Total edges:", summary["total_edges"])
    print("Communities:", summary["total_communities"])
    print("Valid interactions:", summary["valid_interactions"])

    print("\nTop 10 Influencers:")
    for index, influencer in enumerate(summary["top_10_influencers"], start=1):
        print(
            index,
            influencer["channel_id"],
            "PageRank:",
            influencer["pagerank_score"]
        )
