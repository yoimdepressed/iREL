import os
import json
import yaml
import networkx as nx
from pyvis.network import Network


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_graph(prerequisite_edges: list) -> nx.DiGraph:
    """Build a directed graph from prerequisite edges. Arrow: prerequisite → dependent."""
    G = nx.DiGraph()

    for edge in prerequisite_edges:
        src = edge["from"]
        dst = edge["to"]
        confidence = edge.get("confidence", 1.0)
        evidence = edge.get("evidence", "")
        nli = edge.get("nli_score", 0)

        G.add_node(src)
        G.add_node(dst)
        G.add_edge(src, dst, confidence=confidence, evidence=evidence, nli_score=nli)

    return G


def compute_learning_order(G: nx.DiGraph) -> list:
    """Topological sort = recommended learning order. Prereqs first.
    If cycles exist, iteratively remove the weakest edge in each cycle."""
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning: cycle detected — removing weakest edges to break cycles")
        # Iteratively find cycles and remove the lowest-confidence edge
        max_iterations = G.number_of_edges()  # safety cap
        for _ in range(max_iterations):
            try:
                cycle = nx.find_cycle(G)
            except nx.NetworkXNoCycle:
                break
            # find weakest edge in the cycle
            weakest_edge = min(
                cycle,
                key=lambda e: G.edges[e[0], e[1]].get("confidence", 1.0)
            )
            conf = G.edges[weakest_edge[0], weakest_edge[1]].get("confidence", 0)
            print(f"  Removing cycle edge: {weakest_edge[0]} → {weakest_edge[1]} (confidence={conf:.3f})")
            G.remove_edge(weakest_edge[0], weakest_edge[1])

        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            print("Warning: could not break all cycles")
            return []


def save_interactive_graph(G: nx.DiGraph, video_id: str, output_dir: str):
    """Generate an interactive HTML graph with Pyvis. Open in any browser."""
    os.makedirs(output_dir, exist_ok=True)

    net = Network(height="700px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white",
                  cdn_resources="in_line")  # embed all JS inline — no local lib/ files needed
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 18,
        "font": {"size": 14, "color": "white"},
        "borderWidth": 2
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "cubicBezier"}
      },
      "physics": {
        "stabilization": true
      }
    }
    """)

    # color by role: red = foundational, purple = intermediate, blue = advanced
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if in_deg == 0:
            color = "#e94560"   # red = no prerequisites (foundational)
        elif out_deg == 0:
            color = "#0f3460"   # blue = nothing depends on it (leaf)
        else:
            color = "#533483"   # purple = intermediate

        net.add_node(node, label=node, color=color,
                     title=f"Prerequisites: {in_deg} | Leads to: {out_deg}")

    for src, dst, data in G.edges(data=True):
        confidence = data.get("confidence", 1.0)
        evidence = data.get("evidence", "")
        nli = data.get("nli_score", 0)
        width = max(1, confidence * 5)
        net.add_edge(src, dst, value=width,
                     title=f"Confidence: {confidence:.2f}\nNLI: {nli:.2f}\n\nEvidence: {evidence[:200]}")

    out_path = os.path.join(output_dir, f"{video_id}_graph.html")
    net.save_graph(out_path)
    print(f"Interactive graph saved: {out_path}")
    return out_path


def save_graph_metadata(G: nx.DiGraph, learning_order: list, video_id: str, output_dir: str):
    """Save a JSON summary: nodes, edges, learning order."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "video_id": video_id,
        "total_concepts": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "recommended_learning_order": learning_order,
        "nodes": list(G.nodes()),
        "edges": [
            {
                "from": u,
                "to": v,
                "confidence": d.get("confidence"),
                "nli_score": d.get("nli_score"),
                "evidence": d.get("evidence"),
            }
            for u, v, d in G.edges(data=True)
        ],
    }

    out_path = os.path.join(output_dir, f"{video_id}_graph_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Graph metadata saved: {out_path}")
    return out_path


if __name__ == "__main__":
    config = load_config()
    json_dir = config["paths"]["json_dir"]
    graphs_dir = config["paths"]["graphs_dir"]

    for video in config["videos"]:
        prereq_path = os.path.join(json_dir, f"{video['id']}_prerequisites.json")
        if not os.path.exists(prereq_path):
            print(f"Prerequisites not found for {video['id']}")
            continue

        with open(prereq_path, "r", encoding="utf-8") as f:
            prereq_data = json.load(f)

        print(f"\nBuilding graph for: {video['id']}")
        G = build_graph(prereq_data["prerequisite_edges"])
        order = compute_learning_order(G)

        if order:
            print(f"Learning order: {' → '.join(order)}")

        save_interactive_graph(G, video["id"], graphs_dir)
        save_graph_metadata(G, order, video["id"], json_dir)
