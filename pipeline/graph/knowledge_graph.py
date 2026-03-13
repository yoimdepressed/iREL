import os
import json
import yaml
import networkx as nx


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


def save_interactive_graph(G: nx.DiGraph, video_id: str, output_dir: str, vis_min_confidence: float = 0.35):
    """Generate a clean interactive HTML graph using D3.js (no pyvis bloat).
    Only edges with confidence >= vis_min_confidence are shown, keeping the graph readable.
    vis_min_confidence is configurable via config.yaml graph.visualization_min_confidence.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only keep edges above the confidence threshold to reduce visual clutter
    strong_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("confidence", 0) >= vis_min_confidence]

    SG = nx.DiGraph()
    for u, v, d in strong_edges:
        SG.add_edge(u, v, **d)
    # Only include nodes that have at least one strong edge
    # (isolated nodes with no strong edges are omitted — they clutter the graph without adding info)

    nodes_data = []
    for node in SG.nodes():
        in_deg = SG.in_degree(node)
        out_deg = SG.out_degree(node)
        if in_deg == 0:
            group = "foundational"
            color = "#ff6b6b"
        elif out_deg == 0:
            group = "advanced"
            color = "#4ecdc4"
        else:
            group = "intermediate"
            color = "#a855f7"
        nodes_data.append({
            "id": node,
            "label": node,
            "group": group,
            "color": color,
            "in_degree": in_deg,
            "out_degree": out_deg,
        })

    edges_data = []
    for u, v, d in SG.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "confidence": round(d.get("confidence", 1.0), 3),
            "nli_score": round(d.get("nli_score", 0), 3),
        })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    # Compute learning order for display
    try:
        order = list(nx.topological_sort(SG))
    except Exception:
        order = list(SG.nodes())

    order_html = " → ".join(f'<span class="order-node">{n}</span>' for n in order)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Knowledge Graph — {video_id}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  #header {{
    padding: 16px 24px;
    background: #1a1a2e;
    border-bottom: 1px solid #2d2d4e;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }}
  #header h1 {{ font-size: 1.1rem; color: #c084fc; letter-spacing: 0.05em; }}
  #header p {{ font-size: 0.78rem; color: #888; margin-top: 2px; }}
  #legend {{
    display: flex;
    gap: 20px;
    font-size: 0.75rem;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 50%;
  }}
  #order-bar {{
    padding: 10px 24px;
    background: #13131f;
    border-bottom: 1px solid #2d2d4e;
    font-size: 0.73rem;
    color: #aaa;
    flex-shrink: 0;
    overflow-x: auto;
    white-space: nowrap;
  }}
  #order-bar strong {{ color: #c084fc; margin-right: 8px; }}
  .order-node {{
    color: #e0e0e0;
    background: #1e1e3a;
    padding: 2px 8px;
    border-radius: 10px;
    margin: 0 2px;
  }}
  #graph {{ flex: 1; }}
  svg {{ width: 100%; height: 100%; }}
  .node circle {{
    stroke: #0f0f1a;
    stroke-width: 2px;
    cursor: pointer;
    transition: r 0.2s, filter 0.2s;
  }}
  .node circle:hover {{
    filter: brightness(1.4);
  }}
  .node text {{
    fill: #f0f0f0;
    font-size: 12px;
    font-weight: 500;
    pointer-events: none;
    text-shadow: 0 1px 3px #000;
  }}
  .link {{
    fill: none;
    stroke-opacity: 0.55;
    transition: stroke-opacity 0.2s;
  }}
  .link:hover {{ stroke-opacity: 1; }}
  .node.dimmed circle {{ opacity: 0.15; }}
  .node.dimmed text {{ opacity: 0.1; }}
  .link.dimmed {{ stroke-opacity: 0.05; }}
  #tooltip {{
    position: fixed;
    background: #1e1e3a;
    border: 1px solid #4a4a7a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    pointer-events: none;
    display: none;
    max-width: 260px;
    line-height: 1.6;
    color: #ddd;
    z-index: 100;
  }}
  #tooltip strong {{ color: #c084fc; display: block; margin-bottom: 4px; font-size: 0.85rem; }}
  #tooltip .tag {{
    display: inline-block;
    background: #2d2d50;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    margin-top: 4px;
  }}
  .arrowhead {{ fill: #888; }}
</style>
</head>
<body>
<div id="header">
  <div>
    <h1>📚 Knowledge Graph — {video_id.replace("_", " ").title()}</h1>
    <p>{len(nodes_data)} concepts &nbsp;·&nbsp; {len(edges_data)} prerequisite links</p>
  </div>
  <div id="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Foundational</div>
    <div class="legend-item"><div class="legend-dot" style="background:#a855f7"></div>Intermediate</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4ecdc4"></div>Advanced</div>
  </div>
</div>
<div id="order-bar">
  <strong>Learning Order:</strong>{order_html}
</div>
<div id="graph"></div>
<div id="tooltip"></div>

<script>
const nodesData = {nodes_json};
const edgesData = {edges_json};

const container = document.getElementById('graph');
const W = container.clientWidth || window.innerWidth;
const H = container.clientHeight || window.innerHeight - 100;

const svg = d3.select('#graph').append('svg')
  .attr('viewBox', `0 0 ${{W}} ${{H}}`)
  .attr('preserveAspectRatio', 'xMidYMid meet');

// Arrow marker
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 22)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-5L10,0L0,5')
  .attr('class', 'arrowhead');

const g = svg.append('g');

// Zoom & pan
svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', e => g.attr('transform', e.transform)));

// Edge color by confidence
const edgeColor = d => d3.interpolateRgb('#3a3a6a', '#c084fc')(d.confidence);

const link = g.append('g').selectAll('line')
  .data(edgesData).enter().append('line')
  .attr('class', 'link')
  .attr('stroke', edgeColor)
  .attr('stroke-width', d => 1.5 + d.confidence * 2.5)
  .attr('marker-end', 'url(#arrow)');

const nodeSize = d => 10 + d.in_degree * 2 + d.out_degree * 1.5;

const node = g.append('g').selectAll('g')
  .data(nodesData).enter().append('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag',  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}));

node.append('circle')
  .attr('r', nodeSize)
  .attr('fill', d => d.color)
  .attr('stroke', '#0f0f1a')
  .attr('stroke-width', 2);

node.append('text')
  .text(d => d.label)
  .attr('x', d => nodeSize(d) + 4)
  .attr('y', 4)
  .attr('font-size', '12px');

// Hover highlight
const tooltip = document.getElementById('tooltip');

node.on('mouseover', (e, d) => {{
  const connected = new Set([d.id]);
  edgesData.forEach(ed => {{
    // After D3 simulation runs, ed.source and ed.target are mutated from string IDs
    // to full node objects. Use .id if available, fall back to the value itself.
    const s = ed.source.id ?? ed.source;
    const t = ed.target.id ?? ed.target;
    if (s === d.id) connected.add(t);
    if (t === d.id) connected.add(s);
  }});
  node.classed('dimmed', n => !connected.has(n.id));
  link.classed('dimmed', l => {{
    const s = l.source.id ?? l.source;
    const t = l.target.id ?? l.target;
    return s !== d.id && t !== d.id;
  }});
  tooltip.style.display = 'block';
  tooltip.innerHTML = `<strong>${{d.label}}</strong>
    <div>Role: <b>${{d.group}}</b></div>
    <div>← Prerequisites: ${{d.in_degree}}</div>
    <div>→ Leads to: ${{d.out_degree}}</div>`;
}})
.on('mousemove', e => {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY - 10) + 'px';
}})
.on('mouseout', () => {{
  node.classed('dimmed', false);
  link.classed('dimmed', false);
  tooltip.style.display = 'none';
}});

// Force simulation
const sim = d3.forceSimulation(nodesData)
  .force('link', d3.forceLink(edgesData).id(d => d.id).distance(120).strength(0.6))
  .force('charge', d3.forceManyBody().strength(-350))
  .force('center', d3.forceCenter(W / 2, H / 2))
  .force('collide', d3.forceCollide().radius(d => nodeSize(d) + 18))
  .on('tick', () => {{
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
  }});
</script>
</body>
</html>"""

    out_path = os.path.join(output_dir, f"{video_id}_graph.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
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
