import json
import typing as _t
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import warnings  # Import warnings at the top

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# -------------------------------------------------------------
# Toroidal distance
# -------------------------------------------------------------
def toroidal_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    diff = np.abs(p - q)
    diff = np.where(diff > 0.5, 1.0 - diff, diff)
    return float(np.linalg.norm(diff))


# -------------------------------------------------------------
# Visualizer (Plotly version only)
# -------------------------------------------------------------
class RGGVisualizer:
    """Visualize RGGs into HTML using Plotly (fixed geometric positions)."""

    def __init__(self, G: _t.Optional[nx.Graph] = None, scale: int = 800):
        self.G = G
        self.scale = scale

    def from_networkx(self, G: nx.Graph) -> "RGGVisualizer":
        self.G = G
        return self

    # ---------------------------------------------------------
    # Wrap-edge detector
    # ---------------------------------------------------------
    def _is_wrap_edge(self, p, q):
        dx = abs(p[0] - q[0])
        dy = abs(p[1] - q[1])
        return (dx > 0.5) or (dy > 0.5)

    # ---------------------------------------------------------
    # Largest component
    # ---------------------------------------------------------
    def _get_component(self, largest: bool) -> nx.Graph:
        if self.G is None:
            raise ValueError("No graph set. Call from_networkx(G) first.")
        if not largest:
            return self.G
        comps = list(nx.connected_components(self.G))
        if not comps:
            return nx.Graph()
        g = max(comps, key=len)
        return self.G.subgraph(g).copy()

    # ---------------------------------------------------------
    # Enforce correct 2D pos
    # ---------------------------------------------------------
    def _ensure_pos(self, G: nx.Graph):
        pos = nx.get_node_attributes(G, "pos")
        if len(pos) != G.number_of_nodes():
            raise ValueError("All nodes must have 'pos' attribute.")

        cleaned = {}
        for n, p in pos.items():
            if len(p) != 2:
                raise ValueError(f"pos for node {n} must be 2D.")
            cleaned[int(n)] = (float(p[0]), float(p[1]))
        return cleaned

    # ---------------------------------------------------------
    # Pair info (distances + reff)
    # ---------------------------------------------------------
    def _compute_pair_info(self, G: nx.Graph, metric: str = "toroidal") -> _t.Dict[str, _t.Dict[str, float]]:
        nodes = [int(n) for n in G.nodes()]
        n = len(nodes)
        if n == 0:
            return {}

        comp_id = {}
        for cid, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                comp_id[int(node)] = cid

        pos_map = self._ensure_pos(G)
        coords = np.array([pos_map[nodes[i]] for i in range(n)], dtype=float)
        degs = np.array([float(G.degree(nodes[i])) for i in range(n)], dtype=float)

        L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        try:
            L_plus = np.linalg.pinv(L)
        except Exception:
            L_plus = np.linalg.pinv(L + np.eye(n) * 1e-12)

        pair_info = {}
        for i in range(n):
            for j in range(i + 1, n):
                u, v = nodes[i], nodes[j]
                if comp_id.get(u) != comp_id.get(v):
                    continue

                tor_d = float(toroidal_distance(coords[i], coords[j]))
                eucl_d = float(np.linalg.norm(coords[i] - coords[j]))
                reff = float(L_plus[i, i] + L_plus[j, j] - 2.0 * L_plus[i, j])

                du = degs[i] if degs[i] > 0 else np.inf
                dv = degs[j] if degs[j] > 0 else np.inf
                inv_deg_sum = float((1.0 / du) + (1.0 / dv)) if (du != np.inf and dv != np.inf) else float(np.inf)

                key = f"{u}|{v}"
                pair_info[key] = {
                    "toroidal": tor_d,
                    "euclidean": eucl_d,
                    "reff": reff,
                    "inv_deg_sum": inv_deg_sum,
                }
        return pair_info

    # ---------------------------------------------------------
    # Plotly only
    # ---------------------------------------------------------
    def _write_plotly_html(self, filename, Gvis, pos, pair_info, metric):

        # edges
        edge_x, edge_y = [], []
        for u, v in Gvis.edges():
            pu = np.array(pos[int(u)])
            pv = np.array(pos[int(v)])
            if self._is_wrap_edge(pu, pv):
                continue
            edge_x.extend([pu[0], pv[0], None])
            edge_y.extend([pu[1], pv[1], None])

        # nodes
        node_x = [pos[int(n)][0] for n in Gvis.nodes()]
        node_y = [pos[int(n)][1] for n in Gvis.nodes()]

        # Node size
        node_size = max(10, 35 * (len(Gvis.nodes()) ** (-0.35)))

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color='#cccccc'), # Light gray edges
            hoverinfo="none"
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=[f"node: {n}\ndegree: {Gvis.degree(int(n))}" for n in Gvis.nodes()],
            customdata=[int(n) for n in Gvis.nodes()],  # store node ID here
            marker=dict(size=node_size, color="skyblue", line=dict(width=1, color="black"))
        )
        
        # --- MODIFIED LAYOUT ---
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=f"RGG Visualization (metric={metric})", font=dict(size=20), x=0.5),
                xaxis=dict(
                    range=[0, 1],
                    scaleanchor="y",
                    scaleratio=1,
                    showgrid=False,
                    zeroline=False,
                    showline=True,    # --- ADDED BOX OUTLINE ---
                    linewidth=2,      # --- ADDED BOX OUTLINE ---
                    linecolor='black' # --- ADDED BOX OUTLINE ---
                ),
                yaxis=dict(
                    range=[0, 1],
                    showgrid=False,
                    zeroline=False,
                    showline=True,    # --- ADDED BOX OUTLINE ---
                    linewidth=2,      # --- ADDED BOX OUTLINE ---
                    linecolor='black' # --- ADDED BOX OUTLINE ---
                ),
                width=900,
                height=900,
                plot_bgcolor="#f9f9f9", # Light bg for the plot itself
                showlegend=False,
                hovermode="closest",
                dragmode="pan",           # Stops click-drag from zooming
                clickmode="event+select", # Tells Plotly to fire click events
                margin=dict(l=40, r=40, t=80, b=40) # Added margin
            )
        )
        
        config = {"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset"}
        fig.write_html(filename, include_plotlyjs="cdn", config=config)
        
        # --- INJECTED CSS STYLES ---
        style_block = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f0f2f5; /* Light gray-blue page background */
    margin: 0;
    padding: 20px;
    display: grid;
    place-items: center;
    min-height: 100vh;
  }
  /* Add shadow and border-radius to the plot */
  .plotly-graph-div {
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  }
  /* Style the dashboard */
  #pair-info-container {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 320px; /* --- MADE WIDER --- */
    background: #ffffff;
    padding: 20px; /* --- INCREASED PADDING --- */
    border: 1px solid #ddd;
    white-space: pre-wrap;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 14px; /* --- INCREASED FONT SIZE --- */
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    z-index: 1000;
  }
  #pair-info {
    line-height: 1.6; /* Better readability */
  }
  #clear-selection {
    width: 100%;
    margin-top: 15px;
    background-color: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 5px;
    cursor: pointer;
    padding: 8px 0; /* Taller button */
    font-size: 14px;
    font-weight: 600;
    transition: background-color 0.2s;
  }
  #clear-selection:hover {
    background-color: #e0e0e0;
  }
</style>
"""

        # --- UPDATED DASHBOARD HTML (no inline styles) ---
        info_box_html = """
<div id="pair-info-container">
  <div id="pair-info">Click a node...</div>
  <button id="clear-selection">Clear Selection</button>
</div>
"""

        # --- *** THIS IS THE CORRECTED JAVASCRIPT BLOCK *** ---
        # We must escape the JS template literal braces `${...}`
        # by doubling the braces: `${{...}}`
        js = f"""
<script>
const DATA_PAIR = {json.dumps(pair_info)};
(function(){{
  const gd = document.querySelector('.plotly-graph-div');
  const box = document.getElementById('pair-info');
  const clearBtn = document.getElementById('clear-selection');

  if(!gd || !box || !clearBtn) {{
      console.error("Plotly elements not found.");
      return;
  }}

  let selected = [];  // up to 2 nodes
  const START_MSG = 'Click a node, then another...';
  const nodeTraceIndex = 1;  // the node trace index in your figure

  function clearSelection() {{
      selected = [];
      box.textContent = START_MSG;
      // Reset marker highlighting
      Plotly.restyle(gd, {{ selectedpoints: [null] }}, [nodeTraceIndex]);
  }}

  clearBtn.addEventListener('click', clearSelection);

  gd.on('plotly_click', function(eventData) {{
      const point = eventData.points[0];
      if(!point || point.curveNumber !== nodeTraceIndex) return;

      const node = point.customdata;

      // Reset if node already selected
      if(selected.includes(node)) return;

      // Add node or reset if already 2 nodes
      if(selected.length < 2){{
          selected.push(node);
      }} else {{
          selected = [node];
      }}

      // Highlight currently selected nodes
      const selectedIndices = gd.data[nodeTraceIndex].customdata
          .map((n, i) => selected.includes(n) ? i : null)
          .filter(i => i !== null);
      Plotly.restyle(gd, {{ selectedpoints: [selectedIndices] }}, [nodeTraceIndex]);

      // Update dashboard
      if(selected.length === 1){{
          box.textContent = 'Node: ' + node + ' (Click another)';
      }} else if(selected.length === 2){{
          const a = selected[0], b = selected[1];
          const key1 = a + '|' + b;
          const key2 = b + '|' + a;
          const rec = DATA_PAIR[key1] || DATA_PAIR[key2];
          const p = 4;

          if(rec){{
              // *** FIX IS HERE: Escaped JS template literals ***
              box.textContent =
                `Nodes: ${{a}} — ${{b}}\n` +
                `Euclidean: ${{rec.euclidean.toFixed(p)}}\n` +
                `Toroidal: ${{rec.toroidal.toFixed(p)}}\n` +
                `Reff: ${{rec.reff.toFixed(p)}}\n` +
                `(1/deg_u + 1/deg_v): ${{rec.inv_deg_sum.toFixed(p)}}`;
          }} else {{
              box.textContent = `Nodes: ${{a}} — ${{b}}\n(Not in the same component)`;
          }}
      }}
  }});

  // Initialize dashboard
  clearSelection();
}})();
</script>
"""

        with open(filename, "r", encoding="utf-8") as f:
            html = f.read()

        # --- UPDATED INJECTION ---
        # Inject styles, dashboard HTML, and JS right before </body>
        injection_payload = style_block + info_box_html + js
        html = html.replace("</body>", injection_payload + "</body>")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)


    # ---------------------------------------------------------
    # Main API
    # ---------------------------------------------------------
    def show_html(self, filename="rgg.html", largest_gc=False, metric="toroidal"):
        if self.G is None:
            raise ValueError("No graph set.")

        # Gvis is now the graph we intend to visualize (either full or largest_gc)
        Gvis = self._get_component(largest_gc)
        pos = self._ensure_pos(Gvis)
        
        print(f"Computing pair info for {Gvis.number_of_nodes()} nodes... (This may take a moment)")
        pair_info = self._compute_pair_info(Gvis, metric)
        print("Computation complete.")

        self._write_plotly_html(filename, Gvis, pos, pair_info, metric)
        return filename