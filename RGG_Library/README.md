# RGG Library

**Author:** Filip Rupchin

A lightweight Python library for building, analyzing, and visualizing **Random Geometric Graphs (RGGs)**. Supports multiple graph spaces (unit square, torus, lattices), graph analysis via spectral methods, and effective resistance / commute time sampling.

---

## Dependencies

- `networkx`
- `numpy`
- `scipy`
- `matplotlib`

---

## Classes

### `RGGBuilder`

The main class for constructing random geometric graphs and running graph-theoretic analyses on them.

#### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `n` | `int` | Number of nodes to generate |
| `k` | `float` | Expected mean degree of a node; used to derive the connection radius `r` |
| `connectivity_regime` | `str` | Controls how the radius `r` is computed (see below) |
| `space` | `str` | The geometric space on which the graph is built (see below) |
| `seed` | `int` | Random seed for reproducible graph generation |
| `order` | `int` | Number of "shells" of nearest neighbours connected to each node in lattice graphs |
| `perturb` | `bool` | If `True`, perturbs lattice node positions using a bivariate normal distribution |
| `perturb_scale` | `float` | Standard deviation of the perturbation distribution |
| `perturb_radius_multiplier` | `float` | Scales the connection radius for the lattice cases |

#### `connectivity_regime` Options

| Value | Radius Formula | Description |
|---|---|---|
| `"sc"` | `r = sqrt(k / ((n-1) * π))` | **Supercritical** regime — graph is likely connected but not guaranteed |
| `"c"` | `r = sqrt(k * log(n) / ((n-1) * π))` | **Connected** regime — radius grows with `log(n)` to ensure full connectivity |

#### `space` Options

| Value | Description |
|---|---|
| `"unit_square"` | Unit square `[0,1]²` with hard boundary conditions |
| `"torus"` | Unit torus with periodic boundary conditions (no edge effects) |
| `"triangular_torus"` | Triangular lattice embedded on the unit torus |
| `"square_torus"` | Square lattice embedded on the unit torus |

---

## Static Methods

### Graph Statistics & Visualization

---

#### `print_graph_stats(G, radius=None)`

Prints a summary of key structural properties of a graph.

```python
RGGBuilder.print_graph_stats(G, radius=0.15)
```

**Parameters:**
- `G` (`nx.Graph`): The input graph.
- `radius` (`float`, optional): If provided, the connection radius is printed alongside the stats.

**Output includes:**
- Node and edge counts
- Average, min, and max degree
- Graph density
- Average clustering coefficient
- Largest connected component size (as % of total nodes)

---

#### `plot_degree_distribution(G)`

Plots a histogram of the degree distribution of the graph.

```python
RGGBuilder.plot_degree_distribution(G)
```

**Parameters:**
- `G` (`nx.Graph`): The input graph.

Bins are centered on integer degree values. Displays the plot via `matplotlib`.

---

### Distance & Geometry

---

#### `toroidal_distance(pos1, pos2)`

Computes the Euclidean distance between two points under **toroidal (periodic) boundary conditions** — i.e., the minimum image convention on a unit torus.

```python
d = RGGBuilder.toroidal_distance(np.array([0.05, 0.5]), np.array([0.95, 0.5]))
# Returns 0.1 rather than 0.9
```

**Parameters:**
- `pos1`, `pos2` (`np.ndarray`): 2D coordinate arrays.

**Returns:** `float` — the toroidal distance.

---

### Spectral / Resistance Analysis

---

#### `laplacian_sparse(G)`

Constructs the **sparse graph Laplacian** `L = D - A` for a graph.

```python
L, degs = RGGBuilder.laplacian_sparse(G)
```

**Parameters:**
- `G` (`nx.Graph`): The input graph.

**Returns:**
- `L` (`sp.csr_matrix`): The sparse Laplacian matrix.
- `degs` (`np.ndarray`): Degree sequence of the nodes.

---

#### `effective_resistance_pair(L, u, v)`

Computes the **effective resistance** between a single pair of nodes `u` and `v` by solving a grounded linear system derived from the Laplacian.

```python
Reff = RGGBuilder.effective_resistance_pair(L, u=0, v=5)
```

**Parameters:**
- `L` (`sp.csr_matrix`): The sparse Laplacian (from `laplacian_sparse`).
- `u`, `v` (`int`): Node indices.

**Returns:** `float` — the effective resistance `R_eff(u, v)`. Returns `nan` if the system is singular (e.g. disconnected graph).

**Note:** This method grounds node `n-1` and solves a reduced `(n-1) × (n-1)` system, which is efficient for sparse graphs.

---

#### `sample_commute_times_even_distance(G, nsamples, n_bins, seed, min_dist, max_dist)`

Samples node pairs **uniformly across distance bins** and computes effective resistance and degree-based predictions for each pair. Pairs closer than `min_dist` or farther than `max_dist` are excluded.

```python
res, preds, dists, pairs = RGGBuilder.sample_commute_times_even_distance(
    G, nsamples=500, n_bins=20, seed=42, min_dist=0.05, max_dist=0.5
)
```

**Parameters:**
- `G` (`nx.Graph`): Input graph with `"pos"` node attributes.
- `nsamples` (`int`): Total number of pairs to sample (spread evenly across bins).
- `n_bins` (`int`): Number of distance bins.
- `seed` (`int`): Random seed.
- `min_dist` (`float`): Minimum toroidal distance; pairs closer than this are excluded.
- `max_dist` (`float`): Maximum toroidal distance; pairs farther than this are excluded.

**Returns:** A 4-tuple:
- `res` (`np.ndarray`): Effective resistances for sampled pairs.
- `preds` (`np.ndarray`): Degree-based predictions `1/deg(u) + 1/deg(v)` for each pair.
- `dists` (`np.ndarray`): Toroidal distances for sampled pairs.
- `pairs` (`list`): List of `(u, v)` node index tuples.

If the graph is disconnected, sampling is performed on the largest connected component with a warning.

---

#### `sample_commute_times_even_distance_w_angles(G, nsamples, n_bins, seed, min_dist, max_dist)`

Identical to `sample_commute_times_even_distance`, but additionally computes the **angle** of the displacement vector between each sampled pair. Angles are normalized to `[0, π)` to account for the undirected nature of the graph.

```python
res, preds, dists, pairs, angles = RGGBuilder.sample_commute_times_even_distance_w_angles(
    G, nsamples=500, n_bins=20, seed=42, min_dist=0.05, max_dist=0.5
)
```

**Parameters:** Same as `sample_commute_times_even_distance`.

**Returns:** A 5-tuple — same as above, plus:
- `angles` (`np.ndarray`): Angles (radians, in `[0, π)`) of displacement vectors for sampled pairs. Useful for detecting anisotropy in the resistance structure.

---

#### `compute_all_pairs_metrics(G)`

Computes effective resistance, degree-based predictions, and toroidal distances for **all** `N(N-1)/2` node pairs using the **Moore-Penrose pseudoinverse** of the Laplacian.

```python
res, deg_sums, dists, pairs = RGGBuilder.compute_all_pairs_metrics(G)
```

**Parameters:**
- `G` (`nx.Graph`): A connected graph with integer node labels `0` to `N-1` and `"pos"` node attributes.

**Returns:** A 4-tuple:
- `res` (`np.ndarray`): Effective resistance for every pair.
- `deg_sums` (`np.ndarray`): `1/deg(u) + 1/deg(v)` for every pair.
- `dists` (`np.ndarray`): Toroidal distances for every pair.
- `pairs` (`np.ndarray`, shape `[N*(N-1)/2, 2]`): All `(u, v)` index pairs.

> ⚠️ **Performance Warning:** This method computes the dense pseudoinverse, which is `O(N³)` in time and `O(N²)` in memory. It becomes very slow for graphs with `N > 2000–3000` nodes. For large graphs, prefer `sample_commute_times_even_distance`.

---

## Usage Example

```python
builder = RGGBuilder(n=200, k=6, connectivity_regime="c", space="torus", seed=0)
G = builder.build()

RGGBuilder.print_graph_stats(G, radius=builder.radius)
RGGBuilder.plot_degree_distribution(G)

res, preds, dists, pairs = RGGBuilder.sample_commute_times_even_distance(
    G, nsamples=300, n_bins=15, seed=1, min_dist=0.05
)
```