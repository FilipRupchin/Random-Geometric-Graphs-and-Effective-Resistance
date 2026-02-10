import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt
import typing as _t
import warnings

try:
    from scipy.spatial import cKDTree  # optional fast neighbor search
except Exception:
    cKDTree = None

class RGGBuilder:
    """
    Build Random Geometric Graphs (RGGs) on several spaces.
    """
    def __init__(self, n: int, k: float, connectivity_regime: str = 'c',
                 space: str = 'torus', seed: _t.Optional[int] = None, 
                 order: int = 1, perturb: bool = False, 
                 perturb_scale: float = 0.01, perturb_radius_multiplier: float = 1.5):    
        
        self.n = int(n)
        if self.n <= 1:
            raise ValueError("n must be > 1 for radius formulas that use (n-1) in denominator")
        
        self.k = float(k)
        if not (self.k > 0 and np.isfinite(self.k)):
            raise ValueError("k must be a positive finite number")
        
        # sc for fixed mean degree / c for fully connected
        self.connectivity_regime = connectivity_regime
        # Options: unit_square, torus, triangular_torus, square_torus
        self.space = space
        # Store shell order for triangular_torus and square_torus
        self.order = order
        # Whether to perturb lattice positions
        self.perturb = perturb
        # Standard deviation for perturbation
        self.perturb_scale = perturb_scale
        # Multiply radius for perturbed graphs
        self.perturb_radius_multiplier = perturb_radius_multiplier
        # Seed setting
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.seed)
        
        # Validate perturbation parameters
        if self.perturb and self.perturb_scale <= 0:
            raise ValueError("perturb_scale must be positive when perturb=True")
        if self.perturb and self.perturb_radius_multiplier <= 0:
            raise ValueError("perturb_radius_multiplier must be positive when perturb=True")
        
        # Compute radius according to selected space and regime
        self.radius = self._compute_radius()
        
        # Error check: radius
        if not (np.isfinite(self.radius) and self.radius > 0):
            raise ValueError(f"Computed radius is invalid: {self.radius!r}")
    
    def _compute_radius(self) -> float:
        """
        Compute the connection radius based on space type and connectivity regime.
        
        For triangular_torus and square_torus: radius is based on the order-th shell distance
        For other spaces: radius follows the standard RGG formulas
        """
        
        if self.space == 'triangular_torus':
            # For triangular lattice, calculate the actual shell distance
            cols = int(np.round(np.sqrt(self.n)))
            rows = int(np.ceil(self.n / cols))
            
            # Spacing in the triangular lattice
            dx = 1.0 / cols
            dy = 1.0 / rows
            
            # Calculate distance to order-th shell
            shell_dist = self._get_triangular_shell_distance(self.order, dx, dy)
            
            # Add small buffer to ensure connections (1% extra)
            return shell_dist * 1.01
            
        elif self.space == 'square_torus':
            # For square lattice, calculate the shell distance
            cols = int(np.round(np.sqrt(self.n)))
            rows = int(np.ceil(self.n / cols))
            
            dx = 1.0 / cols
            dy = 1.0 / rows
            
            shell_dist = self._get_square_shell_distance(self.order, dx, dy)
            
            # Add small buffer to ensure connections (1% extra)
            return shell_dist * 1.01
            
        else:
            # Standard RGG radius formulas for torus and unit_square
            if self.connectivity_regime == 'sc':
                # supercritical: target expected degree k -> r = sqrt( k / ((n-1) * pi) )
                return np.sqrt(self.k / ((self.n - 1) * np.pi))
            elif self.connectivity_regime == 'c':
                # connectivity regime: r = sqrt( k * log(n) / ((n-1) * pi) )
                return np.sqrt(self.k * np.log(self.n) / ((self.n - 1) * np.pi))
            else:
                raise ValueError(f"Unknown connectivity_regime '{self.connectivity_regime}'")
    
    def _get_lattice_shell_distances_and_offsets(self, lattice_type: str, order: int, 
                                                  dx: float = 1.0, dy: float = 1.0, 
                                                  is_odd_row: bool = False):
        """
        Compute shell offsets.
        FIX: Uses IDEALIZED geometry for sorting (to ensure topology is correct)
        but returns ACTUAL distance (for radius calculation).
        """
        scan_range = 2 * order + 2
        
        if lattice_type == 'square':
            offsets_with_dist = []
            for dr in range(-scan_range, scan_range + 1):
                for dc in range(-scan_range, scan_range + 1):
                    if dr == 0 and dc == 0: continue
                    
                    actual_dx = dc * dx
                    actual_dy = dr * dy
                    dist_sq = actual_dx**2 + actual_dy**2
                    
                    # For square, ideal and actual sort order is usually the same
                    # unless dx/dy are wildly different. 
                    offsets_with_dist.append((dist_sq, (dr, dc)))
            
            offsets_with_dist.sort(key=lambda x: x[0])
            
            shells_dist_sq = []
            shells_offsets = []
            
            if not offsets_with_dist: return [], []
            
            current_dist = offsets_with_dist[0][0]
            current_shell = []
            
            for dist_sq, offset in offsets_with_dist:
                if np.isclose(dist_sq, current_dist, atol=1e-9):
                    current_shell.append(offset)
                else:
                    shells_dist_sq.append(current_dist)
                    shells_offsets.append(current_shell)
                    current_dist = dist_sq
                    current_shell = [offset]
                if len(shells_dist_sq) >= order: break
            
            if len(shells_dist_sq) < order and current_shell:
                shells_dist_sq.append(current_dist)
                shells_offsets.append(current_shell)
            
            return shells_dist_sq, shells_offsets
        
        elif lattice_type == 'triangular':
            # Store tuple: (IDEAL_DIST_SQ, ACTUAL_DIST_SQ, OFFSET)
            candidates = []
            
            src_x_shift = 0.5 if is_odd_row else 0.0
            
            # Perfect equilateral constants for sorting topology
            ideal_dx = 1.0
            ideal_dy = np.sqrt(3) / 2.0
            
            for dr in range(-scan_range, scan_range + 1):
                for dc in range(-scan_range, scan_range + 1):
                    if dr == 0 and dc == 0: continue
                    
                    # --- 1. Calculate Offsets ---
                    target_is_odd = (dr % 2 != 0) if not is_odd_row else (dr % 2 == 0)
                    tgt_x_shift = 0.5 if target_is_odd else 0.0
                    
                    # Horizontal delta in "column units"
                    d_col = (dc + tgt_x_shift) - src_x_shift
                    
                    # --- 2. Idealized Distance (For Sorting/Grouping) ---
                    # This ensures (1,0) and the diagonals fall in the same shell
                    # even if the physical torus is slightly stretched.
                    ideal_d_sq = (d_col * ideal_dx)**2 + (dr * ideal_dy)**2
                    
                    # --- 3. Actual Physical Distance (For Radius) ---
                    actual_d_sq = (d_col * dx)**2 + (dr * dy)**2
                    
                    candidates.append((ideal_d_sq, actual_d_sq, (dr, dc)))
            
            # Sort by IDEAL distance to preserve topology (6 neighbors in shell 1)
            candidates.sort(key=lambda x: x[0])
            
            shells_max_actual_dist_sq = []
            shells_offsets = []
            
            if not candidates: return [], []
            
            current_ideal_dist = candidates[0][0]
            current_shell_offsets = []
            current_shell_actual_dists = []
            
            for ideal_d, actual_d, offset in candidates:
                # Group by ideal distance (using a looser tolerance for float noise)
                if np.isclose(ideal_d, current_ideal_dist, atol=1e-5):
                    current_shell_offsets.append(offset)
                    current_shell_actual_dists.append(actual_d)
                else:
                    # Save completed shell
                    # Use the MAX actual distance in this shell to ensure radius covers all
                    shells_max_actual_dist_sq.append(max(current_shell_actual_dists))
                    shells_offsets.append(current_shell_offsets)
                    
                    # Start new shell
                    current_ideal_dist = ideal_d
                    current_shell_offsets = [offset]
                    current_shell_actual_dists = [actual_d]
                
                if len(shells_offsets) >= order: break
            
            # Last shell
            if len(shells_offsets) < order and current_shell_offsets:
                shells_max_actual_dist_sq.append(max(current_shell_actual_dists))
                shells_offsets.append(current_shell_offsets)
                
            return shells_max_actual_dist_sq, shells_offsets
    
    def _get_triangular_shell_distance(self, order: int, dx: float, dy: float) -> float:
        """
        Calculate actual distance for order-th shell in triangular lattice.
        Distances are computed directly in actual coordinates - no rescaling needed.
        """
        # Compute with is_odd_row=False (distances are the same for both)
        shells_dist_sq, _ = self._get_lattice_shell_distances_and_offsets(
            'triangular', order, dx=dx, dy=dy, is_odd_row=False
        )
        
        if order > len(shells_dist_sq):
            warnings.warn(f"Requested order {order} exceeds computed shells {len(shells_dist_sq)}.")
            dist_sq = shells_dist_sq[-1] if shells_dist_sq else (order * dx) ** 2
        else:
            dist_sq = shells_dist_sq[order - 1]  # order is 1-indexed
        
        return np.sqrt(dist_sq)
    
    def _get_square_shell_distance(self, order: int, dx: float, dy: float) -> float:
        """
        Calculate actual distance for order-th shell in square lattice.
        Distances are computed directly in actual coordinates - no rescaling needed.
        """
        shells_dist_sq, _ = self._get_lattice_shell_distances_and_offsets(
            'square', order, dx=dx, dy=dy
        )
        
        if order > len(shells_dist_sq):
            warnings.warn(f"Requested order {order} exceeds computed shells {len(shells_dist_sq)}.")
            dist_sq = shells_dist_sq[-1] if shells_dist_sq else (order * min(dx, dy)) ** 2
        else:
            dist_sq = shells_dist_sq[order - 1]  # order is 1-indexed
        
        return np.sqrt(dist_sq)
            
    def _generate_triangular_torus_positions(self) -> _t.Tuple[np.ndarray, int, int]:
        """
        Generates a triangular lattice that perfectly fills the unit torus.
        
        Adjustments made:
        1. Updates self.n to (rows * cols) so there are no missing nodes.
        2. Stretches the y-axis slightly to fill the [0,1] range (closes the topological gap).
        3. Optionally perturbs positions with Gaussian noise if self.perturb=True.
        """
        # 1. Determine optimal grid size close to requested n
        cols = int(np.round(np.sqrt(self.n)))
        rows = int(np.ceil(self.n / cols))
        
        # 2. Update self.n to fill the perfect grid
        new_n = rows * cols
        if new_n != self.n:
            print(f"  [RGGBuilder] Adjusting n from {self.n} to {new_n} to fill lattice.")
            self.n = new_n
            # Recalculate radius with updated n
            self.radius = self._compute_radius()
        
        # 3. Define spacing to fill [0, 1] x [0, 1] perfectly
        dx = 1.0 / cols
        dy = 1.0 / rows
        pos = np.empty((self.n, 2), dtype=float)
        idx = 0
        
        for r in range(rows):
            # Triangular offset: shift odd rows by half a column
            x_offset = 0.5 * dx if (r % 2 == 1) else 0.0
            
            y = r * dy 
            
            for c in range(cols):
                x = c * dx + x_offset
                
                # Wrap x if the offset pushed it slightly over 1.0 
                pos[idx] = (x % 1.0, y % 1.0)
                idx += 1
        
        # 4. Apply perturbation if requested
        if self.perturb:
            # print(f"  [RGGBuilder] Applying Gaussian perturbation with scale={self.perturb_scale:.4f}")
            self.rng = np.random.default_rng(self.seed)
            # Add Gaussian noise to each coordinate
            noise = self.rng.normal(0, self.perturb_scale, size=(self.n, 2))
            pos = pos + noise
            # Wrap positions back to [0, 1) for toroidal topology
            pos = pos % 1.0
        
        return pos, rows, cols
    
    def _pairs_triangular_torus(self, rows: int, cols: int, order: int = 1) -> _t.Set[_t.Tuple[int, int]]:
        """Generate edges for triangular lattice torus up to the specified order."""
        pairs = set()
        
        def idx(r, c):
            return (r % rows) * cols + (c % cols)
        
        # Get offsets for even and odd rows
        _, even_shells_offsets = self._get_lattice_shell_distances_and_offsets('triangular', order, is_odd_row=False)
        _, odd_shells_offsets = self._get_lattice_shell_distances_and_offsets('triangular', order, is_odd_row=True)
        
        # Flatten shells up to order
        even_neighbors = []
        odd_neighbors = []
        
        for i in range(min(len(even_shells_offsets), order)):
            even_neighbors.extend(even_shells_offsets[i])
        
        for i in range(min(len(odd_shells_offsets), order)):
            odd_neighbors.extend(odd_shells_offsets[i])
        
        # Build edges
        for r in range(rows):
            nbrs_offsets = odd_neighbors if (r % 2 != 0) else even_neighbors
            
            for c in range(cols):
                u = idx(r, c)
                if u >= self.n:
                    continue
                
                for (dr, dc) in nbrs_offsets:
                    rr = (r + dr) % rows
                    cc = (c + dc) % cols
                    v = idx(rr, cc)
                    
                    if v < self.n and u < v:
                        pairs.add((u, v))
        
        return pairs
    
    def _generate_square_torus_positions(self) -> _t.Tuple[np.ndarray, int, int]:
        """
        Generates a square lattice that perfectly fills the unit torus.
        
        Returns:
            - pos: (n, 2) array of positions
            - rows: number of rows in the grid
            - cols: number of columns in the grid
        """
        # 1. Determine optimal grid size close to requested n
        cols = int(np.round(np.sqrt(self.n)))
        rows = int(np.ceil(self.n / cols))
        
        # 2. Update self.n to fill the perfect grid
        new_n = rows * cols
        if new_n != self.n:
            print(f"  [RGGBuilder] Adjusting n from {self.n} to {new_n} to fill square lattice.")
            self.n = new_n
            # Recalculate radius with updated n
            self.radius = self._compute_radius()
        
        # 3. Define spacing to fill [0, 1] x [0, 1] perfectly
        dx = 1.0 / cols
        dy = 1.0 / rows
        
        pos = np.empty((self.n, 2), dtype=float)
        idx = 0
        
        for r in range(rows):
            y = r * dy
            for c in range(cols):
                x = c * dx
                pos[idx] = (x, y)
                idx += 1
        
        # 4. Apply perturbation if requested
        if self.perturb:
            # print(f"  [RGGBuilder] Applying Gaussian perturbation with scale={self.perturb_scale:.4f}")
            self.rng = np.random.default_rng(self.seed)
            noise = self.rng.normal(0, self.perturb_scale, size=(self.n, 2))
            pos = (pos + noise) % 1.0
        
        return pos, rows, cols

    def _pairs_square_torus(self, rows: int, cols: int, order: int = 1) -> _t.Set[_t.Tuple[int, int]]:
        """Generate edges for square lattice torus up to the specified order."""
        pairs = set()
        
        def idx(r, c):
            return (r % rows) * cols + (c % cols)
        
        # Get offsets for all shells up to order
        _, shells_offsets = self._get_lattice_shell_distances_and_offsets('square', order)
        
        # Flatten all shells up to order
        neighbor_offsets = []
        for i in range(min(len(shells_offsets), order)):
            neighbor_offsets.extend(shells_offsets[i])
        
        # Build edges
        for r in range(rows):
            for c in range(cols):
                u = idx(r, c)
                if u >= self.n:
                    continue
                
                for (dr, dc) in neighbor_offsets:
                    rr = (r + dr) % rows
                    cc = (c + dc) % cols
                    v = idx(rr, cc)
                    
                    if v < self.n and u < v:
                        pairs.add((u, v))
        
        return pairs
    
    def generate_positions(self) -> np.ndarray:
        if self.space == "triangular_torus":
            pos, self._tri_rows, self._tri_cols = self._generate_triangular_torus_positions()
            return pos
        elif self.space == "square_torus":
            pos, self._sq_rows, self._sq_cols = self._generate_square_torus_positions()
            return pos
        return self.rng.uniform(0, 1, size=(self.n, 2))
    
    def build(self, positions: _t.Optional[np.ndarray] = None, use_kdtree: bool = True) -> nx.Graph:
        """
        Build and return a networkx.Graph with node attribute 'pos' set.
        - positions: optional precomputed positions (overrides generator)
        - use_kdtree: attempt to use scipy.spatial.cKDTree when available
        - order: (triangular_torus/square_torus only) depth of neighbors to connect
                 If None, uses self.order. Only used for unperturbed lattices.
        """
        order = self.order
            
        # Warn if KDTree requested but not available for large n
        if use_kdtree and cKDTree is None and self.n > 2000:
            warnings.warn("scipy.spatial.cKDTree not available; falling back to O(n^2) pairwise checks; "
                          "consider installing scipy for large n", RuntimeWarning)
        # normalize and validate positions
        pos = positions if positions is not None else self.generate_positions()
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[0] != self.n or pos.shape[1] != 2:
            raise ValueError(f"positions must be an array of shape (n,2); got {pos.shape}")
        
        # For torus, reduce coords to [0,1)
        if self.space in ["torus", "triangular_torus", "square_torus"]:
            pos = pos % 1.0
        
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        # build mapping node -> tuple(position) for networkx
        pos_map = {i: (float(pos[i,0]), float(pos[i,1])) for i in range(self.n)}
        nx.set_node_attributes(G, pos_map, "pos")
        
        if self.space == "unit_square":
            pairs = self._pairs_euclidean(pos, use_kdtree=use_kdtree)

        elif self.space == "torus":
            pairs = self._pairs_torus(pos, use_kdtree=use_kdtree)

        elif self.space == "triangular_torus":
            # If perturbed, use radius-based connection on the torus
            # Otherwise, use the lattice-based connection
            if self.perturb:
                # Apply radius multiplier for perturbed graphs to maintain connectivity
                effective_radius = self.radius * self.perturb_radius_multiplier
                #print(f"  [RGGBuilder] Using radius-based connection for perturbed lattice")
                #print(f"  [RGGBuilder] Base radius={self.radius:.6f}, effective radius={effective_radius:.6f} (multiplier={self.perturb_radius_multiplier:.2f})")
                
                # Temporarily override radius for connection
                original_radius = self.radius
                self.radius = effective_radius
                pairs = self._pairs_torus(pos, use_kdtree=use_kdtree)
                self.radius = original_radius  # Restore original radius
            else:
                # Pass the order parameter for unperturbed lattice
                pairs = self._pairs_triangular_torus(self._tri_rows, self._tri_cols, order=order)

        elif self.space == "square_torus":
            # If perturbed, use radius-based connection on the torus
            # Otherwise, use the lattice-based connection
            if self.perturb:
                # Apply radius multiplier for perturbed graphs to maintain connectivity
                effective_radius = self.radius * self.perturb_radius_multiplier
                #print(f"  [RGGBuilder] Using radius-based connection for perturbed lattice")
                #print(f"  [RGGBuilder] Base radius={self.radius:.6f}, effective radius={effective_radius:.6f} (multiplier={self.perturb_radius_multiplier:.2f})")
                
                # Temporarily override radius for connection
                original_radius = self.radius
                self.radius = effective_radius
                pairs = self._pairs_torus(pos, use_kdtree=use_kdtree)
                self.radius = original_radius  # Restore original radius
            else:
                # Pass the order parameter for unperturbed lattice
                pairs = self._pairs_square_torus(self._sq_rows, self._sq_cols, order=order)
        else:
            raise ValueError(f"Unsupported space '{self.space}'")
        
        # add edges (undirected)
        for i, j in pairs:
            G.add_edge(i, j)
        return G

    def _pairs_euclidean(self, pos: np.ndarray, use_kdtree: bool) -> _t.Set[_t.Tuple[int, int]]:
        """
        Finds pairs within radius in Euclidean space.
        Optimized to remove redundant sorting.
        """
        if use_kdtree and cKDTree is not None:
            tree = cKDTree(pos)
            # query_pairs returns a set of sorted tuples (i, j) where i < j
            # No need to sort or cast again.
            return tree.query_pairs(self.radius)
        
        # Fallback: Dense pairwise distance
        # 1. Compute displacement
        disp = pos[:, None, :] - pos[None, :, :]
        # 2. Compute squared distance
        d2 = np.sum(disp**2, axis=-1)
        # 3. Find indices where dist < radius (exclude diagonal k=1)
        idx_i, idx_j = np.where(np.triu(d2, k=1) <= (self.radius**2))
        
        return set(zip(idx_i.tolist(), idx_j.tolist()))

    def _pairs_torus(self, pos: np.ndarray, use_kdtree: bool) -> _t.Set[_t.Tuple[int, int]]:
        """
        Finds pairs within radius in Toroidal space.
        Fixed: Vectorized index mapping (10x-50x faster) and memory efficient.
        """
        n = pos.shape[0]
        
        if use_kdtree and cKDTree is not None:
            # 1. Create the 3x3 grid of offsets
            shifts = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
            
            # 2. Tile positions: Vectorized broadcast add
            # Shape becomes (9*N, 2)
            # Row i in 'pos' corresponds to rows i, i+n, i+2n, ... in 'tiled'
            tiled_pos = np.vstack([pos + s for s in shifts])
            
            # 3. Build Tree on tiled space
            tree = cKDTree(tiled_pos)
            
            # 4. Query pairs
            # This returns pairs of indices in the range [0, 9*N)
            raw_pairs = tree.query_pairs(self.radius)
            
            if not raw_pairs:
                return set()
            
            # 5. Vectorized Mapping back to [0, N)
            # Convert set to array for fast processing
            raw_arr = np.array(list(raw_pairs))
            
            # Map tiled indices back to original node indices using modulo
            u = raw_arr[:, 0] % n
            v = raw_arr[:, 1] % n
            
            # 6. Filter & Deduplicate
            # Remove self-loops (node connecting to its own ghost image)
            mask = u != v
            u, v = u[mask], v[mask]
            
            # Sort pairs (min, max) to ensure (u,v) is same as (v,u)
            final_u = np.minimum(u, v)
            final_v = np.maximum(u, v)
            
            # Combine and find unique rows
            # This handles cases where nodes connect via multiple boundaries simultaneously
            combined = np.vstack((final_u, final_v)).T
            unique_pairs = np.unique(combined, axis=0)
            
            # Return as set of tuples
            return set(map(tuple, unique_pairs))
        
        # Fallback: Dense Toroidal Distance
        # 1. Absolute difference
        diff = np.abs(pos[:, None, :] - pos[None, :, :])
        
        # 2. Toroidal wrap: min(dx, 1-dx)
        # We ensure 1.0-diff is not negative due to floating point precision
        dx = np.minimum(diff, 1.0 - diff)
        
        # 3. Squared Euclidean distance on the wrapped coords
        d2 = np.sum(dx**2, axis=-1)
        
        # 4. Filter
        idx_i, idx_j = np.where(np.triu(d2, k=1) <= (self.radius**2))
        
        return set(zip(idx_i.tolist(), idx_j.tolist()))
    
    # -----------------------------------------------------------------
    # Graph statistics
    # -----------------------------------------------------------------
    @staticmethod
    def print_graph_stats(G: nx.Graph, radius: _t.Optional[float] = None):
        degrees = [d for n, d in G.degree()]
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_nodes == 0:
            print("Graph Statistics: Empty graph (0 nodes)")
            return
        avg_degree = np.mean(degrees)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        # Find the largest connected component
        components = list(nx.connected_components(G))
        if components:
            largest_comp = max(components, key=len)
            largest_comp_size = len(largest_comp)
        else:
            largest_comp_size = 0
        print(f"Graph Statistics:")
        # --- NEW: Print radius if provided ---
        if radius is not None:
            print(f"  Radius: {radius:.6f}")
        # -------------------------------------
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")
        print(f"  Average degree: {avg_degree:.3f}")
        print(f"  Min degree: {min_degree}")
        print(f"  Max degree: {max_degree}")
        print(f"  Density: {density:.6f}")
        print(f"  Avg clustering coefficient: {clustering:.4f}")
        print(f"  Largest component size: {largest_comp_size/n_nodes*100:.2f}%")
        
    # -----------------------------------------------------------------
    # Degree distribution plot
    # -----------------------------------------------------------------
    @staticmethod
    def plot_degree_distribution(G: nx.Graph):
        degrees = [d for n, d in G.degree()]
        if not degrees:
            print("Graph has no nodes or edges.")
            return
        max_deg = max(degrees)
        bins = np.arange(0, max_deg + 2) - 0.5  # bins centered on integers
        plt.figure(figsize=(10,6))
        plt.hist(degrees, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel("Degree")
        plt.ylabel("Number of nodes")
        plt.title("Degree Distribution")
        plt.xticks(range(0, max_deg + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
    
    # -----------------------------------------------------------------
    # Graph Analysis (Laplacian, Resistance)
    # -----------------------------------------------------------------
    @staticmethod
    def toroidal_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        diff = np.abs(pos1 - pos2)
        toroidal_diff = np.where(diff > 0.5, 1.0 - diff, diff)
        return np.linalg.norm(toroidal_diff)
    
    @staticmethod
    def laplacian_sparse(G: nx.Graph) -> _t.Tuple[sp.csr_matrix, np.ndarray]:
        A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
        degs = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degs)
        L = D - A
        return L, degs
    
    @staticmethod
    def effective_resistance_pair(L: sp.csr_matrix, u: int, v: int) -> float:
        n = L.shape[0]
        b = np.zeros(n)
        b[u] = 1
        b[v] = -1  
        Lg = L[:-1, :-1]
        bg = b[:-1]
        
        try:
            x = spla.spsolve(Lg, bg)
            x = np.append(x, 0) # Add the 0 for the grounded node
            return x[u] - x[v]
        except spla.MatrixRankWarning:
            warnings.warn(f"Matrix is singular, graph may be disconnected. Reff between {u}-{v} failed.")
            return np.nan
        except Exception as e:
            warnings.warn(f"spsolve failed for {u}-{v}: {e}")
            return np.nan

    @staticmethod
    def sample_commute_times_even_distance(G: nx.Graph, nsamples: int = 500, 
                                           n_bins: int = 20, seed: int = 0,
                                           min_dist: float = 0.0, max_dist: float = 2.0) -> _t.Tuple:
        """
        Samples node pairs and computes metrics, excluding pairs closer than min_dist.
        """
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        
        if not nx.is_connected(G):
            warnings.warn("Graph is not connected. Sampling from largest component.")
            largest_cc = max(nx.connected_components(G), key=len)
            Gsub = G.subgraph(largest_cc).copy()
            Gsub = nx.convert_node_labels_to_integers(Gsub, ordering="sorted")
        else:
            Gsub = G
            
        if Gsub.number_of_nodes() < 2:
            warnings.warn("Graph component has < 2 nodes. Cannot sample pairs.")
            return np.array([]), np.array([]), np.array([]), []
        
        L, degs = RGGBuilder.laplacian_sparse(Gsub)
        pos = nx.get_node_attributes(Gsub, "pos")
        nodes = list(Gsub.nodes())
        
        # 1. Collect ALL pairs and distances (O(N^2))
        all_pairs_list = []
        all_dists_list = []
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                pu, pv = np.array(pos[u]), np.array(pos[v])
                d = RGGBuilder.toroidal_distance(pu, pv) 
                all_pairs_list.append((u, v))
                all_dists_list.append(d)
        
        if not all_dists_list:
            warnings.warn("No node pairs found.")
            return np.array([]), np.array([]), np.array([]), []
            
        all_pairs = np.array(all_pairs_list)
        all_dists = np.array(all_dists_list)
        
        # 2. FILTER: Exclude pairs within the radius (min_dist)
        if min_dist > 0:
            # Keep only distances strictly greater than min_dist
            mask_1 = all_dists > min_dist
            all_pairs = all_pairs[mask_1]
            all_dists = all_dists[mask_1]

        # 3. FILTER: Exclude pairs outside the max_dist
        if max_dist > min_dist:
            mask_2 = all_dists < max_dist
            all_pairs = all_pairs[mask_2]
            all_dists = all_dists[mask_2]
        
        # Check if we filtered everything out
        if len(all_dists) == 0:
            warnings.warn(f"No pairs found with distance > {min_dist}. Returning empty.")
            return np.array([]), np.array([]), np.array([]), []
        
        # 3. Binning and Sampling
        # Bins now cover the range [min(remaining), max(remaining)]
        bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
        rng = np.random.default_rng(seed)
        samples_per_bin = nsamples // n_bins
        pairs_sampled = []
        
        for i in range(n_bins):
            # Handle the last bin inclusive edge
            if i == n_bins - 1:
                 idx_bin = np.where((all_dists >= bins[i]) & (all_dists <= bins[i+1]))[0]
            else:
                 idx_bin = np.where((all_dists >= bins[i]) & (all_dists < bins[i+1]))[0]
            
            if len(idx_bin) == 0:
                continue
            
            n_to_choose = min(samples_per_bin, len(idx_bin))
            if n_to_choose == 0 and nsamples > 0:
                 n_to_choose = 1 
            chosen = rng.choice(idx_bin, n_to_choose, replace=False)
            pairs_sampled.extend(all_pairs[chosen])
        
        # 4. Compute Metrics for chosen pairs
        res, preds, dists_sampled = [], [], []
        for u, v in pairs_sampled:
            Reff = RGGBuilder.effective_resistance_pair(L, u, v)
            if np.isnan(Reff):
                continue
            
            res.append(Reff)
            preds.append((1 / degs[u]) + (1 / degs[v]))
            pu, pv = np.array(pos[u]), np.array(pos[v])
            dists_sampled.append(RGGBuilder.toroidal_distance(pu, pv))
        
        return np.array(res), np.array(preds), np.array(dists_sampled), pairs_sampled
    
    @staticmethod
    def sample_commute_times_even_distance_w_angles(G: nx.Graph, nsamples: int = 500, 
                                           n_bins: int = 20, seed: int = 0,
                                           min_dist: float = 0.0, max_dist: float = 2.0) -> _t.Tuple:
        """
        Samples node pairs and computes metrics, including angles, 
        excluding pairs closer than min_dist.
        """
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        
        if not nx.is_connected(G):
            warnings.warn("Graph is not connected. Sampling from largest component.")
            largest_cc = max(nx.connected_components(G), key=len)
            Gsub = G.subgraph(largest_cc).copy()
            Gsub = nx.convert_node_labels_to_integers(Gsub, ordering="sorted")
        else:
            Gsub = G
            
        if Gsub.number_of_nodes() < 2:
            warnings.warn("Graph component has < 2 nodes. Cannot sample pairs.")
            return np.array([]), np.array([]), np.array([]), [], np.array([])
        
        L, degs = RGGBuilder.laplacian_sparse(Gsub)
        pos = nx.get_node_attributes(Gsub, "pos")
        nodes = list(Gsub.nodes())
        
        # 1. Collect ALL pairs and distances (O(N^2))
        all_pairs_list = []
        all_dists_list = []
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                pu, pv = np.array(pos[u]), np.array(pos[v])
                d = RGGBuilder.toroidal_distance(pu, pv) 
                all_pairs_list.append((u, v))
                all_dists_list.append(d)
        
        if not all_dists_list:
            warnings.warn("No node pairs found.")
            return np.array([]), np.array([]), np.array([]), [], np.array([])
            
        all_pairs = np.array(all_pairs_list)
        all_dists = np.array(all_dists_list)
        
        # 2. FILTER: Exclude pairs within the radius (min_dist) and outside max_dist
        mask = (all_dists > min_dist) & (all_dists < max_dist)
        all_pairs = all_pairs[mask]
        all_dists = all_dists[mask]
        
        if len(all_dists) == 0:
            warnings.warn(f"No pairs found with distance > {min_dist}. Returning empty.")
            return np.array([]), np.array([]), np.array([]), [], np.array([])
        
        # 3. Binning and Sampling
        bins = np.linspace(all_dists.min(), all_dists.max(), n_bins + 1)
        rng = np.random.default_rng(seed)
        samples_per_bin = nsamples // n_bins
        pairs_sampled = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                 idx_bin = np.where((all_dists >= bins[i]) & (all_dists <= bins[i+1]))[0]
            else:
                 idx_bin = np.where((all_dists >= bins[i]) & (all_dists < bins[i+1]))[0]
            
            if len(idx_bin) == 0:
                continue
            
            n_to_choose = min(samples_per_bin, len(idx_bin))
            if n_to_choose == 0 and nsamples > 0:
                 n_to_choose = 1 
            chosen = rng.choice(idx_bin, n_to_choose, replace=False)
            pairs_sampled.extend(all_pairs[chosen])
        
        # 4. Compute Metrics for chosen pairs
        res, preds, dists_sampled, angles_sampled = [], [], [], []
        for u, v in pairs_sampled:
            Reff = RGGBuilder.effective_resistance_pair(L, int(u), int(v))
            if np.isnan(Reff):
                continue
            
            res.append(Reff)
            preds.append((1 / degs[u]) + (1 / degs[v]))
            
            pu, pv = np.array(pos[u]), np.array(pos[v])
            
            # 1. Toroidal Displacement (Minimum Image Convention)
            pu, pv = np.array(pos[u]), np.array(pos[v])
            disp = pv - pu
            toroidal_disp = disp - np.round(disp)
            
            dx = toroidal_disp[0]
            dy = toroidal_disp[1]
            
            # 2. Distance
            dists_sampled.append(np.sqrt(dx**2 + dy**2))
            
            # 3. Angle Calculation (Radians)
            # np.arctan2 returns values in the range [-pi, pi]
            theta = np.arctan2(dy, dx)
            
            # Normalize to [0, pi) (0 to 180 degrees)
            # Because the graph is undirected, a line at -45 deg is the same as 135 deg.
            if theta < 0:
                theta += np.pi
                
            angles_sampled.append(theta)
        
        # Returns: res, degs (preds), dists, pairs, angles
        return (np.array(res), 
                np.array(preds), 
                np.array(dists_sampled), 
                pairs_sampled, 
                np.array(angles_sampled))
    
    # -----------------------------------------------------------------
    # NEW FUNCTION
    # -----------------------------------------------------------------
    @staticmethod
    def compute_all_pairs_metrics(G: nx.Graph) -> _t.Tuple:
        """
        Computes Reff, 1/deg+1/deg, and distance for ALL node pairs
        using the O(N^3) pseudoinverse method.
        
        WARNING: This is computationally expensive and is very slow
        for graphs with N > 2000-3000. Use with caution.
        
        Assumes G is a single connected component and is labeled 0 to N-1.
        
        Returns:
            - res: array of all effective resistances (Reff)
            - degs: array of all (1/deg_u + 1/deg_v)
            - dists: array of all toroidal distances
            - pairs: array of (u, v) tuples
        """
        n = G.number_of_nodes()
        if n < 2:
            warnings.warn("Graph has < 2 nodes. Cannot compute pair metrics.")
            return (np.array([]), np.array([]), np.array([]), np.array([]))
            
        if not nx.is_connected(G):
            warnings.warn("Graph is not connected! Metrics may be incorrect or fail. "
                          "Pass only the largest connected component.", RuntimeWarning)
        
        # Ensure graph is labeled 0 to n-1
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        
        print(f"Starting all-pairs calculation for {n} nodes... (O(N^3), may be slow)")
        
        # 1. Get positions and degrees
        # We get degs from the sparse L, which is fast
        _, degs = RGGBuilder.laplacian_sparse(G) 
        pos = nx.get_node_attributes(G, "pos")
        nodes = list(G.nodes())
        
        # 2. Compute the O(N^3) pseudoinverse
        # We need the dense Laplacian for pinv
        L_dense = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        try:
            L_plus = np.linalg.pinv(L_dense)
        except Exception as e:
            warnings.warn(f"Pseudoinverse calculation failed: {e}")
            # Add a small identity matrix to stabilize
            L_plus = np.linalg.pinv(L_dense + np.eye(n) * 1e-12)
        
        print(f"Pseudoinverse calculated. Collating {n*(n-1)//2} pairs (O(N^2))...")
        
        # 3. Iterate all N^2 pairs
        num_pairs = n * (n - 1) // 2
        res = np.empty(num_pairs, dtype=float)
        deg_sums = np.empty(num_pairs, dtype=float)
        dists = np.empty(num_pairs, dtype=float)
        pairs = np.empty((num_pairs, 2), dtype=int)
        
        # Handle 0-degree nodes to avoid divide-by-zero
        degs_copy = degs.copy()
        degs_copy[degs_copy == 0] = np.inf
        inv_degs = 1.0 / degs_copy
        
        idx = 0
        for i in range(n):
            pos_i = np.array(pos[i])
            inv_deg_i = inv_degs[i]
            L_plus_ii = L_plus[i, i]
            
            for j in range(i + 1, n):
                # i and j are the node labels
                res[idx] = L_plus_ii + L_plus[j, j] - 2.0 * L_plus[i, j]
                deg_sums[idx] = inv_deg_i + inv_degs[j]
                
                pos_j = np.array(pos[j])
                dists[idx] = RGGBuilder.toroidal_distance(pos_i, pos_j)
                
                pairs[idx, 0] = i
                pairs[idx, 1] = j
                
                idx += 1
                
        print("All-pairs calculation complete.")
        return res, deg_sums, dists, pairs