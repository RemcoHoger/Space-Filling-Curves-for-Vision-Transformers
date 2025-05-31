import matplotlib.pyplot as plt
from functools import cache, lru_cache
import numpy as np
import sys
import math
from typing import List, Tuple, Callable


def onion_curve(order, size=1.0):
    """
    Generate points for an Onion curve of a given order.

    Args:
        order (int): Order of the curve (depth of recursion).
        size (float): Length of one side of the entire curve's square.

    Returns:
        List[Tuple[float, float]]: Ordered list of (x, y) points.
    """
    # Ensure even order (onion curves are defined for even-sized grids)
    order *= 2

    def generate_coords(j, offset_x=0, offset_y=0):
        """Generate coordinates for a j×j grid with the given offset."""
        if j == 2:
            # Base case: 2×2 grid in clockwise order
            return [(offset_x, offset_y),          # (0,0)
                    (offset_x + 1, offset_y),      # (1,0)
                    (offset_x + 1, offset_y + 1),  # (1,1)
                    (offset_x, offset_y + 1)]      # (0,1)

        coords = []

        # Bottom row (left to right)
        for x in range(j):
            coords.append((offset_x + x, offset_y))

        # Right column (bottom to top, excluding bottom-right)
        for y in range(1, j):
            coords.append((offset_x + j - 1, offset_y + y))

        # Top row (right to left, excluding top-right)
        for x in range(j - 2, -1, -1):
            coords.append((offset_x + x, offset_y + j - 1))

        # Left column (top to bottom, excluding top-left and bottom-left)
        for y in range(j - 2, 0, -1):
            coords.append((offset_x, offset_y + y))

        # Recursively add the inner grid
        if j > 2:
            inner_coords = generate_coords(j - 2, offset_x + 1, offset_y + 1)
            coords.extend(inner_coords)

        return coords

    # Generate and scale the coordinates
    integer_coords = generate_coords(order)
    cell_size = size / order
    scaled_points = [(x * cell_size + cell_size / 2,
                     y * cell_size + cell_size / 2)
                     for x, y in integer_coords]

    # Apply transformation for consistency with other curves
    # Using identity transformation (no rotation)
    deg = 0
    rotation_matrix = np.array([[np.cos(deg), -np.sin(deg)],
                               [np.sin(deg), np.cos(deg)]])

    return [tuple(np.dot(rotation_matrix, [x_i, y_i]))
            for x_i, y_i in scaled_points]


def peano_curve(order, size=1.0):
    """
    Generate points for a Peano curve of a given order.

    Args:
        order (int): Depth of recursion.
        size (float): Length of one side of the entire curve's square.

    Returns:
        List[Tuple[float, float]]: Ordered list of curve points.
    """
    @cache
    def generate(x, y, size, order, pattern=0):
        if order == 0:
            # Center point of current cell
            return [(x + size / 2, y + size / 2)]

        points = []
        size /= 3

        # Orientation patterns for S-traversal and flips
        patterns = [
            [((0, 0), 0), ((1, 0), 1), ((2, 0), 0),
             ((2, 1), 1), ((1, 1), 0), ((0, 1), 1),
             ((0, 2), 0), ((1, 2), 1), ((2, 2), 0)],
            [((2, 0), 1), ((1, 0), 0), ((0, 0), 1),
             ((0, 1), 0), ((1, 1), 1), ((2, 1), 0),
             ((2, 2), 1), ((1, 2), 0), ((0, 2), 1)],
            [((0, 2), 2), ((1, 2), 3), ((2, 2), 2),
             ((2, 1), 3), ((1, 1), 2), ((0, 1), 3),
             ((0, 0), 2), ((1, 0), 3), ((2, 0), 2)],
            [((2, 2), 3), ((1, 2), 2), ((0, 2), 3),
             ((0, 1), 2), ((1, 1), 3), ((2, 1), 2),
             ((2, 0), 3), ((1, 0), 2), ((0, 0), 3)]
        ]

        sequence = patterns[pattern]

        for idx, ((dx, dy), next_pattern) in enumerate(sequence):
            nx = x + dx * size
            ny = y + dy * size
            sub = generate(nx, ny, size, order - 1, next_pattern)

            # Flip middle column for continuity
            if idx % 3 == 1:
                sub = sub[::-1]

            points.extend(sub)

        return points
    points = generate(0, 0, size, order)
    deg = np.pi / 2
    rotation_matrix = np.array([[np.cos(deg), -np.sin(deg)],
                                [np.sin(deg), np.cos(deg)]])
    mirror = np.array([[-1, 0],
                       [0, 1]])
    fin = mirror @ rotation_matrix
    return [tuple(np.dot(fin, [x_i, y_i])) for x_i, y_i in points]


def z_curve(order, size=1.0):
    """
    Generate points for a Z-order (Morton) curve of a given order.

    Args:
        order (int): Recursion depth.
        size (float): Length of the side of the square.

    Returns:
        List[Tuple[float, float]]: List of (x, y) points.
    """
    points = []

    @cache
    def z(x0, y0, w, n):
        if n == 0:
            points.append((x0 + w / 2, y0 + w / 2))
        else:
            half = w / 2
            z(x0 + half, y0, half, n - 1)  # Top-right
            z(x0, y0, half, n - 1)  # Top-left
            z(x0 + half, y0 + half, half, n - 1)  # Bottom-right
            z(x0, y0 + half, half, n - 1)  # Bottom-left

    z(0, 0, size, order)
    deg = np.pi
    rotation_matrix = np.array([[np.cos(deg), -np.sin(deg)],
                                [np.sin(deg), np.cos(deg)]])
    mirror = np.array([[-1, 0],
                       [0, -1]])
    fin = mirror @ rotation_matrix
    return [tuple(np.dot(fin, [x_i, y_i])) for x_i, y_i in points]


def hilbert_curve(order, size=1.0):
    """
    Generate points for a Hilbert curve of a given order.

    Args:
        order (int): Recursion depth of the Hilbert curve.
        size (float): Length of one side of the entire curve's square.

    Returns:
        List[Tuple[float, float]]: The list of (x, y) points.
    """
    points = []

    @cache
    def hilbert(x0, y0, xi, xj, yi, yj, n):
        if n <= 0:
            x = x0 + (xi + yi) / 2
            y = y0 + (xj + yj) / 2
            points.append((x, y))
        else:
            hilbert(x0, y0, yi/2, yj / 2, xi/2, xj/2, n-1)
            hilbert(x0 + xi/2, y0 + xj/2, xi/2, xj / 2, yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2,
                    yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi, y0 + xj/2 + yj, - yi/2, -yj/2,
                    -xi/2, -xj/2, n-1)

    hilbert(0, 0, size, 0, 0, size, order)
    deg = np.pi / 2
    rotation_matrix = np.array([[np.cos(deg), -np.sin(deg)],
                                [np.sin(deg), np.cos(deg)]])
    mirror = np.array([[-1, 0],
                       [0, 1]])
    fin = mirror @ rotation_matrix
    return [tuple(np.dot(fin, [x_i, y_i])) for x_i, y_i in points]


def moore_curve(order, size=1.0):
    """
    Generate points for a Moore curve of a given order.

    Args:
        order (int): Recursion depth.
        size (float): Length of the side of the square.

    Returns:
        List[Tuple[float, float]]: List of (x, y)
                                   points forming the Moore curve.
    """
    points = []

    @cache
    def hilbert(x0, y0, xi, xj, yi, yj, n):
        if n <= 0:
            x = x0 + (xi + yi) / 2
            y = y0 + (xj + yj) / 2
            points.append((x, y))
        else:
            hilbert(x0, y0, yi/2, yj / 2, xi/2, xj/2, n-1)
            hilbert(x0 + xi/2, y0 + xj/2, xi/2, xj / 2, yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2,
                    xj/2, yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi, y0 + xj/2 + yj, -yi/2, -yj/2,
                    -xi/2, -xj/2, n-1)

    def moore(x0, y0, xi, xj, yi, yj, n):
        if n <= 0:
            x = x0 + (xi + yi) / 2
            y = y0 + (xj + yj) / 2
            points.append((x, y))
        else:
            hilbert(x0 + xi/2, y0 + xj/2, -xi/2, xj/2, yi/2, yj/2, n - 1)
            hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj /
                    2, -xi/2, xj/2, yi/2, yj/2, n - 1)
            hilbert(x0 + xi/2 + yi, y0 + xj/2 + yj,
                    xi/2, xj/2, yi/2, -yj/2, n - 1)
            hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj /
                    2, xi/2, xj/2, yi/2, -yj/2, n - 1)

    moore(0, 0, size, 0, 0, size, order)
    deg = np.pi * 2
    rotation_matrix = np.array([[np.cos(deg), -np.sin(deg)],
                                [np.sin(deg), np.cos(deg)]])
    return [tuple(np.dot(rotation_matrix, [x_i, y_i])) for x_i, y_i in points]


def raster_curve(order, size=1.0):
    """
    Generate points for a raster (row-major) curve of a given order.
    Args:
        order (int): Order of the curve (depth of recursion).
        size (float): Length of one side of the entire curve's square.
    Returns:
        List[Tuple[float, float]]: Ordered list of (x, y) points.
    """
    points = []
    cell_size = size / (2 ** order)
    for y in range(2 ** order):
        for x in range(2 ** order):
            # Calculate the center of each cell
            x_center = (x + 0.5) * cell_size
            y_center = (y + 0.5) * cell_size
            points.append((x_center, y_center))
    return points

def find_hamiltonian_path(width, height, adjacency_order=None, diag=False):
    """
    Attempt to find a Hamiltonian path on a 2D grid with 8-way connectivity.

    This function uses a depth-first search (DFS) strategy to construct a
    Hamiltonian path that visits every cell in a width x height grid exactly
    once. It supports optional priority ordering of traversal based on a
    provided adjacency_order.

    Diagonal neighbors are permitted but deprioritized unless needed.
    The algorithm also includes pruning based on flood-fill reachability to
    improve performance.

    Args:
        width (int): Width of the grid.
        height (int): Height of the grid.
        adjacency_order (dict[Tuple[int, int], int], optional): Mapping of grid
            coordinates to priority values for neighbor sorting.
            Lower values are higher priority.

    Returns:
        List[Tuple[int, int]] | None: A list of coordinates representing
                                      a Hamiltonian path,
                                      or None if no path was found.
    """
    sys.setrecursionlimit(10_000_000)
    total = width * height

    # Initialize visited grid and path
    visited = [[False] * height for _ in range(width)]
    path = []

    # Precompute static neighbors for each cell. A static neighbor is one
    # that is always a neighbor, regardless of the current path.
    # This is used to speed up the flood-fill check.
    dirs_cardinal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        dirs_diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        dirs_diag = []
    static_nbrs = {}
    for x in range(width):
        for y in range(height):
            nbrs = []
            for dx, dy in (dirs_cardinal + dirs_diag):
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height:
                    nbrs.append((nx, ny))
            static_nbrs[(x, y)] = nbrs

    def get_neighbors(x, y):
        """
        Get neighbors of (x, y) in a specific order:
        1) cardinal neighbors first, then diagonal
        2) sort by adjacency_order if provided
        3) sort by diagonal vs. cardinal
        4) sort by adjacency_order if provided

        This function returns a sorted list of neighbors for the given
        coordinates (x, y). The sorting is based on the adjacency_order
        """
        nbrs = list(static_nbrs[(x, y)])

        def key_fn(v):
            """
            Custom key function for sorting neighbors.
            """
            dx = abs(v[0] - x)
            dy = abs(v[1] - y)
            is_diag = 1 if dx == 1 and dy == 1 else 0
            score = adjacency_order.get(v, total) if adjacency_order else 0
            return (is_diag, score)
        nbrs.sort(key=key_fn)
        return nbrs

    def flood_check(sx, sy, remaining):
        """
        Heuristic to check if there are enough reachable cells from (sx, sy)
        to complete the Hamiltonian path. This is a check which
        counts the number of reachable cells from (sx, sy) that are not
        already visited. If the count is less than the remaining cells
        needed to complete the path, we can prune this branch of the search.
        Args:
            sx (int): Starting x-coordinate.
            sy (int): Starting y-coordinate.
            remaining (int): Number of remaining cells to visit.
        Returns:
            bool: True if there are enough reachable cells, False otherwise.
        """
        stack = [(sx, sy)]
        seen = {(sx, sy)}
        cnt = 0
        while stack:
            x, y = stack.pop()
            cnt += 1
            if cnt >= remaining:
                return True
            for nx, ny in static_nbrs[(x, y)]:
                if not visited[nx][ny] and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        return cnt >= remaining

    def dfs(x, y):
        """
        Depth-first search to find Hamiltonian path.

        Args:
            x (int): Current x-coordinate.
            y (int): Current y-coordinate.

        Returns:
            bool: True if a Hamiltonian path is found, False otherwise.
        """
        if len(path) == total:
            return True

        # Get unvisited neighbors
        nbrs = [(nx, ny) for nx, ny in get_neighbors(x, y)
                if not visited[nx][ny]]

        # Bridge pruning: if there are no unvisited neighbors,
        # we can prune this branch
        forced, filtered = [], []
        for nx, ny in nbrs:
            exits = 0
            for ux, uy in static_nbrs[(nx, ny)]:
                if not visited[ux][uy] and (ux, uy) != (x, y):
                    exits += 1
            if exits == 0 and len(path)+1 < total:
                continue
            if exits == 1:
                forced.append((nx, ny))
            filtered.append((nx, ny))
        nbrs = forced or filtered

        # If no neighbors, we can't proceed
        for nx, ny in nbrs:
            visited[nx][ny] = True
            path.append((nx, ny))

            rem = total - len(path)
            if rem == 0 or flood_check(nx, ny, rem):
                if dfs(nx, ny):
                    return True

            visited[nx][ny] = False
            path.pop()

        return False

    # Start DFS from corners or specified starting points
    # If an adjacency_order is provided, start from the first point in it
    # Otherwise, start from the four corners of the grid
    if adjacency_order:
        start_pts = [min(adjacency_order, key=adjacency_order.get)]
    else:
        start_pts = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]

    for sx, sy in start_pts:
        visited[sx][sy] = True
        path[:] = [(sx, sy)]
        if dfs(sx, sy):
            return path
        visited[sx][sy] = False

    # If no Hamiltonian path is found, first try with diagonal, then None
    # if not diag:
    #     return find_hamiltonian_path(width, height, adjacency_order, diag=True)

    return None


def refine_curve_to_hamiltonian(curve, width, height):
    """
    Given an initial curve (list of (x,y)), find a true Hamiltonian
    path using that order as a guide.
    """
    # Build adjacency priority from initial curve
    priority = {pt: idx for idx, pt in enumerate(curve)}
    # Find exact Hamiltonian path
    ham = find_hamiltonian_path(width, height, adjacency_order=priority)
    return ham


def grid_size(order, sfc):
    name = sfc.__name__
    if name in ("hilbert_curve", "z_curve", "moore_curve"):
        return 2**order
    elif name == "peano_curve":
        return 3**order
    elif name == "onion_curve":
        # onion_curve(order) builds an order×order grid (enforces even)
        return order + (order % 2)
    else:
        raise ValueError(f"Unknown SFC: {name}")


def embed_and_prune_sfc(sfc, width, height):
    """
    Embed a space-filling curve (SFC) into a rectangular domain and prune
    out-of-domain points.
    """
    order = 0
    while grid_size(order, sfc) < max(width, height):
        order += 1

    # Generate on the padded square, then prune out‑of‑domain points
    P = grid_size(order, sfc)
    raw = sfc(order, size=P)   # returns [(x, y), ...] in [0..P] coords

    # Convert float centers back to integer cell indices and filter
    curve = []
    for x, y in raw:
        i, j = int(np.floor(x)), int(np.floor(y))
        if 0 <= i < width and 0 <= j < height:
            curve.append((i, j))

    return curve


def get_symmetries(B: int) -> List[Callable[[float, float], Tuple[float, float]]]:
    """
    Return the 8 dihedral symmetries (rotations/reflections) for a B×B block
    as functions mapping (x, y) -> (x', y').
    """
    def id_(x, y): return (x, y)
    def rot90(x, y): return (y, B - x)
    def rot180(x, y): return (B - x, B - y)
    def rot270(x, y): return (B - y, x)
    def refl_x(x, y): return (B - x, y)
    def refl_x_rot90(x, y): return (y, x)               # reflect over y=x
    # reflect over horizontal mid
    def refl_x_rot180(x, y): return (x, B - y)
    # reflect over anti-diagonal
    def refl_x_rot270(x, y): return (B - y, B - x)
    return [id_, rot90, rot180, rot270,
            refl_x, refl_x_rot90, refl_x_rot180, refl_x_rot270]


def block_stitch_sfc(
    sfc: Callable[[int, int], List[Tuple[float, float]]],
    width: int,
    height: int
) -> List[Tuple[int, int]]:
    """
    Block-stitching with two-end alignment:
    - Precompute the sequence of power-of-two blocks (x0,y0,B,k).
    - For each block i, pick the symmetry that minimizes:
         dist(prev_exit, entry_i_sym) + dist(exit_i_sym, default_entry_{i+1})
    """
    # First, collect the block list in order
    blocks = []

    def collect(x0, y0, w, h):
        if w <= 0 or h <= 0:
            return
        base = 3 if sfc.__name__ == "peano_curve" else 2
        k = np.floor(np.log(min(w, h)) / np.log(base))
        B = base ** k
        blocks.append((x0, y0, B, k))
        # right stripe and bottom stripe
        collect(x0 + B, y0, w - B, B)
        collect(x0, y0 + B, w, h - B)
    collect(0, 0, width, height)

    # Precompute default entries of next blocks (unrotated)
    default_entries = []
    for (_x0, _y0, _B, _k) in blocks:
        raw = sfc(_k, _B)
        # default entry is floor of first raw point
        x, y = raw[0]
        default_entries.append((math.floor(_x0 + x), math.floor(_y0 + y)))

    visited = set()
    curve: List[Tuple[int, int]] = []
    blocked_curve: List[List[Tuple[int, int]]] = []

    def manh(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    prev_exit = None
    # Iterate through blocks
    for idx, (x0, y0, B, k) in enumerate(blocks):
        raw = sfc(k, B)
        syms = get_symmetries(B)
        best_score = math.inf
        best_oriented = None
        # target for next block entry
        next_entry = default_entries[idx+1] if idx+1 < len(blocks) else None

        # Try each symmetry
        for sym in syms:
            # apply to all raw points, then floor+translate
            pts = [(x0 + math.floor(sym(x, y)[0]), y0 +
                    math.floor(sym(x, y)[1])) for x, y in raw]
            # filter only new points
            new_pts = [p for p in pts if p not in visited]
            if not new_pts:
                continue
            entry = new_pts[0]
            exit_ = new_pts[-1]
            # score = dist from prev_exit to entry + dist exit to next_entry
            score = 0
            if prev_exit is not None:
                score += manh(prev_exit, entry)
            if next_entry is not None:
                score += manh(exit_, next_entry)
            if score < best_score:
                best_score = score
                best_oriented = new_pts

        # append best_oriented
        for p in best_oriented:
            visited.add(p)
            curve.append(p)
        blocked_curve.append(best_oriented)
        prev_exit = best_oriented[-1]

    return curve, blocked_curve


if __name__ == "__main__":
    # sfc = onion_curve
    # sfc = peano_curve
    # sfc = z_curve
    sfc = hilbert_curve
    # sfc = moore_curve

    width, height = 12, 12

    curve, blocked_curve = block_stitch_sfc(sfc, width, height)
    # curve = embed_and_prune_sfc(sfc, width, height)

    print(curve)
    x_vals, y_vals = zip(*curve)
    plt.plot(x_vals, y_vals, marker='o')
    plt.axis('equal')
    plt.title(f"{sfc.__name__} on {width}×{height} via embed‑&‑prune")
    # plt.show()

    # Plot the blocked curve
    fig, ax = plt.subplots()
    for block in blocked_curve:
        x_vals, y_vals = zip(*block)
        ax.plot(x_vals, y_vals, marker='o')
    ax.set_aspect('equal')
    ax.set_title(f"{sfc.__name__} on {width}×{height} via block-stitching")
    plt.show()
