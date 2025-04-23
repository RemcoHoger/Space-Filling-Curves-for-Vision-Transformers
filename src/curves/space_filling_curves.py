import matplotlib.pyplot as plt
from functools import cache
import numpy as np
import sys


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
    if order % 2 != 0:
        order += 1

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

    # Generate integer coordinates
    integer_coords = generate_coords(order)

    # Scale coordinates to the requested size
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
            z(x0 + half, y0,       half, n - 1)  # Top-right
            z(x0,       y0,       half, n - 1)  # Top-left
            z(x0 + half, y0 + half, half, n - 1)  # Bottom-right
            z(x0,       y0 + half, half, n - 1)  # Bottom-left

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
            hilbert(x0, y0,               yi/2, yj /
                    2,               xi/2, xj/2, n-1)
            hilbert(x0 + xi/2, y0 + xj/2, xi/2, xj /
                    2,               yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi/2, y0 + xj/2 +
                    yj/2, xi/2, xj/2, yi/2, yj/2, n-1)
            hilbert(x0 + xi/2 + yi, y0 + xj/2 + yj, -
                    yi/2, -yj/2, -xi/2, -xj/2, n-1)

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


def find_hamiltonian_path(width, height, adjacency_order=None):
    """
    Find a Hamiltonian path on the width×height grid graph.
    Optionally uses adjacency_order, a dict mapping node -> priority index.
    Returns list of (x,y) or None if not found.
    """
    sys.setrecursionlimit(10_000_000)
    total = width * height

    # visited flags and path
    visited = [[False] * height for _ in range(width)]
    path = []

    # 1) Precompute static neighbors for every cell once
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    static_nbrs = {
        (x, y): [(x + dx, y + dy)
                 for dx, dy in dirs
                 if 0 <= x + dx < width and 0 <= y + dy < height]
        for x in range(width) for y in range(height)
    }

    def get_neighbors(x, y):
        # Return a fresh list each time so sorting doesn't mutate static_nbrs
        nbrs = list(static_nbrs[(x, y)])
        if adjacency_order and (x, y) in adjacency_order:
            nbrs.sort(key=lambda v: adjacency_order.get(v, total))
        return nbrs

    def flood_check(sx, sy, remaining):
        """Quick flood-fill on unvisited cells from (sx,sy)."""
        stack = [(sx, sy)]
        seen = {(sx, sy)}
        count = 0
        while stack:
            x, y = stack.pop()
            count += 1
            if count >= remaining:
                return True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height
                        and not visited[nx][ny]
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        return count >= remaining

    def dfs(x, y):
        if len(path) == total:
            return True

        # 2) Get all unvisited neighbors of current
        nbrs = [(nx, ny) for nx, ny in get_neighbors(x, y)
                if not visited[nx][ny]]

        # --- BRIDGE / forced-move logic ----------------------------------
        forced = []
        filtered = []
        for nx, ny in nbrs:
            # count how many free exits (excluding back to (x,y))
            exits = 0
            for ux, uy in static_nbrs[(nx, ny)]:
                if not visited[ux][uy] and (ux, uy) != (x, y):
                    exits += 1
            # if no exits and we're not at goal, this move is a dead-end → skip
            if exits == 0 and len(path) + 1 < total:
                continue
            # a single exit means this edge is 'forced'
            if exits == 1:
                forced.append((nx, ny))
            filtered.append((nx, ny))

        # if any forced moves exist, *only* explore them
        if forced:
            nbrs = forced
        else:
            nbrs = filtered
        # ------------------------------------------------------------------

        for nx, ny in nbrs:
            visited[nx][ny] = True
            path.append((nx, ny))

            remaining = total - len(path)
            if remaining == 0 or flood_check(nx, ny, remaining):
                if dfs(nx, ny):
                    return True

            # backtrack
            visited[nx][ny] = False
            path.pop()

        return False

    # Try each corner as a start
    for sx, sy in [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]:
        visited[sx][sy] = True
        path[:] = [(sx, sy)]
        if dfs(sx, sy):
            return path
        visited[sx][sy] = False

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


def grid_size(order):
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
    while grid_size(order) < max(width, height):
        order += 1

    # —————— generate on the padded square, then prune out‑of‑domain points
    P = grid_size(order)
    raw = sfc(order, size=P)   # returns [(x, y), ...] in [0..P] coords

    # Convert float centers back to integer cell indices and filter
    curve = []
    for x, y in raw:
        i, j = int(np.floor(x)), int(np.floor(y))
        if 0 <= i < width and 0 <= j < height:
            curve.append((i, j))

    return curve


if __name__ == "__main__":
    # sfc = onion_curve
    # sfc = peano_curve
    # sfc = z_curve
    sfc = hilbert_curve
    # sfc = moore_curve

    width, height = 14, 14

    curve = embed_and_prune_sfc(sfc, width, height)
    curve = refine_curve_to_hamiltonian(curve, width, height)

    print(curve)
    x_vals, y_vals = zip(*curve)
    plt.plot(x_vals, y_vals, marker='o')
    plt.axis('equal')
    plt.title(f"{sfc.__name__} on {width}×{height} via embed‑&‑prune")
    plt.show()
