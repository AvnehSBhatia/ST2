"""
K-means clustering uses full-dimensional personality vectors (all dims).
2D coordinates here are for visualization only (PCA); clustering is never done in 2D.
"""

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from network import kmeans_auto_k


def get_2d_coordinates(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    For visualization only: reduce (n, dim) to (n, 2) via PCA.
    Clustering is always done in full dimension; this does not affect k-means.
    Returns: (n, 2) array of coordinates; row i corresponds to agent i.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[1] < 2:
        return np.hstack([vectors, np.zeros((vectors.shape[0], 1))])
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(vectors)


def shares_to_edges(shares: list[tuple[Any, list]]) -> list[tuple[Any, Any]]:
    """
    Convert shares from pipeline format to a list of directed edges.
    shares: list of (sharer_uid, [recipient_uid, ...])
    Returns: list of (start_uid, end_uid) for each share.
    """
    return [(sharer_uid, rec_uid) for sharer_uid, rec_list in shares for rec_uid in rec_list]


def space_clusters_apart(
    coords: np.ndarray,
    labels: np.ndarray,
    strength: float = 0.75,
    iterations: int = 24,
) -> np.ndarray:
    """
    Increase visual separation between clusters while preserving each cluster's
    internal point layout.

    The transformation only translates whole clusters; it does not warp the points
    inside a cluster. A light outward expansion is combined with centroid repulsion
    so nearby clusters stop overlapping in the 2D visualization.
    """
    coords = np.asarray(coords, dtype=np.float64).copy()
    labels = np.asarray(labels)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be shape (n, 2)")
    if coords.shape[0] == 0:
        return coords

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return coords

    global_center = coords.mean(axis=0)
    cluster_centers: dict[Any, np.ndarray] = {}
    cluster_radii: dict[Any, float] = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_points = coords[mask]
        center = cluster_points.mean(axis=0)
        cluster_centers[cluster_id] = center
        if len(cluster_points) <= 1:
            cluster_radii[cluster_id] = 0.5
        else:
            radii = np.linalg.norm(cluster_points - center, axis=1)
            cluster_radii[cluster_id] = max(float(np.percentile(radii, 85)), 0.5)

    # First, expand clusters radially away from the global center.
    expanded_centers: dict[Any, np.ndarray] = {}
    for idx, cluster_id in enumerate(unique_labels):
        center = cluster_centers[cluster_id]
        direction = center - global_center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            angle = 2.0 * np.pi * idx / max(len(unique_labels), 1)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            norm = 1.0
        expanded_centers[cluster_id] = global_center + direction * (1.0 + strength)

    # Then apply centroid repulsion so visually adjacent clusters separate further.
    centers_array = np.stack([expanded_centers[cluster_id] for cluster_id in unique_labels], axis=0)
    radii_array = np.array([cluster_radii[cluster_id] for cluster_id in unique_labels], dtype=np.float64)
    min_gap_scale = 1.2 + 0.9 * float(strength)

    for _ in range(max(iterations, 1)):
        shifts = np.zeros_like(centers_array)
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                delta = centers_array[j] - centers_array[i]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    angle = 2.0 * np.pi * (i + j + 1) / max(len(unique_labels), 1)
                    unit = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
                    dist = 1.0
                else:
                    unit = delta / dist

                target_gap = (radii_array[i] + radii_array[j]) * min_gap_scale
                if dist >= target_gap:
                    continue

                overlap = target_gap - dist
                push = unit * (overlap * 0.5)
                shifts[i] -= push
                shifts[j] += push

        centers_array += shifts

    out = coords.copy()
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        translation = centers_array[idx] - cluster_centers[cluster_id]
        out[mask] = coords[mask] + translation
    return out

def get_clustering_output(
    uids: list,
    vectors: np.ndarray,
    labels: np.ndarray | None = None,
    shares: list[tuple[Any, list]] | None = None,
    pca_random_state: int = 42,
    cluster_pull_strength: float | None = None,
    cluster_spacing_strength: float | None = None,
) -> tuple[np.ndarray, list[tuple[Any, Any]]]:
    """
    Compute 2D coordinates (for plotting) and share edges. K-means uses all dimensions.

    Clustering (if labels is None) is run on full vectors, not on 2D.
    2D is only for visualization.

    Args:
        uids: List of n agent UIDs (same order as vectors).
        vectors: (n, dim) full-dimensional personality vectors; k-means uses all dims.
        labels: (n,) cluster label per agent. If None, k-means is run on vectors (all dims).
        shares: List of (sharer_uid, [recipient_uid, ...]) from run_media_pipeline. Optional.
        pca_random_state: Seed for PCA (used only for 2D coords, not for clustering).
        cluster_pull_strength: Backward-compatible alias for cluster spacing.
        cluster_spacing_strength: How strongly to separate clusters in 2D.

    Returns:
        coords: (n, 2) array for visualization; coords[i] is the 2D position for uids[i].
        edges: List of (start_uid, end_uid) for each share.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(uids)
    if vectors.shape[0] != n:
        raise ValueError("len(uids) must equal vectors.shape[0]")

    if labels is None:
        labels, _, _ = kmeans_auto_k(vectors)  # full-dimensional clustering

    spacing_strength = (
        cluster_spacing_strength
        if cluster_spacing_strength is not None
        else (cluster_pull_strength if cluster_pull_strength is not None else 0.75)
    )
    coords = get_2d_coordinates(vectors, random_state=pca_random_state)  # 2D only for plotting
    coords = space_clusters_apart(coords, labels, strength=float(spacing_strength))
    edges = shares_to_edges(shares) if shares else []
    return coords, edges
