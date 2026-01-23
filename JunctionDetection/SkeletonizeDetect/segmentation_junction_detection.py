import torch
import numpy as np
from skimage.morphology import skeletonize
from skan.csr import skeleton_to_csgraph, summarize, Skeleton
import networkx as nx


MIN_BRANCH_LENGTH = 100
MIN_BRIDGE_LENGTH = 20


def skeletonize_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    '''
    Skeletonize a single segmentation mask
    segmentation_mask: torch.Tensor of shape (C, H, W)
    Returns: np.ndarray of shape (C, H, W) representing the skeletonized mask
    '''
    mask_np = segmentation_mask.detach().cpu().squeeze().numpy().astype(np.uint8)
    binary_mask = (mask_np > 0).astype(np.uint8)
    return skeletonize(binary_mask).astype(np.uint8)


def get_graph_with_degrees(skeleton: np.ndarray):
    '''
    Convert skeleton to graph and compute node degrees
    skeleton: np.ndarray of shape (H, W)
    Returns: graph in CSR format and degrees of each node
    '''
    graph, coordinates = skeleton_to_csgraph(skeleton)
    degrees = np.diff(graph.indptr)
    coordinates = np.stack(np.asarray(coordinates), axis=1)
    return graph, coordinates, degrees


def filter_junctions_by_length(skeleton: np.ndarray, junction_indices: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    skel_obj = Skeleton(skeleton)
    skel_stats = summarize(skel_obj)

    # 1. Find all nodes in cycles (loops)
    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)
    nodes_in_cycles = set()
    try:
        # Use cycle_basis to find all loops in the graph
        cycles = nx.cycle_basis(nx_graph)
        for cycle in cycles:
            nodes_in_cycles.update(cycle)
    except:
        pass

    valid_coords = []
    nodes_handled = set()
    BRIDGE_MAX_SIZE = 15  # Consolidate nodes closer than this

    # 2. Cleanup stats
    clean_stats = skel_stats[skel_stats['node-id-src']
                             != skel_stats['node-id-dst']].copy()

    # Pass 1: Handle 4-way artifacts and Bridges
    for _, branch in clean_stats.iterrows():
        n1, n2 = int(branch['node-id-src']), int(branch['node-id-dst'])

        if (degrees[n1] > 2 and degrees[n2] > 2 and branch['branch-distance'] < BRIDGE_MAX_SIZE):
            if n1 in nodes_handled or n2 in nodes_handled:
                continue

            # If the junction is part of a loop, it's an artifact
            if n1 in nodes_in_cycles or n2 in nodes_in_cycles:
                nodes_handled.update([n1, n2])
                continue

            # Check for 4 long arms
            cluster_arms = clean_stats[((clean_stats['node-id-src'].isin([n1, n2])) |
                                        (clean_stats['node-id-dst'].isin([n1, n2]))) &
                                       (clean_stats.index != branch.name)]

            sig_arms = sum(1 for _, arm in cluster_arms.iterrows()
                           if arm['branch-distance'] >= MIN_BRANCH_LENGTH)

            if sig_arms >= 4:
                # Store the actual coordinate (y, x)
                midpoint = (
                    skel_obj.coordinates[n1] + skel_obj.coordinates[n2]) / 2
                valid_coords.append(midpoint)
                nodes_handled.update([n1, n2])
            else:
                nodes_handled.update([n1, n2])

    # Pass 2: Handle standard 3-way junctions
    for j_idx in junction_indices:
        if j_idx in nodes_handled or j_idx in nodes_in_cycles:
            continue

        node_branches = clean_stats[(clean_stats['node-id-src'] == j_idx) |
                                    (clean_stats['node-id-dst'] == j_idx)]

        # Strict check: An arm is ONLY significant if it is long
        sig_arms = 0
        for _, branch in node_branches.iterrows():
            if branch['branch-distance'] >= MIN_BRANCH_LENGTH:
                sig_arms += 1
            # If it leads to a distant junction, it's also valid
            else:
                u, v = int(branch['node-id-src']), int(branch['node-id-dst'])
                other = v if u == j_idx else u
                if degrees[other] > 2 and branch['branch-distance'] > BRIDGE_MAX_SIZE:
                    sig_arms += 1

        if sig_arms >= 3:
            valid_coords.append(skel_obj.coordinates[j_idx])

    if not valid_coords:
        return np.empty((0, 2))

    # Return (N, 2) array of coordinates
    return np.array(valid_coords)


def detect_junctions_in_segmentation_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    skeleton = skeletonize_mask(segmentation_mask)
    _, _, degrees = get_graph_with_degrees(skeleton)

    # Get all junction candidates
    junction_indices = np.where(degrees > 2)[0]

    # This now returns actual COORDINATES (y, x), not indices
    filtered_coords = filter_junctions_by_length(
        skeleton, junction_indices, degrees)

    if filtered_coords.size == 0:
        return np.empty((0, 2)), skeleton

    # Convert (y, x) to (x, y) for plotting
    return filtered_coords[:, ::-1], skeleton


def detect_junctions_in_batched_segmentation_masks(segmentation_masks: torch.Tensor) -> np.ndarray:
    '''
    Detect junctions in batched segmentation masks
    segmentation_masks: torch.Tensor of shape (B, C, H, W) where B is batch size, C is number of channels (usually 1 for binary masks), H is height, W is width 
        these shouldn't be logits, so sigmoid activation should have already been applied
    Returns: np.ndarray of shape (B, N_i, 2) where N_i is number of junctions in the i-th mask, each row is (x, y) pixel coordinates of a junction
    '''
    batch_junction_coords = []
    for i in range(segmentation_masks.shape[0]):
        mask = segmentation_masks[i]
        junction_coords = detect_junctions_in_segmentation_mask(mask)
        batch_junction_coords.append(junction_coords)
    return np.array(batch_junction_coords)
