import torch
import numpy as np
from skimage.morphology import skeletonize
from skan.csr import skeleton_to_csgraph, summarize, Skeleton
import networkx as nx


MIN_BRANCH_LENGTH = 75
MIN_JUNCTION_CONNECTOR_LENGTH = 10
MAX_JUNCTION_CONNECTOR_LENGTH = 20
MIN_TERMINAL_BRANCH_LENGTH = 30


def skeletonize_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    '''
    Skeletonize a single segmentation mask
    segmentation_mask: torch.Tensor of shape (C, H, W)
    Returns: np.ndarray of shape (C, H, W) representing the skeletonized mask
    '''
    mask_np = segmentation_mask.detach().cpu().squeeze().numpy().astype(np.uint8)
    binary_mask = (mask_np > 0).astype(np.uint8)
    return skeletonize(binary_mask).astype(np.uint8)


def prune_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """
    Removes short terminal branches (spurs), these are likely artifacts
    """
    skel_obj = Skeleton(skeleton)
    stats = summarize(skel_obj)

    _, _, degrees = get_graph_coordinates_degrees(skeleton)

    # Find paths that end in a degree-1 node and are too short
    bad_paths = []
    for i, row in stats.iterrows():
        u, v = int(row['node-id-src']), int(row['node-id-dst'])
        if (degrees[u] == 1 or degrees[v] == 1) and row['branch-distance'] < MIN_TERMINAL_BRANCH_LENGTH:
            bad_paths.append(i)

    # Zero out the pixels belonging to short terminal branches
    pruned_skeleton = skeleton.copy()
    for path_idx in bad_paths:
        row = stats.loc[path_idx]
        u, v = int(row['node-id-src']), int(row['node-id-dst'])
        coords = skel_obj.path_coordinates(path_idx)

        # Identify junction coordinates to preserve them - don't zero out pixel where spur attaches to skeleton
        junction_coords = []
        if degrees[u] > 2:
            junction_coords.append(skel_obj.coordinates[u].astype(int))
        if degrees[v] > 2:
            junction_coords.append(skel_obj.coordinates[v].astype(int))

        for r, c in coords.astype(int):
            is_junction = any(np.array_equal([r, c], j_coord)
                              for j_coord in junction_coords)
            if not is_junction:
                pruned_skeleton[r, c] = 0

    return pruned_skeleton


def get_graph_coordinates_degrees(skeleton: np.ndarray):
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
    skel_stats = skel_stats[skel_stats['node-id-src']
                            != skel_stats['node-id-dst']]

    # find all nodes in cycles
    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)
    nodes_in_cycles = {n for cycle in nx.cycle_basis(nx_graph) for n in cycle}

    valid_coords = []
    nodes_handled = set()

    # handle 4-way artifacts and junctions connected by a short branch (H shape)
    for _, branch in skel_stats.iterrows():
        n1, n2 = int(branch['node-id-src']), int(branch['node-id-dst'])

        # check if short branch connects two junctions
        if (degrees[n1] > 2 and degrees[n2] > 2 and branch['branch-distance'] < MAX_JUNCTION_CONNECTOR_LENGTH):
            if n1 in nodes_handled or n2 in nodes_handled:
                continue

            # If the junction is part of a loop, it's an artifact
            if n1 in nodes_in_cycles or n2 in nodes_in_cycles:
                nodes_handled.update([n1, n2])
                continue

            # Check significant arms for both junctions (junction connector ends)
            cluster_arms = skel_stats[((skel_stats['node-id-src'].isin([n1, n2])) |
                                       (skel_stats['node-id-dst'].isin([n1, n2]))) &
                                      (skel_stats.index != branch.name)]

            nof_significant_arms = sum(1 for _, arm in cluster_arms.iterrows()
                                       if arm['branch-distance'] >= MIN_BRANCH_LENGTH)

            # if the cluster connected by the short branch has at least 4 significant arms, keep as junction
            if nof_significant_arms >= 4:
                midpoint = (
                    skel_obj.coordinates[n1] + skel_obj.coordinates[n2]) / 2
                valid_coords.append(midpoint)

            nodes_handled.update([n1, n2])

    # handle standard 3-way and 4-way junctions
    for j_idx in junction_indices:
        if j_idx in nodes_handled or j_idx in nodes_in_cycles:
            continue

        node_branches = skel_stats[(skel_stats['node-id-src'] == j_idx) |
                                   (skel_stats['node-id-dst'] == j_idx)]

        # an arm is only significant if it is long enough or leads to a distant junction
        nof_significant_arms = 0
        for _, branch in node_branches.iterrows():
            if branch['branch-distance'] >= MIN_BRANCH_LENGTH:
                nof_significant_arms += 1
            else:
                u, v = int(branch['node-id-src']), int(branch['node-id-dst'])
                other = v if u == j_idx else u
                if degrees[other] > 2 and branch['branch-distance'] > MAX_JUNCTION_CONNECTOR_LENGTH:
                    nof_significant_arms += 1

        if nof_significant_arms >= 3:
            valid_coords.append(skel_obj.coordinates[j_idx])

    if not valid_coords:
        return np.empty((0, 2))
    return np.array(valid_coords)


def detect_junctions_in_segmentation_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    skeleton = skeletonize_mask(segmentation_mask)
    skeleton = prune_skeleton(skeleton)

    _, _, degrees = get_graph_coordinates_degrees(skeleton)

    # get indices of junction nodes (degree > 2)
    junction_indices = np.where(degrees > 2)[0]
    # filter out junctions based on branch lengths, cycles, and artifacts
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
        junction_coords, _ = detect_junctions_in_segmentation_mask(mask)
        batch_junction_coords.append(junction_coords)
    return np.array(batch_junction_coords)
