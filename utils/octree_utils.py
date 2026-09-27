import torch
import ocnn.octree
import ocnn


def normalize_points_for_octree(points, scale=0.99):
    """Normalize one point cloud to the coordinate range expected by OCNN."""
    bbmin = points.min(dim=0).values
    bbmax = points.max(dim=0).values
    center = (bbmin + bbmax) * 0.5
    box_size = (bbmax - bbmin).max().clamp_min(1.0e-6)
    return (points - center) * (2.0 * scale / box_size)


def print_points_range(points, name="points"):
    """Print point coordinate range for octree debugging."""
    points = points.detach().cpu()
    print(f"{name} min:", points.min(dim=0).values.tolist())
    print(f"{name} max:", points.max(dim=0).values.tolist())


def print_octree_node_counts(octree):
    """Print total and non-empty node counts of each octree level."""
    print("depth | total nodes | non-empty nodes")
    print("--------------------------------------")
    for depth in range(octree.depth + 1):
        total = int(octree.nnum[depth].item())
        nempty = int(octree.nnum_nempty[depth].item())
        print(f"{depth:5d} | {total:11d} | {nempty:15d}")


def estimate_normals_torch(points, knn=30, batch_size=1024):
    """Estimate normals torch.

    Args:
        points: Point cloud input.
        knn: Input parameter.
        batch_size: Batch input.

    Returns:
        Computed result.
    """
    N = points.shape[0]
    effective_knn = min(int(knn), int(N))

    idx_list = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)

        dist = torch.cdist(points[i:end], points)
        _, idx = dist.topk(k=effective_knn, largest=False, dim=-1)
        idx_list.append(idx)

    idx = torch.cat(idx_list, dim=0)

    neighbors = points[idx]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)

    cov = torch.matmul(centered.transpose(-2, -1), centered) / max(effective_knn - 1, 1)

    _, eigenvectors = torch.linalg.eigh(cov)

    normals = eigenvectors[:, :, 0]

    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1

    return normals

def init_batch_points(batch_data):
    """Run init batch points.

    Args:
        batch_data: Batch input.

    Returns:
        Function output.
    """
    batch_points = []
    for i in range(batch_data["pts"].size(0)):
        points = batch_data["pts"][i].float()
        points = normalize_points_for_octree(points)
        if "normals" in batch_data:
            normals = batch_data["normals"][i].to(device=points.device, dtype=points.dtype)
        else:
            normals = estimate_normals_torch(points)
        features = batch_data["rgb_points"][i]

        labels = torch.full(
            [batch_data["pts"].size(1)],
            batch_data["category_label"][i][0].item(),
            device=points.device,
        )
        labels = labels.reshape(-1, 1)
        oct_points = ocnn.octree.Points(points, normals, features, labels)
        batch_points.append(oct_points)
    return batch_points

def build_batch_octree(batch_points, depth, full_depth):
    """Build batch octree.

    Args:
        batch_points: Batch input.
        depth: Input parameter.
        full_depth: Input parameter.

    Returns:
        Generated result.
    """
    batch_octrees = []
    for points in batch_points:
        octree = ocnn.octree.Octree(depth, full_depth=full_depth, batch_size=1, device='cuda')
        octree.build_octree(points.to('cuda'))
        batch_octrees.append(octree)
    octree_batched = ocnn.octree.merge_octrees(batch_octrees)
    octree_batched.construct_all_neigh()
    return octree_batched.to('cuda')
