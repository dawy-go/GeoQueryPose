import torch


def canonical_bbox_axis_keypoints(device=None, dtype=torch.float32):
    corners = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ],
        device=device,
        dtype=dtype,
    )
    axes = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        device=device,
        dtype=dtype,
    )
    return torch.cat([corners, axes], dim=0)


def metric_keypoints_from_size(size, canonical_keypoints=None):
    if canonical_keypoints is None:
        canonical_keypoints = canonical_bbox_axis_keypoints(size.device, size.dtype)
    canonical_keypoints = canonical_keypoints.to(device=size.device, dtype=size.dtype)
    return canonical_keypoints.unsqueeze(0) * size.unsqueeze(1)


def transform_keypoints(metric_keypoints, rotation, translation):
    return torch.bmm(metric_keypoints, rotation.transpose(1, 2)) + translation.unsqueeze(1)


def inverse_transform_points(camera_points, rotation, translation, size=None, eps=1.0e-6):
    metric_points = torch.bmm(camera_points - translation.unsqueeze(1), rotation)
    if size is None:
        return metric_points
    return metric_points / size.unsqueeze(1).clamp_min(eps)


def solve_rotation_from_keypoints(metric_keypoints, camera_keypoints, eps=1.0e-6):
    source_center = metric_keypoints.mean(dim=1, keepdim=True)
    target_center = camera_keypoints.mean(dim=1, keepdim=True)
    source = metric_keypoints - source_center
    target = camera_keypoints - target_center

    covariance = torch.bmm(source.transpose(1, 2), target)
    u, _, vh = torch.linalg.svd(covariance.float(), full_matrices=False)
    u = u.to(dtype=metric_keypoints.dtype)
    vh = vh.to(dtype=metric_keypoints.dtype)

    row_transform = torch.bmm(u, vh)
    det = torch.det(row_transform.float()).to(dtype=metric_keypoints.dtype)
    flip = torch.ones((metric_keypoints.shape[0], 3), device=metric_keypoints.device, dtype=metric_keypoints.dtype)
    flip[:, 2] = torch.where(det < 0.0, -flip[:, 2], flip[:, 2])
    row_transform = torch.bmm(torch.bmm(u, torch.diag_embed(flip)), vh)

    rotation = row_transform.transpose(1, 2)
    translation = target_center.squeeze(1) - torch.bmm(
        source_center, row_transform).squeeze(1)
    fitted = torch.bmm(metric_keypoints, rotation.transpose(1, 2)) + translation.unsqueeze(1)
    residual = torch.linalg.norm(fitted - camera_keypoints, dim=-1).mean(dim=1).clamp_min(eps)
    return rotation, translation, residual


def solve_weighted_rotation_from_points(
    metric_points,
    camera_points,
    weights=None,
    eps=1.0e-6,
    svd_regularization=0.0,
):
    if weights is None:
        weights = torch.ones(metric_points.shape[:2], device=metric_points.device, dtype=metric_points.dtype)
    weights = weights.to(device=metric_points.device, dtype=metric_points.dtype).clamp_min(0.0)
    finite = (
        torch.isfinite(metric_points).all(dim=-1)
        & torch.isfinite(camera_points).all(dim=-1)
        & torch.isfinite(weights)
    )
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    normalized = weights / weight_sum

    source_center = (metric_points * normalized.unsqueeze(-1)).sum(dim=1, keepdim=True)
    target_center = (camera_points * normalized.unsqueeze(-1)).sum(dim=1, keepdim=True)
    source = metric_points - source_center
    target = camera_points - target_center

    covariance = torch.bmm((source * normalized.unsqueeze(-1)).transpose(1, 2), target)
    covariance_fp32 = covariance.float()
    if svd_regularization > 0.0:
        diagonal = covariance_fp32.new_tensor([1.0, 2.0, 3.0])
        covariance_fp32 = covariance_fp32 + torch.diag_embed(
            diagonal.expand(covariance_fp32.shape[0], -1)
            * float(svd_regularization)
        )
    u, _, vh = torch.linalg.svd(covariance_fp32, full_matrices=False)
    u = u.to(dtype=metric_points.dtype)
    vh = vh.to(dtype=metric_points.dtype)

    row_transform = torch.bmm(u, vh)
    det = torch.det(row_transform.float()).to(dtype=metric_points.dtype)
    flip = torch.ones((metric_points.shape[0], 3), device=metric_points.device, dtype=metric_points.dtype)
    flip[:, 2] = torch.where(det < 0.0, -flip[:, 2], flip[:, 2])
    row_transform = torch.bmm(torch.bmm(u, torch.diag_embed(flip)), vh)

    rotation = row_transform.transpose(1, 2)
    translation = target_center.squeeze(1) - torch.bmm(source_center, row_transform).squeeze(1)
    fitted = torch.bmm(metric_points, rotation.transpose(1, 2)) + translation.unsqueeze(1)
    point_residual = torch.linalg.norm(fitted - camera_points, dim=-1)
    residual = (point_residual * normalized).sum(dim=1).clamp_min(eps)
    valid_ratio = finite.float().mean(dim=1)
    return rotation, translation, residual, valid_ratio
