import torch
import torch.nn.functional as F


def rotation_matrix_to_6d(matrix):
    """
    Convert rotation matrix to 6D representation by taking the first two columns.
    Input: [Batch, 3, 3], Output: [Batch, 6]
    """
    # Extract first two columns and reshape to [Batch, 6]
    return matrix[..., :3, :2].transpose(-1, -2).reshape(-1, 6)


def rotation_matrix_to_9d(matrix):
    """
    Flatten a batch of rotation matrices to the 9D representation used by
    MonoDiff9D-style pose diffusion.
    """
    return matrix.reshape(-1, 9)


def axis_angle_to_rotation_matrix(axis_angle):
    """Convert exponential-map rotation vectors to SO(3) matrices.

    ``axis_angle`` may have any leading shape and must end in three values.  The
    sinc form keeps both the forward value and gradients finite at the exact
    zero residual used by the V6 warm-start parity contract.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            "axis_angle_to_rotation_matrix expects a final dimension of 3, "
            f"got {tuple(axis_angle.shape)}"
        )

    vector = axis_angle
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)

    theta = torch.linalg.vector_norm(vector, dim=-1)
    sin_over_theta = torch.sinc(theta / torch.pi)
    one_minus_cos_over_theta_sq = 0.5 * torch.sinc(
        theta / (2.0 * torch.pi)
    ).square()
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    identity = identity.expand(*vector.shape[:-1], 3, 3)
    return (
        identity
        + sin_over_theta[..., None, None] * skew
        + one_minus_cos_over_theta_sq[..., None, None]
        * torch.matmul(skew, skew)
    )


def six_d_to_rotation_matrix(d6):
    """
    Recover rotation matrix from 6D representation using Gram-Schmidt process.
    Input: [Batch, 6], Output: [Batch, 3, 3]
    """
    d6 = d6.view(-1, 6)
    a1, a2 = d6[:, :3], d6[:, 3:]

    # Normalize the first vector
    b1 = F.normalize(a1, dim=1, eps=1.0e-6)

    # Orthogonalize and normalize the second vector
    dot = torch.sum(b1 * a2, dim=1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=1, eps=1.0e-6)

    # Compute the third vector via cross product
    b3 = torch.cross(b1, b2, dim=1)

    # Stack to form [Batch, 3, 3] rotation matrix
    return torch.stack((b1, b2, b3), dim=-1)


def nine_d_to_rotation_matrix(d9):
    """
    Project arbitrary 9D matrix samples onto SO(3) with an SVD projection.
    Input: [Batch, 9], Output: [Batch, 3, 3]
    """
    dtype = d9.dtype
    try:
        autocast_context = torch.amp.autocast(
            device_type="cuda" if d9.is_cuda else "cpu",
            enabled=False,
        )
    except (AttributeError, TypeError):
        from torch.cuda.amp import autocast as cuda_autocast
        autocast_context = cuda_autocast(enabled=False)

    with autocast_context:
        mat = d9.view(-1, 3, 3).float()
        u, _, vh = torch.linalg.svd(mat, full_matrices=False)
        rot = torch.matmul(u, vh).float()
        det = torch.det(rot)
        correction = torch.ones((rot.shape[0], 3), device=rot.device, dtype=rot.dtype)
        correction[:, -1] = torch.where(
            det < 0.0,
            correction.new_tensor(-1.0),
            correction.new_tensor(1.0),
        )
        rot = torch.matmul(torch.matmul(u, torch.diag_embed(correction)), vh)
    return rot.to(dtype=dtype)


def project_6d_to_rotation_6d(d6):
    """
    Project an arbitrary 6D rotation sample back onto the valid SO(3) 6D manifold.
    """
    return rotation_matrix_to_6d(six_d_to_rotation_matrix(d6))


def project_9d_to_rotation_9d(d9):
    """
    Project arbitrary 9D rotation samples back to flattened SO(3).
    """
    return rotation_matrix_to_9d(nine_d_to_rotation_matrix(d9))


# --- Quick Verification ---
if __name__ == "__main__":
    # Create a random valid rotation matrix via QR decomposition
    q, _ = torch.linalg.qr(torch.randn(2, 3, 3))

    # Matrix -> 6D -> Matrix
    d6 = rotation_matrix_to_6d(q)
    recovered_q = six_d_to_rotation_matrix(d6)

    # Check error (should be near zero)
    error = torch.mean(torch.abs(q - recovered_q))
    print(f"6D vector sample: {d6[0]}")
    print(f"Reconstruction Error: {error.item():.2e}")
