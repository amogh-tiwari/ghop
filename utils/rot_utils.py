""" Many functions borrowed from grabnet repo: https://github.com/otaheri/GrabNet/blob/master/grabnet/tools/utils.py """
import torch
import torch.nn.functional as F
import numpy as np

def CRot2rotmat(pose):

    reshaped_input = pose.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def rotmat2aa(rotmat):
    '''
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = rotmat.size(0)
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0,1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose


def euler_to_rotmat(euler_angles):
    """
    Convert Euler angles (in radians) to rotation matrices
    
    Args:
        euler_angles: tensor of shape (..., 3) representing rotations around x, y, z axes
        
    Returns:
        rotation matrices of shape (..., 3, 3)
    """
    # Extract individual angles
    rx, ry, rz = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    
    # Compute trigonometric values
    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)
    
    # Create rotation matrices for each axis
    # Rotation around x-axis
    Rx = torch.stack([
        torch.stack([torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx)], dim=-1),
        torch.stack([torch.zeros_like(rx), cos_x, -sin_x], dim=-1),
        torch.stack([torch.zeros_like(rx), sin_x, cos_x], dim=-1)
    ], dim=-2)
    
    # Rotation around y-axis
    Ry = torch.stack([
        torch.stack([cos_y, torch.zeros_like(ry), sin_y], dim=-1),
        torch.stack([torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry)], dim=-1),
        torch.stack([-sin_y, torch.zeros_like(ry), cos_y], dim=-1)
    ], dim=-2)
    
    # Rotation around z-axis
    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros_like(rz)], dim=-1),
        torch.stack([sin_z, cos_z, torch.zeros_like(rz)], dim=-1),
        torch.stack([torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)], dim=-1)
    ], dim=-2)
    
    # Combine rotations: R = Rz * Ry * Rx (ZYX convention)
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    
    return R

def euler_to_aa(euler):
    rotmat = euler_to_rotmat(euler)
    aa = rotmat2aa(rotmat)
    return aa

def transform_meshes(meshes, rot_mats, trans_vecs):
    """Apply rotation and translation to meshes."""

    transformed_meshes = []
    if isinstance(rot_mats, list): rot_mats = np.asarray(rot_mats)
    if isinstance(trans_vecs, list): trans_vecs = np.asarray(trans_vecs)
    
    assert rot_mats.dtype == trans_vecs.dtype

    for mesh, rot, trans in zip(meshes, rot_mats, trans_vecs):
        mesh_copy = mesh.copy()

        # Create 4x4 transform matrix
        tranform_matrix = np.eye(4) if isinstance(rot, np.ndarray) else torch.eye(4, dtype=rot.dtype)
        tranform_matrix[:3, :3] = rot
        tranform_matrix[:3, 3] = trans
        
        mesh_copy.apply_transform(tranform_matrix.T)
        transformed_meshes.append(mesh_copy)
    transformed_verts = [mesh.vertices for mesh in transformed_meshes]
    transformed_faces = [mesh.faces for mesh in transformed_meshes]
    return transformed_meshes, np.asarray(transformed_verts), np.asarray(transformed_faces, dtype=object)
