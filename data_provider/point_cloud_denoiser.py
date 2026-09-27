import cv2
import numpy as np


class PointCloudDenoiser:
    """Utilities for denoising object masks and point clouds."""

    @staticmethod
    def erode_binary_mask(binary_mask, kernel_size=3, iterations=1):
        """Shrink a binary mask to remove small noisy boundary regions."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(binary_mask, kernel, iterations=iterations)

    @staticmethod
    def compute_depth_gradient(depth_raw):
        """Compute per-pixel depth gradient magnitude from a depth map."""
        sobel_x = cv2.Sobel(depth_raw, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_raw, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    @staticmethod
    def filter_mask_by_depth_gradient(binary_mask, depth_gradient, percentile=85):
        """Remove mask pixels whose depth gradients are above a percentile threshold."""
        valid_gradients = depth_gradient[binary_mask > 0]
        if valid_gradients.size == 0:
            return binary_mask

        gradient_threshold = np.percentile(valid_gradients, percentile)
        filtered_mask = binary_mask.copy()
        filtered_mask[depth_gradient > gradient_threshold] = 0
        return filtered_mask

    @staticmethod
    def keep_largest_connected_component(mask):
        """Keep only the largest connected component in a binary mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 1
        return clean_mask

    @classmethod
    def denoise_scene_object_mask(
        cls,
        mask_raw,
        depth_raw,
        erosion_kernel_size=3,
        erosion_iterations=2,
        gradient_percentile=92,
    ):
        """Denoise an object mask using erosion, depth gradients, and component filtering."""
        binary_mask = (mask_raw > 0).astype(np.uint8)
        binary_mask = cls.erode_binary_mask(
            binary_mask,
            kernel_size=erosion_kernel_size,
            iterations=erosion_iterations,
        )
        depth_gradient = cls.compute_depth_gradient(depth_raw)
        binary_mask = cls.filter_mask_by_depth_gradient(
            binary_mask,
            depth_gradient,
            percentile=gradient_percentile,
        )
        return binary_mask

    @staticmethod
    def denoise_point_cloud_statistical_outlier(pcd, nb_neighbors=20, std_ratio=2.0):
        """Remove sparse outlier points from a point cloud with statistical filtering."""
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        return pcd.select_by_index(ind)
