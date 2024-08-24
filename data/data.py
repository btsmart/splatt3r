import random

import numpy as np
import PIL
import torch
import torchvision

from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
from src.mast3r_src.dust3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf
from src.mast3r_src.dust3r.dust3r.utils.misc import invalid_to_zeros
import src.mast3r_src.dust3r.dust3r.datasets.utils.cropping as cropping


def crop_resize_if_necessary(image, depthmap, intrinsics, resolution):
    """Adapted from DUST3R's Co3D dataset implementation"""

    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # Downscale with lanczos interpolation so that image.size == resolution cropping centered on the principal point
    # The new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    assert min_margin_x > W / 5
    assert min_margin_y > H / 5
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # High-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    # Actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2


class DUST3RSplattingDataset(torch.utils.data.Dataset):

    def __init__(self, data, coverage, resolution, num_epochs_per_epoch=1, alpha=0.3, beta=0.3):

        super(DUST3RSplattingDataset, self).__init__()
        self.data = data
        self.coverage = coverage

        self.num_context_views = 2
        self.num_target_views = 3

        self.resolution = resolution
        self.transform = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()
        self.num_epochs_per_epoch = num_epochs_per_epoch

        self.alpha = alpha
        self.beta = beta

    def __getitem__(self, idx):

        sequence = self.data.sequences[idx // self.num_epochs_per_epoch]
        sequence_length = len(self.data.color_paths[sequence])

        context_views, target_views = self.sample(sequence, self.num_target_views, self.alpha, self.beta)

        views = {"context": [], "target": [], "scene": sequence}

        # Fetch the context views
        for c_view in context_views:

            assert c_view < sequence_length, f"Invalid view index: {c_view}, sequence length: {sequence_length}, c_views: {context_views}"

            view = self.data.get_view(sequence, c_view, self.resolution)

            # Transform the input
            view['img'] = self.transform(view['original_img'])
            view['original_img'] = self.org_transform(view['original_img'])

            # Create the point cloud and validity mask
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            assert view['valid_mask'].any(), f"Invalid mask for sequence: {sequence}, view: {c_view}"

            views['context'].append(view)

        # Fetch the target views
        for t_view in target_views:

            view = self.data.get_view(sequence, t_view, self.resolution)
            view['original_img'] = self.org_transform(view['original_img'])
            views['target'].append(view)

        return views

    def __len__(self):

        return len(self.data.sequences) * self.num_epochs_per_epoch

    def sample(self, sequence, num_target_views, context_overlap_threshold=0.5, target_overlap_threshold=0.6):

        first_context_view = random.randint(0, len(self.data.color_paths[sequence]) - 1)

        # Pick a second context view that has sufficient overlap with the first context view
        valid_second_context_views = []
        for frame in range(len(self.data.color_paths[sequence])):
            if frame == first_context_view:
                continue
            overlap = self.coverage[sequence][first_context_view][frame]
            if overlap > context_overlap_threshold:
                valid_second_context_views.append(frame)
        if len(valid_second_context_views) > 0:
            second_context_view = random.choice(valid_second_context_views)

        # If there are no valid second context views, pick the best one
        else:
            best_view = None
            best_overlap = None
            for frame in range(len(self.data.color_paths[sequence])):
                if frame == first_context_view:
                    continue
                overlap = self.coverage[sequence][first_context_view][frame]
                if best_view is None or overlap > best_overlap:
                    best_view = frame
                    best_overlap = overlap
            second_context_view = best_view

        # Pick the target views
        valid_target_views = []
        for frame in range(len(self.data.color_paths[sequence])):
            if frame == first_context_view or frame == second_context_view:
                continue
            overlap_max = max(
                self.coverage[sequence][first_context_view][frame],
                self.coverage[sequence][second_context_view][frame]
            )
            if overlap_max > target_overlap_threshold:
                valid_target_views.append(frame)
        if len(valid_target_views) >= num_target_views:
            target_views = random.sample(valid_target_views, num_target_views)

        # If there are not enough valid target views, pick the best ones
        else:
            overlaps = []
            for frame in range(len(self.data.color_paths[sequence])):
                if frame == first_context_view or frame == second_context_view:
                    continue
                overlap = max(
                    self.coverage[sequence][first_context_view][frame],
                    self.coverage[sequence][second_context_view][frame]
                )
                overlaps.append((frame, overlap))
            overlaps.sort(key=lambda x: x[1], reverse=True)
            target_views = [frame for frame, _ in overlaps[:num_target_views]]

        return [first_context_view, second_context_view], target_views


class DUST3RSplattingTestDataset(torch.utils.data.Dataset):

    def __init__(self, data, samples, resolution):

        self.data = data
        self.samples = samples

        self.resolution = resolution
        self.transform = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()

    def get_view(self, sequence, c_view):

        view = self.data.get_view(sequence, c_view, self.resolution)

        # Transform the input
        view['img'] = self.transform(view['original_img'])
        view['original_img'] = self.org_transform(view['original_img'])

        # Create the point cloud and validity mask
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        view['pts3d'] = pts3d
        view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
        assert view['valid_mask'].any(), f"Invalid mask for sequence: {sequence}, view: {c_view}"

        return view

    def __getitem__(self, idx):

        sequence, c_view_1, c_view_2, target_view = self.samples[idx]
        c_view_1, c_view_2, target_view = int(c_view_1), int(c_view_2), int(target_view)
        fetched_c_view_1 = self.get_view(sequence, c_view_1)
        fetched_c_view_2 = self.get_view(sequence, c_view_2)
        fetched_target_view = self.get_view(sequence, target_view)

        views = {"context": [fetched_c_view_1, fetched_c_view_2], "target": [fetched_target_view], "scene": sequence}

        return views

    def __len__(self):

        return len(self.samples)
