import json
import logging
import os
import sys

import cv2
import numpy as np

# Add dust3r to the sys.path
sys.path.append('src/dust3r_src')
from data.data import crop_resize_if_necessary, DUST3RSplattingDataset, DUST3RSplattingTestDataset
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2

logger = logging.getLogger(__name__)


class ScanNetPPData():

    def __init__(self, root, stage):

        self.root = root
        self.stage = stage
        self.png_depth_scale = 1000.0

        # Dictionaries to store the data for each scene
        self.color_paths = {}
        self.depth_paths = {}
        self.intrinsics = {}
        self.c2ws = {}

        # Fetch the sequences to use
        if stage == "train":
            sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_train.txt")
            bad_scenes = ['303745abc7']
        elif stage == "val" or stage == "test":
            sequence_file = os.path.join(self.root, "raw", "splits", "nvs_sem_val.txt")
            bad_scenes = ['cc5237fd77']
        with open(sequence_file, "r") as f:
            self.sequences = f.read().splitlines()

        # Remove scenes that have frames with no valid depths
        logger.info(f"Removing scenes that have frames with no valid depths: {bad_scenes}")
        self.sequences = [s for s in self.sequences if s not in bad_scenes]

        P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        ).astype(np.float32)

        # Collect information for every sequence
        scenes_with_no_good_frames = []
        for sequence in self.sequences:

            input_raw_folder = os.path.join(self.root, 'raw', 'data', sequence)
            input_processed_folder = os.path.join(self.root, 'processed', sequence)

            # Load Train & Test Splits
            frame_file = os.path.join(input_raw_folder, "dslr", "train_test_lists.json")
            with open(frame_file, "r") as f:
                train_test_list = json.load(f)

            # Camera Metadata
            cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
            with open(cams_metadata_path, "r") as f:
                cams_metadata = json.load(f)

            # Load the nerfstudio/transforms.json file to check whether each image is blurry
            nerfstudio_transforms_path = f"{input_raw_folder}/dslr/nerfstudio/transforms.json"
            with open(nerfstudio_transforms_path, "r") as f:
                nerfstudio_transforms = json.load(f)

            # Create a reverse mapping from image name to the frame information and nerfstudio transform
            # (as transforms_undistorted.json does not store the frames in the same order as train_test_lists.json)
            file_path_to_frame_metadata = {}
            file_path_to_nerfstudio_transform = {}
            for frame in cams_metadata["frames"]:
                file_path_to_frame_metadata[frame["file_path"]] = frame
            for frame in nerfstudio_transforms["frames"]:
                file_path_to_nerfstudio_transform[frame["file_path"]] = frame

            # Fetch the pose for every frame
            sequence_color_paths = []
            sequence_depth_paths = []
            sequence_c2ws = []
            for train_file_name in train_test_list["train"]:
                is_bad = file_path_to_nerfstudio_transform[train_file_name]["is_bad"]
                if is_bad:
                    continue
                sequence_color_paths.append(f"{input_processed_folder}/dslr/undistorted_images/{train_file_name}")
                sequence_depth_paths.append(f"{input_processed_folder}/dslr/undistorted_depths/{train_file_name.replace('.JPG', '.png')}")
                frame_metadata = file_path_to_frame_metadata[train_file_name]
                c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
                c2w = P @ c2w @ P.T
                sequence_c2ws.append(c2w)

            if len(sequence_color_paths) == 0:
                logger.info(f"No good frames for sequence: {sequence}")
                scenes_with_no_good_frames.append(sequence)
                continue

            # Get the intrinsics data for the frame
            K = np.eye(4, dtype=np.float32)
            K[0, 0] = cams_metadata["fl_x"]
            K[1, 1] = cams_metadata["fl_y"]
            K[0, 2] = cams_metadata["cx"]
            K[1, 2] = cams_metadata["cy"]

            self.color_paths[sequence] = sequence_color_paths
            self.depth_paths[sequence] = sequence_depth_paths
            self.c2ws[sequence] = sequence_c2ws
            self.intrinsics[sequence] = K

        # Remove scenes with no good frames
        self.sequences = [s for s in self.sequences if s not in scenes_with_no_good_frames]

    def get_view(self, sequence, view_idx, resolution):

        # RGB Image
        rgb_path = self.color_paths[sequence][view_idx]
        rgb_image = imread_cv2(rgb_path)

        # Depthmap
        depth_path = self.depth_paths[sequence][view_idx]
        depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32)
        depthmap = depthmap / self.png_depth_scale

        # C2W Pose
        c2w = self.c2ws[sequence][view_idx]

        # Camera Intrinsics
        intrinsics = self.intrinsics[sequence]

        # Resize
        rgb_image, depthmap, intrinsics = crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution
        )

        view = {
            'original_img': rgb_image,
            'depthmap': depthmap,
            'camera_pose': c2w,
            'camera_intrinsics': intrinsics,
            'dataset': 'scannet++',
            'label': f"scannet++/{sequence}",
            'instance': f'{view_idx}',
            'is_metric_scale': True,
            'sky_mask': depthmap <= 0.0,
        }
        return view


def get_scannet_dataset(root, stage, resolution, num_epochs_per_epoch=1):

    data = ScanNetPPData(root, stage)

    coverage = {}
    for sequence in data.sequences:
        with open(f'./data/scannetpp/coverage/{sequence}.json', 'r') as f:
            sequence_coverage = json.load(f)
        coverage[sequence] = sequence_coverage[sequence]

    dataset = DUST3RSplattingDataset(
        data,
        coverage,
        resolution,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )

    return dataset


def get_scannet_test_dataset(root, alpha, beta, resolution, use_every_n_sample=100):

    data = ScanNetPPData(root, 'val')

    samples_file = f'data/scannetpp/test_set_{alpha}_{beta}.json'
    print(f"Loading samples from: {samples_file}")
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    samples = samples[::use_every_n_sample]

    dataset = DUST3RSplattingTestDataset(data, samples, resolution)

    return dataset
