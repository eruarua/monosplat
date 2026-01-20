import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetNersembleCfg(DatasetCfgCommon):
    name: Literal["nersemble"]
    roots: list[Path]
    view_names: list[str]
    make_baseline_1: bool = False
    baseline_epsilon: float = 0.001
    near: float = 0.1
    far: float = 100.0
    shuffle_val: bool = True


class DatasetNersemble(IterableDataset):
    cfg: DatasetNersembleCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    scenes: list[dict]

    def __init__(
        self,
        cfg: DatasetNersembleCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.scenes = []
        for root_raw in cfg.roots:
            root = Path(root_raw)
            # If root has subdirectory named after stage (train/val/test), use it
            stage_root = root / self.stage
            if stage_root.exists() and stage_root.is_dir():
                root = stage_root
                print(f"Using stage-specific root: {root}")
            # id_list = [x for x in sorted(glob.glob(os.path.join(self.base_path, "*"))) if os.path.isdir(x)]
            id_list = sorted([x for x in root.iterdir() if x.is_dir()])
            print(f"Root {root} has {len(id_list)} ids")
            for id_path in id_list:
                image_root = id_path / "image"
                pose_path = id_path / "pose" / "cam_param.json"
                if not image_root.exists():
                    print(f"  Image root missing: {image_root}")
                    continue
                if not pose_path.exists():
                    print(f"  Pose path missing: {pose_path}")
                    continue

                frame_folders = sorted([x for x in image_root.iterdir() if x.is_dir() and x.name != "BACKGROUND"])
                # print(f"  Id {id_path.name} has {len(frame_folders)} frames")
                for frame_folder in frame_folders:
                    self.scenes.append({
                        "id": id_path.name,
                        "frame": frame_folder.name,
                        "rgb_folder": frame_folder,
                        "pose_path": pose_path
                    })
        print(f"Total scenes loaded: {len(self.scenes)}")

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.scenes = self.shuffle(self.scenes)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.scenes = [
                scene
                for scene_index, scene in enumerate(self.scenes)
                if scene_index % worker_info.num_workers == worker_info.id
            ]

        # print(f"Iterating over {len(self.scenes)} scenes, stage={self.stage}")
        for scene_idx, scene in enumerate(self.scenes):
            try:
                # print(f"Loading scene {scene_idx}: {scene['id']}_{scene['frame']}")
                example = self.load_scene(scene)
                if self.stage == "train":
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
            except Exception as e:
                import traceback, sys
                print(f"Error loading scene {scene['id']}_{scene['frame']}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                continue

    def load_scene(self, scene: dict) -> dict:
        # print(f"load_scene: {scene['id']}_{scene['frame']}")
        # print(f"cfg type: {type(self.cfg)}")
        # print(f"cfg keys: {dir(self.cfg)}")
        # if hasattr(self.cfg, 'view_names'):
        #     print(f"view_names attr: {self.cfg.view_names}")
        # else:
        #     print("view_names not an attribute")
        rgb_folder = scene["rgb_folder"]
        pose_path = scene["pose_path"]
        view_names = self.cfg.view_names
        
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            
        all_images = []
        all_intrinsics = []
        all_extrinsics = []
        for view_name in view_names:
            rgb_path = rgb_folder / f"cam_{view_name}.png"
            img = Image.open(rgb_path)
            w, h = img.size
            short = min(w, h)
            left = (w - short) // 2
            top = (h - short) // 2
            right = left + short
            bottom = top + short

            img = img.crop((left, top, right, bottom))
            all_images.append(self.to_tensor(img))
            
            # Intrinsics
            intr = np.array(pose_data['intrinsics'], dtype=np.float64)
            # Intrinsics in cam_param.json are already normalized by resolution?
            # Looking at nersemble.py, they seem to be absolute pixels and then scaled.
            # In MonoSplat, they are usually normalized (0-1).
            # Let's check how they are used in MonoSplat.
            # In DatasetRE10k:
            # fx, fy, cx, cy = poses[:, :4].T
            # intrinsics[:, 0, 0] = fx
            # intrinsics[:, 1, 1] = fy
            # intrinsics[:, 0, 2] = cx
            # intrinsics[:, 1, 2] = cy
            # RE10k poses usually have normalized intrinsics.
            
            # For Nersemble, we need to normalize them.
            
            intr_normalized = intr.copy()
            # 进行center crop
            
            dx = (w - short) / 2.0
            dy = (h - short) / 2.0
            intr_normalized[0, 2] -= dx
            intr_normalized[1, 2] -= dy

            intr_normalized[0, 0] /= short
            intr_normalized[1, 1] /= short
            intr_normalized[0, 2] /= short
            intr_normalized[1, 2] /= short


            
            all_intrinsics.append(torch.from_numpy(intr_normalized).float())
            
            # Extrinsics: Data provides world-to-camera (w2c) matrices
            # MonoSplat uses camera-to-world (c2w) format internally
            # So we need to invert the w2c matrix to get c2w
            w2c = np.array(pose_data['world_2_cam'][view_name], dtype=np.float64)  # [4, 4] w2c matrix
            w2c_torch = torch.eye(4, dtype=torch.float32)
            w2c_torch[:3, :4] = torch.from_numpy(w2c[:3, :]).float()
            c2w = w2c_torch.inverse()  # Convert to camera-to-world
            all_extrinsics.append(c2w)

        images = torch.stack(all_images)
        intrinsics = torch.stack(all_intrinsics)
        extrinsics = torch.stack(all_extrinsics)
        print(f"images shape: {images.shape}")
        print(f"intrinsics shape: {intrinsics.shape}")
        print(f"extrinsics shape: {extrinsics.shape}")
        
        scene_name = f"{scene['id']}_{scene['frame']}"
        
        context_indices, target_indices = self.view_sampler.sample(
            scene_name,
            extrinsics,
            intrinsics,
        )
        # print(f"context_indices: {context_indices}, target_indices: {target_indices}")
        
        # Resize the world to make the baseline 1 if requested
        # extrinsics are camera-to-world (c2w) matrices, so [:3, 3] gives camera center in world
        context_extrinsics = extrinsics[context_indices]
        # print(f"context_extrinsics shape: {context_extrinsics.shape}")
        if context_extrinsics.shape[0] >= 2 and self.cfg.make_baseline_1:
            # Get camera centers from c2w matrices
            centers = context_extrinsics[:, :3, 3] # [N, 3] camera positions in world
            
            if context_extrinsics.shape[0] == 2:
                # Standard case: distance between two cameras
                scale = (centers[0] - centers[1]).norm()
            else:
                # Multi-camera case: average pairwise distance
                # This generalizes the N=2 case
                dist_matrix = torch.cdist(centers, centers)
                # Sum upper triangle and divide by number of pairs
                num_cameras = centers.shape[0]
                num_pairs = num_cameras * (num_cameras - 1) / 2
                scale = dist_matrix.triu(diagonal=1).sum() / num_pairs

            if scale > self.cfg.baseline_epsilon:
                extrinsics[:, :3, 3] /= scale
        else:
            scale = 1.0

        # Normalize extrinsics relative to the first context camera
        # This puts all cameras in the coordinate frame of the first context camera
        # extrinsics contains camera-to-world (c2w) matrices
        if len(context_indices) > 0:
            # Get the first context camera's extrinsics (c2w matrix)
            ref_c2w = extrinsics[context_indices[0]]  # [4, 4] c2w of reference camera
            # Compute inverse to get world-to-camera (w2c) of reference camera
            ref_w2c = ref_c2w.inverse()
            # Apply normalization to all cameras: c2w_norm_i = w2c_ref @ c2w_i
            # This transforms all cameras to the coordinate frame of the first context camera
            # Result: first context camera becomes identity, others are relative to it
            extrinsics = ref_w2c @ extrinsics

        # print(f"Scene loaded successfully: {scene_name}")
        return {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": images[context_indices],
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": images[target_indices],
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": scene_name,
        }

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.scenes)
