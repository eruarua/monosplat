import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
from src.dataset.dataset_nersemble import DatasetNersemble, DatasetNersembleCfg
from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from src.misc.step_tracker import StepTracker

# Create view sampler config
view_sampler_cfg = ViewSamplerArbitraryCfg(
    name="arbitrary",
    num_context_views=3,
    num_target_views=4,
    context_views=[0, 2, 4],
    target_views=[1, 3, 5, 6]
)

# Create dataset config
cfg = DatasetNersembleCfg(
    name="nersemble",
    roots=[Path("/data/baosongze/workspace/3D/3D_debug_data/nersemble_data_expand/train")],
    view_names=['222200037', '220700191', '221501007', '222200036', '222200046', '222200047', '222200049'],
    make_baseline_1=False,
    baseline_epsilon=0.001,
    near=0.1,
    far=100.0,
    shuffle_val=True,
    image_shape=[360, 640],
    background_color=[0.0, 0.0, 0.0],
    cameras_are_circular=False,
    overfit_to_scene=None,
    view_sampler=view_sampler_cfg,
)

# Create view sampler
step_tracker = StepTracker()
view_sampler = ViewSamplerArbitrary(
    view_sampler_cfg,
    "train",
    False,  # is_overfitting
    False,  # cameras_are_circular
    step_tracker
)

# Create dataset
dataset = DatasetNersemble(cfg, "train", view_sampler)
print(f"Dataset has {len(dataset)} scenes")

# Try to load just the first scene
if len(dataset.scenes) > 0:
    scene = dataset.scenes[0]
    print(f"Testing scene: {scene['id']}_{scene['frame']}")
    try:
        # Call load_scene directly
        result = dataset.load_scene(scene)
        print("Success! Loaded scene with keys:", result.keys())
        print(f"Context images shape: {result['context']['image'].shape}")
        print(f"Target images shape: {result['target']['image'].shape}")
        print(f"Context extrinsics shape: {result['context']['extrinsics'].shape}")
        print(f"Target extrinsics shape: {result['target']['extrinsics'].shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No scenes loaded")