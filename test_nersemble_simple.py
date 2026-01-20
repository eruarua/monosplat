import torch
from pathlib import Path
from src.dataset.dataset_nersemble import DatasetNersemble, DatasetNersembleCfg
from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from src.misc.step_tracker import StepTracker

def test_dataset():
    view_sampler_cfg = ViewSamplerBoundedCfg(
        name="bounded",
        num_context_views=2,
        num_target_views=1,
        min_distance_between_context_views=0.0,
        max_distance_between_context_views=100.0,
        min_distance_to_context_views=0.0,
        warm_up_steps=0,
    )
    
    view_sampler = ViewSamplerBounded(
        view_sampler_cfg,
        "train",
        False,
        False,
        None
    )

    cfg = DatasetNersembleCfg(
        name="nersemble",
        roots=[Path("/data/baosongze/workspace/3D/3D_debug_data/nersemble_data_expand")],
        view_names=['222200037', '220700191', '221501007', '222200036', '222200046', '222200047', '222200049'],
        image_shape=[360, 640],
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=view_sampler_cfg,
        near=0.1,
        far=100.0
    )
    
    dataset = DatasetNersemble(cfg, "train", view_sampler)
    
    print(f"Dataset length: {len(dataset)}")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Context images shape: {batch['context']['image'].shape}")
        print(f"  Target images shape: {batch['target']['image'].shape}")
        print(f"  Scene: {batch['scene']}")
        if i >= 2:
            break

if __name__ == "__main__":
    test_dataset()
