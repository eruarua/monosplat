import sys
sys.path.insert(0, '.')

from src.dataset.dataset_nersemble import DatasetNersemble, DatasetNersembleCfg
from src.dataset.view_sampler import ViewSampler
from pathlib import Path

# Create config
cfg = DatasetNersembleCfg(
    name="nersemble",
    roots=[Path("/data/baosongze/workspace/3D/3D_debug_data/nersemble_data_expand/train")],
    view_names=['222200037', '220700191', '221501007', '222200036', '222200046', '222200047', '222200049'],
    make_baseline_1=False,
    baseline_epsilon=0.001,
    near=0.1,
    far=100.0,
    shuffle_val=True,
    image_shape=[512, 512],
    background_color=[0.0, 0.0, 0.0],
    cameras_are_circular=False,
    augment=True,
)

# Create view sampler (arbitrary)
from src.dataset.view_sampler.arbitrary import ViewSamplerArbitrary
view_sampler_cfg = type('Cfg', (), {
    'name': 'arbitrary',
    'num_target_views': 3,
    'num_context_views': 4,
    'context_views': [0, 2, 4],
    'target_views': [1, 3, 5, 6],
})()
view_sampler = ViewSamplerArbitrary(view_sampler_cfg)

# Create dataset
dataset = DatasetNersemble(cfg, stage="train", view_sampler=view_sampler)
print(f"Number of scenes: {len(dataset)}")
print(f"Scenes list: {dataset.scenes[:2] if dataset.scenes else 'empty'}")

# Try to iterate
count = 0
for example in dataset:
    print(f"Example keys: {example.keys()}")
    print(f"Context image shape: {example['context']['image'].shape}")
    print(f"Target image shape: {example['target']['image'].shape}")
    count += 1
    if count >= 2:
        break
print(f"Successfully loaded {count} examples")