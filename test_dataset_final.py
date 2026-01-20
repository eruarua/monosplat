import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os
import torch
from src.dataset import get_dataset
from src.misc.step_tracker import StepTracker

config_dir = os.path.join(os.getcwd(), 'config')
with initialize_config_dir(config_dir, version_base="1.2"):
    cfg = compose(config_name="main", overrides=["+experiment=nersemble", "data_loader.train.num_workers=0"])
    print("Dataset config loaded")
    step_tracker = StepTracker()
    dataset = get_dataset(cfg.dataset, "train", step_tracker)
    print(f"Dataset length: {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Context images shape: {batch['context']['image'].shape}")
        print(f"  Target images shape: {batch['target']['image'].shape}")
        print(f"  Scene: {batch['scene']}")
        if i >= 2:
            break
    print("Success!")