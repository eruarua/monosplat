import hydra
from omegaconf import DictConfig
import torch
from src.dataset import get_dataset
from src.misc.step_tracker import StepTracker

@hydra.main(config_path="config", config_name="main", version_base="1.2")
def test_dataset(cfg: DictConfig):
    cfg.dataset.name = "nersemble"
    cfg.dataset.roots = ["/data/baosongze/workspace/3D/3D_debug_data/nersemble_data_expand"]
    
    step_tracker = StepTracker()
    dataset = get_dataset(cfg.dataset, "train", step_tracker)
    
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
