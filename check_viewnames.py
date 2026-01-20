import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os

config_dir = os.path.join(os.getcwd(), 'config')
with initialize_config_dir(config_dir, version_base="1.2"):
    cfg = compose(config_name="main", overrides=["+experiment=nersemble"])
    print("Dataset config keys:", list(cfg.dataset.keys()))
    if 'view_names' in cfg.dataset:
        print("view_names:", cfg.dataset.view_names)
    else:
        print("view_names missing")
    # Print full dataset config
    print(OmegaConf.to_yaml(cfg.dataset))