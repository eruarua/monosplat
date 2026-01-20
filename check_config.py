import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="main", version_base="1.2")
def check(cfg: DictConfig):
    # Apply experiment override
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from hydra._internal.hydra import Hydra
    import sys
    # Simulate command line overrides
    overrides = ["+experiment=nersemble"]
    config_loader = ConfigLoaderImpl(
        config_search_path=sys.path,
        default_config_files=[],
        config_name="main",
        version_base="1.2",
    )
    config = config_loader.load_configuration(
        overrides=overrides,
        strict=False,
    )
    cfg = config.cfg
    print("Dataset config keys:", list(cfg.dataset.keys()))
    print("Dataset config:")
    print(OmegaConf.to_yaml(cfg.dataset))
    # Check view_names
    if 'view_names' in cfg.dataset:
        print("view_names:", cfg.dataset.view_names)
    else:
        print("view_names missing")
    # Check if cfg.dataset is instance of DatasetNersembleCfg
    print("cfg.dataset._metadata:", cfg.dataset._metadata if hasattr(cfg.dataset, '_metadata') else 'no metadata')

if __name__ == "__main__":
    check()