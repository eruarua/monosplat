export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python3 -m src.main \
    +experiment=nersemble \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.name=arbitrary \
    checkpointing.pretrained_monodepth=/data/baosongze/torch_hub/hub/checkpoints/dinov2_vits14_pretrain.pth
