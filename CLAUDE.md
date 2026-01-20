# CLAUDE.md - 项目开发指引

## 🛠️ 常用命令 (Build/Run/Test)
- **运行项目**: bash train_nersemble.sh 基于nersemble 数据集开始模型训练


## 🏗️ 技术栈 (Tech Stack)
- **语言**: python
- **conda环境**: 路径 /data/baosongze/env/splatter_image
- **依赖库**: 需要的安装包记录在requirements.txt中
- **可视化**：通过wandb可视化训练进程，调试时不适用，正式训练时使用
- **技能**: 精通pytorch，torchvision等库的使用，对多视角3D重建算法有深入了解，熟悉3D GS

## 📂 目录结构说明 (Architecture)
- `src/dataset/dataset_nersemble.py`: 自定义的nersemble数据读取接口
- `train_nersemble.sh`: 模型训练入口