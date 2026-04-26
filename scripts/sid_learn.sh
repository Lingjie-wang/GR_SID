#!/bin/bash
#SBATCH --job-name=sid_learn_rqvae_restart          # 作业名称
#SBATCH --output=logs/slurm_logs/%x-%j.out             # 标准输出文件 (%j 会被替换为作业ID)
#SBATCH --error=logs/slurm_logs/%x-%j.err              # 错误输出文件
#SBATCH --partition=GPUNorm           # 指定分区（根据集群情况修改）
#SBATCH --nodes=1                   # 请求节点数
#SBATCH --ntasks=1                  # 任务数（通常1个）
#SBATCH --cpus-per-task=4           # 每个任务需要的CPU核心数
#SBATCH --mem=8G                    # 内存大小
#SBATCH --time=01:00:00             # 预计运行时间 (时:分:秒)

# 加载环境（如果有模块系统）
# module load python/3.9

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate grid

# 执行你的程序
python src/train.py experiment=rqvae_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=logs/inference/runs/2026-03-13/11-09-53/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.quantization_layer.restart_unused_codes=true \
    ++should_skip_retry=True