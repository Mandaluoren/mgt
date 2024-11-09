from huggingface_hub import snapshot_download

SAVED_DIR = "./"  # 指定保存目录

# 下载 HF checkpoints，忽略 .pt 和 .safetensors 文件
snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-v0.1",
    ignore_patterns=["*.pt", "*.safetensors"],
    local_dir=SAVED_DIR,
    local_dir_use_symlinks=False
)
