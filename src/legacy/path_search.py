from transformers.utils import cached_file
import os

model_id = "nvidia/NV-Embed-v2"
filename = "config.json"
file_path = cached_file(model_id, filename, trust_remote_code=True)
print("缓存目录为：", os.path.dirname(file_path))
