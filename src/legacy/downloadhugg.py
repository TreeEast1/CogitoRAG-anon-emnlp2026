from transformers import AutoModel

print("Start downloading model")
model = AutoModel.from_pretrained(
    "nvidia/NV-Embed-v2",
    trust_remote_code=True,
)
print("Model downloaded successfully!")

