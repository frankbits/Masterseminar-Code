import torch
print("CUDA Verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Aktuelle GPU:", torch.cuda.get_device_name(0))
