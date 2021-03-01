import torch.utils.data
from utils.torch_utils import select_device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load('weights/best.pt', map_location=device)
torch.save(ckpt, "weights/new.pt", _use_new_zipfile_serialization=False)