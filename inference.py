import torch
from dinov2.models.vision_transformer import vit_base

model = vit_base()

state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
model.load_state_dict(state_dict)
