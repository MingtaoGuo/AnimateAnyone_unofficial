import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from aldm.model import create_model
#python tool_add_reference.py ./models/v1-5-pruned.ckpt ./models/reference_sd15_ini.ckpt

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/aldm_v15.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

# =============== ReferenceNet and ReferenceUNetModel ================
scratch_dict = model.state_dict()
target_dict = {}
for k in scratch_dict.keys():
    is_reference, name = get_node_name(k, 'reference_') 
    if is_reference:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

# ======================= CLIP Vision Embedder ========================
clip_vision_dict = torch.load("models/clip_vit_base_patch32.pt")
for k in clip_vision_dict.keys():
    copy_k = "cond_stage_model.transformer.vision_model." + k
    target_dict[copy_k] = clip_vision_dict[k] 
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
