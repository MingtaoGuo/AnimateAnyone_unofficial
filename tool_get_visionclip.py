import torch
from transformers import CLIPVisionModel, CLIPModel


if __name__ == "__main__":
    cv = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    torch.save(model.vision_model.state_dict(), "models/clip_vit_base_patch32.pt")

    state_dict = torch.load("models/clip_vit_base_patch32.pt")
    new_state_dict = {}
    for k, v in state_dict.items():
        k = "vision_model." + k
        new_state_dict[k] = v 

    cv.load_state_dict(new_state_dict)
    print("done")







    

    