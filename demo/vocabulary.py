import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import numpy as np
import einops as ein
import os
from tqdm.auto import tqdm
from typing import Literal
from pathlib import Path
from termcolor import cprint
import pickle
# DINOv2 imports
from utilities import DinoV2ExtractFeatures
from utilities import VLAD
from utilities import LocalArgs



# colormap
# green: process or result is OK
# yellow: important info
# blue: notification

def create_vocabulary(imgs_dir: Path, 
                      _largs: LocalArgs, 
                      _tf: tvf.Compose, 
                      _extractor: DinoV2ExtractFeatures):
    # DINO features extraction
    # extract features from all images
    cprint(f"Extracting features from images at {imgs_dir} ...", "yellow")
    names = []
    features_t = []
    for img in tqdm(imgs_dir.iterdir()):
        if Image.open(img).mode != "RGB":
            cprint(f"Skipping image {img}", "red")
            continue
        else:
            names.append(os.path.basename(img).split(".")[0])
        
        # Load image in tensor format
        img_t = _tf(Image.open(img)).to(torch.device("cuda"))
        if max(img_t.shape[-2:]) > _largs.max_img_size:
            c, h, w = img_t.shape
            # Maintain aspect ratio
            if h > w:
                resized_height = _largs.max_img_size
                resized_width = int(w * _largs.max_img_size / h)
            else:
                resized_width = _largs.max_img_size
                resized_height = int(h * _largs.max_img_size / w)

            img_t = T.resize(img_t, 
                             (resized_height, resized_width), 
                             interpolation=T.InterpolationMode.BICUBIC)
        # print(f"Feeding image shape: {img_t.shape}")
            
        # pre-process the image 
        _, h, w = img_t.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_t = tvf.CenterCrop((h_new, w_new))(img_t)[None, ...]
        
        feature = _extractor(img_t)
        features_t.append(feature)
    #feature:[batch_size, num_descriptors, descriptor_dim] 
    cprint(f"DINO features extracted features as ({features_t[0].shape}) shape", "green")
        

    # rearrange the features into numpy format for fit method
    features_np = [f.cpu().numpy() for f in features_t]
    features_np = np.concatenate(features_np, axis=0)
    features_np = ein.rearrange(features_np, 'n k d -> (n k) d')

    # create clusters (VLAD)
    vlad = VLAD(num_clusters=_largs.num_c, desc_dim=features_np.shape[1], 
                intra_norm=True, norm_descs=True, 
                dist_mode="cosine", vlad_mode="hard")
    cprint(f"Features shape for VLAD: {features_np.shape}", "green")
    vlad.fit(features_np)

    # save the VLAD object
    cprint(f"Saving VLAD object at {_largs.out_dir} ...", "yellow")
    torch.save(vlad, f"{_largs.out_dir}/vlad.pt")


    # save the vocabulary
    cprint(f"Saving vocabulary at {_largs.out_dir} ...", "yellow")
    for inx, feature in enumerate(tqdm(features_t)):
        # feature_t = feature.reshape(-1, feature.shape[-1])
        feature_np = feature.cpu().squeeze()
        assert len(feature_np.shape) == 2, "Feature shape is not 2D"
        voc = vlad.generate(feature_np)
        voc_np = voc.numpy()[np.newaxis, ...]

        w_path = f"{_largs.out_dir}/{names[inx]}.npy"
        np.save(w_path, voc_np)
        
    cprint(f"Vocabulary saved at {_largs.out_dir}", "green")
    


def vocabulary_info(pt_path: Path):
    voc = torch.load(pt_path)
    print(f"Vocabulary shape: {voc.shape}")
    print(f"Vocabulary mean: {voc.mean()}")
    print(f"Vocabulary std: {voc.std()}")
    print(f"Vocabulary dtype: {voc.dtype}")


if __name__ == "__main__":
    myLocalArgs = LocalArgs(use_example=False,
                        in_dir="/home/gx-shu/Documents/vpr/train_gray",
                        imgs_ext="png",
                        out_dir="/home/gx-shu/Documents/vpr/GD",
                        max_img_size=512,
                        use_od_example=False,
                        first_n=None,
                        domain="indoor",
                        num_c=32)
    
    with open(f"{myLocalArgs.out_dir}/localargs.pkl", "wb") as f:
        pickle.dump(myLocalArgs, f)
    
    base_tf = tvf.Compose([ # Base image transformations
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda")
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer, desc_facet, device=device)
    
    # create_vocabulary(Path(myLocalArgs.in_dir), 
    #                   myLocalArgs, 
    #                   base_tf, 
    #                   extractor)
    
