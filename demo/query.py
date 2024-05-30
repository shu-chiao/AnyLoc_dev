import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import pickle
import numpy as np
import einops as ein
import os
from tqdm.auto import tqdm
from dataclasses import dataclass
from utilities import od_down_links
from typing import Literal, Union
from pathlib import Path
from termcolor import cprint
from argparse import ArgumentParser
# DINOv2 imports
from utilities import DinoV2ExtractFeatures
from utilities import VLAD
from utilities import LocalArgs
from vocabulary import create_vocabulary



def query(img_path: Path,  
          _tf: tvf.Compose,
          _extractor: DinoV2ExtractFeatures,
          _largs: VLAD):
    # Process the query image
    # Load the image
    cprint(f"Loading query image {img_path}", "yellow")
    img = Image.open(img_path)
    img_t = _tf(img).to(torch.device("cuda"))

    if max(img_t.shape[-2:]) > _largs.max_img_size:
        cprint(f"Resizing image {img_path} ...", "yellow")
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
    
    _, h, w = img_t.shape
    h_new, w_new = (h // 14) * 14, (w // 14) * 14
    img_t = tvf.CenterCrop((h_new, w_new))(img_t)[None, ...]
    
    # Extract features
    features = _extractor(img_t)
    # VLAD encoding
    vlad = _largs(features)
    

    # Querying
    scores = {}
    # Querying
    scores = {}
    for k, v in _vocabulary.items():
        scores[k] = np.dot(vlad, v)
    # print(f"Scores: {scores}")
    
    # Sort the scores
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    # print(f"Sorted scores: {sorted_scores}")
    
    return sorted_scores


if __name__ == "__main__":
    parseargs = ArgumentParser(description=__doc__)
    parseargs.add_argument("-im", "--img_path", type=str, required=False, help="Path to the query image")
    parseargs.add_argument("-db", "--db_path", type=str, required=True, help="Path to the directory of global descriptors database")
    args = parseargs.parse_args()

    # if not os.path.exists(args.img_path):
    #     raise FileNotFoundError(f"Image {args.img_path} not found!")
    
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database directory {args.db_path} not found!")
    else:
        with open(f"{args.db_path}", "rb") as f:
            _localargs = pickle.load(f)


    vlad = torch.load(f"{_localargs.out_dir}/{_localargs.vlad_c}")
    assert vlad is not None, "VLAD object not found!"
    
    base_tf = tvf.Compose([ # Base image transformations
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda")
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer, desc_facet, device=device)
    
    query(args.img_path,
          base_tf,  
          extractor,
          vlad)


