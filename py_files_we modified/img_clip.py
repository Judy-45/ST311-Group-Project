# file we use to convert the list of images we are going to convert to CLIP features
import sys
import os
import pickle
import torch
import argparse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from im2wav_utils import *
from Data.meta import ImageHear_paths
from models.hparams import CLIP_VERSION
import clip

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, Any])
    """
    parser = argparse.ArgumentParser(description='collect CLIP')
    parser.add_argument("-save_dir", dest='save_dir', action='store', type=str, default="image_CLIP_split")
    parser.add_argument("-path_list", dest='path_list', action='store', type=str)
    parser.add_argument("-single_pickle", dest='single_pickle', action='store_true', 
                        help="Save each image as a separate pickle (default: False, save all into one)")
    parser.add_argument("-file_name", dest='file_name', action='store', type=str, default="all_images_CLIP",)
    v = vars(parser.parse_args())
    print(v)
    return v

if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        # Load CLIP model
        model, preprocess = clip.load(CLIP_VERSION, device=device)

        if args['path_list'] is None:
            object2paths = ImageHear_paths
        else:
            with open(args['path_list'], 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
                object2paths = {os.path.basename(path).split('.')[0]: [path] for path in image_paths}

        image_objects = list(object2paths.keys())

        # If saving all features into one big pickle
        if not args['single_pickle']:
            big_CLIP = {"image": {}}

        for i, object in enumerate(image_objects):
            images = torch.cat([preprocess(Image.open(path)).unsqueeze(0).to(device) for path in object2paths[object]])
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
            print("image features class: ", object, image_features.shape)

            if args['single_pickle']:
                # Save each object separately
                CLIP_dict = {"image": {object: image_features}}
                save_path = os.path.join(args['save_dir'], f"{object}.pickle")
                with open(save_path, 'wb') as handle:
                    pickle.dump(CLIP_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved {save_path}")
            else:
                # Save into one big dictionary
                big_CLIP["image"][object] = image_features

        if not args['single_pickle']:
            file_name = args['file_name']
            # After all, save the big dictionary
            save_path = os.path.join(args['save_dir'], f"{file_name}.pickle")
            with open(save_path, 'wb') as handle:
                pickle.dump(big_CLIP, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved all features into {save_path}")
