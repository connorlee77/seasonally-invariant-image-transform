import glob
import os
import argparse

import numpy as np
import tqdm
from PIL import Image


def createTiles(directory_path, save_dir, crop_width, crop_height, overlap_ratio=0, file_extension='png'):

    files = glob.glob(os.path.join(directory_path, '*.{}'.format(file_extension)))
    for file in tqdm.tqdm(files):
        img = Image.open(file)

        name = os.path.basename(file).split('.')[0]
        h, w, c = np.asarray(img).shape

        filecount = 0
        curr_h = 0
        while curr_h + crop_height <= h:
            curr_w = 0
            while curr_w + crop_width <= w:
                c = curr_w
                r = curr_h

                cropped_on = img.crop((c, r, c + crop_width, r + crop_height))
                fileNo = str(filecount).zfill(6)
                filename_on = '{}_{}.png'.format(name, fileNo)

                cropped_on.save(os.path.join(save_dir, filename_on))
                
                curr_w += crop_width * (1 - overlap_ratio)
                filecount += 1
            curr_h += crop_height * (1 - overlap_ratio)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir')
    parser.add_argument('--save_data_dir')
    parser.add_argument('--file_extension', type=str, default='png')
    parser.add_argument('--overlap_ratio', type=float, default=0.2)
    parser.add_argument('--crop_width', type=int, default=600)
    parser.add_argument('--crop_height', type=int, default=600)

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.save_data_dir, exist_ok=True)
    createTiles(
        directory_path=opt.raw_data_dir, 
        save_dir=opt.save_data_dir, 
        crop_width=opt.crop_width, 
        crop_height=opt.crop_height, 
        overlap_ratio=opt.overlap_ratio,
        file_extension=opt.file_extension
    )
