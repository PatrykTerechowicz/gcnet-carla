# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:43:04 2021

@author: Lab_admin
"""
import glob
import re
import cv2
import numpy as np
from skimage import io
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def preprocess_paths(l_list, r_list, d_list):
    '''
        Returns list of paths (l, r, d, frame_id).
    '''
    result = []
    idx_l = 0
    idx_r = 0
    idx_d = 0
    max_idx = min(len(l_list), len(r_list), len(d_list))-1
    while max(idx_l, idx_r, idx_d) <= max_idx:
        fp_l = l_list[idx_l][0]
        fp_r = r_list[idx_r][0]
        fp_d = d_list[idx_d][0]
        id_l = l_list[idx_l][1]
        id_r = r_list[idx_r][1]
        id_d = d_list[idx_d][1]
        move_l = id_l <= id_r and id_l <= id_d
        move_r = id_r <= id_l and id_r <= id_d
        move_d = id_d <= id_l and id_d <= id_r
        if move_l and move_r and move_d:
            result.append((fp_l, fp_r, fp_d, id_l))
        if move_l:
            idx_l += 1
        if move_r:
            idx_r += 1
        if move_d:
            idx_d +=1
    return result
        

class CarlaDataset(Dataset):
    '''
        Dataset class, iterable, returns tuples of (imR, imL, disp, frame_id).
        root_dir should have folders: ,left', ,right', ,depth'.
    '''
    def __init__(self, root_dir=Path.cwd()):
        # load stf
        left_im_dir = root_dir / "left"
        right_im_dir = root_dir / "right"
        depth_im_dir = root_dir / "depth"
        left_paths = CarlaDataset._load_paths_from_dir(left_im_dir)
        right_paths = CarlaDataset._load_paths_from_dir(right_im_dir)
        depth_paths = CarlaDataset._load_paths_from_dir(depth_im_dir)
        # match corresponding frames
        self.data_paths = preprocess_paths(left_paths, right_paths, depth_paths)
    
    
    def __len__(self):
        '''
            returns number of data points
        '''
        return len(self.data_paths)
    
    
    def __getitem__(self, idx):
        dp = self.data_paths[idx]
        imL = io.imread(dp[0])
        imR = io.imread(dp[1])
        imD = io.imread(dp[2])
        return imL, imR, imD, dp[3]
    
    
    def _load_paths_from_dir(dir_path=Path.cwd()):
        '''
            Loads path for every file with extension .png, .bmp, extracts their id(first number in name).
        '''
        file_paths = glob.glob(str(dir_path / "*.png"))
        file_paths.extend(glob.glob(str(dir_path / "*.bmp")))
        #extract frames
        for i, f_p in enumerate(file_paths):
            fn = Path(f_p).stem
            found_id_str = re.findall("[0-9]+", fn)[0]
            frame_id = int(found_id_str)
            file_paths[i] = (f_p, frame_id)
        #sort in respect to frame_id
        file_paths = sorted(file_paths, key=lambda k:k[1])
        return file_paths


def test_1():
    '''
        Tests data_test folder
    '''
    abs_path = Path(r"E:\Patryk_Terechowicz\test_dataset")
    ds = CarlaDataset(root_dir=abs_path)
    ls = ds.data_paths
    ll = []
    for i in ls:
        ll.append(i[3])
    passed = ll == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    if passed:
        print("Test 1 passed")
    else:
        print("Test 1 not passed")


def test_2():
    '''
        Tests on Carla_01_20 DS. Plays video of found sequence.
    '''
    abs_path = Path(r"E:\Patryk_Terechowicz\Carla_20_01")
    ds = CarlaDataset(root_dir=abs_path)
    print("Found %d valid data samples" % len(ds))
    dsize = (256, 256)
    for imL, imR, imD, frame_id in ds:
        imL = cv2.resize(imL, dsize)
        imR = cv2.resize(imR, dsize)
        imD = cv2.resize(imD, dsize)
        im = np.zeros((256, 3*256, 4), dtype=np.uint8)
        im[:,:256,:] = imL
        im[:, 256:2*256,:] = imR
        im[:, 2*256:,:] = np.dstack([imD, imD, imD, imD])
        cv2.imshow("Vid", im)
        cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_1()
    test_2() # will show vid