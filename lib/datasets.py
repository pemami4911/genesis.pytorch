import torch
import numpy as np
import h5py
from pathlib import Path
from sacred import Ingredient

ds = Ingredient('dataset')

@ds.config
def cfg():
    data_path = ''  # base directory for data
    h5_path = 'gqn-dataset/rooms_ring_camera/images.hdf5' # dataset name

class StaticHdF5Dataset(torch.utils.data.Dataset):
    """
    Simple Dataset class for reading images from an HdF5 file

    Only supports [3,64,64] images currently
    """

    @ds.capture
    def __init__(self, data_path, h5_path, d_set='train'):
        super(StaticHdF5Dataset, self).__init__()
        self.h5_path = str(Path(data_path, h5_path))
        self.d_set = d_set.lower()
        self.data = h5py.File(self.h5_path,  'r')
        self.data_size, self.frame_size, _, self.num_channels = self.data[self.d_set].shape

    def preprocess(self, img):
        """
        convert img from [H,W,C] to [C,H,W]
        """
        # model only supports RGB 
        if img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        
        if img.shape[0] < 64:
            # pad with zeros
            pad_amount = (64 - img.shape[0])
            pad_amount_even = pad_amount // 2
            if pad_amount % 2 == 0:
                pad_amount_odd = pad_amount_even
            else:
                pad_amount_odd = pad_amount_even + 1
            img = np.pad(img, mode='constant', pad_width=((pad_amount_even, pad_amount_odd), (pad_amount_even, pad_amount_odd), (0,0)), constant_values=0.)
        
        if np.any( img > 1.0 ):
            img /= 255.

        return np.transpose(img, (2,0,1))

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        return self.preprocess(self.data[self.d_set][i].astype('float32'))

