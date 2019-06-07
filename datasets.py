from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

class FaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        note = self.frame.iloc[ idx, 1 ]
        return ( image, note )

class MaskFaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        mask_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 1 ] )
        mask = np.load( mask_name )
        mask = mask[ 0 :: 2, 0 :: 2 ]
        return ( image, mask )
