from PIL import Image
import argparse
import os
import numpy as np
import torch
from datasets import MaskFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glob import glob
import pdb
from model import ModelInpaint
from dcgan import Generator, Discriminator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--generator',
                         type=str,
                         help='Pretrained generator',
                         default='models/gen_9600.pt' )
    parser.add_argument( '--discriminator',
                         type=str,
                         help='Pretrained discriminator',
                         default='models/dis_9600.pt' )
    parser.add_argument( '--imgSize',
                         type=int,
                         default=64 )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=64 )
    parser.add_argument( '--n_size',
                         type=int,
                         default=7,
                         help='size of neighborhood' )
    parser.add_argument( '--blend',
                         action='store_true',
                         default=True,
                         help="Blend predicted image to original image" )
    # These files are on SFU VC servers
    parser.add_argument( '--mask_csv',
                         type=str,
                         default='/home/csa102/gruvi/celebA/mask.csv',
                         help='path to the masked csv file' )
    parser.add_argument( '--mask_root',
                         type=str,
                         default='/home/csa102/gruvi/celebA',
                         help='path to the masked root' )
    parser.add_argument( '--per_iter_step',
                         type=int,
                         default=15,
                         help='number of steps per iteration' )
    args = parser.parse_args()
    return args

def saveimages( corrupted, completed, blended, index ):
    os.makedirs( 'completion', exist_ok=True )
    save_image( corrupted,
                'completion/%d_corrupted.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )
    save_image( completed,
                'completion/%d_completed.png' % index,
                nrow=completed.shape[ 0 ] // 5,
                normalize=True )
    save_image( blended,
                'completion/%d_blended.png' % index,
                nrow=corrupted.shape[ 0 ] // 5,
                normalize=True )

def main():
    args = parse_args()
    use_selfie = True
    # Configure data loader
    celebA_dataset = MaskFaceDataset( args.mask_csv,
                                      args.mask_root,
                                      transform=transforms.Compose( [
                           transforms.Resize( args.imgSize ),
                           transforms.ToTensor(),
                           transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
                       ] ) )
    dataloader = torch.utils.data.DataLoader( celebA_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False )
    m = ModelInpaint( args )



    if(use_selfie):
        print("SELFIE INPAINTING")
        selfie_img = Image.open('selfie.jpg')
        size = 400
        selfie_img = np.asarray(transforms.functional.five_crop(selfie_img, size)[4])
        selfie_img = Image.fromarray(selfie_img)
        selfie_img = selfie_img.resize((64,64))
        selfie_img = np.array(selfie_img) / 255
        selfie_img = selfie_img[np.newaxis, :, :, :]
        mask = np.ones_like(selfie_img)
        mask[:, 19:38, 19:38, :] = 0
        selfie_img = selfie_img * mask

        masks = mask.reshape((1,3,64,64))
        corrupted = selfie_img
        corrupted = torch.from_numpy(corrupted).float().permute(0,3,1,2)
        print(type(corrupted))
        print(corrupted.shape)
        print(type(masks))
        print(masks.shape)
        completed, blended = m.inpaint(corrupted, masks)
        saveimages(corrupted, completed, blended, -1)
        corrupted = blended
    
    else:
        for i, ( imgs, masks ) in enumerate( dataloader ):
            print(i)
            masks = np.stack( ( masks, ) * 3, axis=1 )
            corrupted = imgs * torch.tensor( masks )
            completed, blended = m.inpaint( corrupted, masks )
            saveimages( corrupted, completed, blended, i )
            corrupted = blended
        


    



if __name__ == '__main__':
    main()
