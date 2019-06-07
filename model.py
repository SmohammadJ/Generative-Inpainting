import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.signal import convolve2d
import external.poissonblending as blending
import numpy as np
import pdb
import math

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ContextLoss( nn.Module ):
    def __init__( self ):
        super( ContextLoss, self ).__init__()
        self.l1_loss = nn.L1Loss()

    def forward( self, generated, corrupted, weight_mask ):
        
        l_c = self.l1_loss(weight_mask * generated, weight_mask * corrupted)
        return l_c

class PriorLoss(nn.Module):
    def __init__(self, discriminator):
        super(PriorLoss, self).__init__()
        self.discriminator = discriminator
        self.lam = 1

    def forward(self, generated):
        validity = self.discriminator(generated)
        ones = torch.ones_like(validity)
        l_p = self.lam * torch.log(ones - validity)
        l_p = torch.mean(l_p)
        return l_p



class ModelInpaint():
    def __init__( self, args ):
        self.batch_size = args.batch_size
        self.z_dim = 100
        self.n_size = args.n_size
        self.per_iter_step = args.per_iter_step

        self.generator = torch.load( args.generator )
        self.generator.eval()
        self.discriminator = torch.load( args.discriminator )
        self.discriminator.eval()

    def create_weight_mask( self, unweighted_masks ):
        kernel = np.ones( ( self.n_size, self.n_size ),
                          dtype=np.float32 )
        kernel = kernel / np.sum( kernel )
        weight_masks = np.zeros( unweighted_masks.shape, dtype=np.float32 )
        for i in range( weight_masks.shape[ 0 ] ):
            for j in range( weight_masks.shape[ 1 ] ):
                weight_masks[ i, j ] = convolve2d( unweighted_masks[ i, j ],
                                                   kernel,
                                                   mode='same',
                                                   boundary='symm' )
        weight_masks = unweighted_masks * ( 1.0 - weight_masks )
        return Tensor( weight_masks )

    def postprocess( self, corrupted, masks, generated ):
        corrupted = corrupted * 0.5 + 0.5
        generated = generated * 0.5 + 0.5
        corrupted = corrupted.permute( 0, 3, 2, 1 ).cpu().numpy()
        processed = generated.permute( 0, 3, 2, 1 ).cpu().detach().numpy()
        masks = np.transpose( masks, axes=( 0, 3, 2, 1 ) )

        for i in range( len( processed ) ):
            processed[ i ] = blending.blend( corrupted[ i ],
                                             processed[ i ],
                                             1 - masks[ i ] )
        processed = torch.tensor( processed ).permute( 0, 3, 2, 1 )
        return ( processed * 2.0 - 1.0 ).cuda()

    def inpaint( self, corrupted, masks ):
        z = torch.tensor( np.float32( np.random.randn( self.batch_size,
                                                       self.z_dim ) ) )
        weight_mask = self.create_weight_mask( masks )
        if cuda:
            z = z.cuda()
            corrupted = corrupted.cuda()
            weight_mask = weight_mask.cuda()

        z_init = z.clone()
        content_loss = ContextLoss()
        prior_loss = PriorLoss(self.discriminator)
        optimizer = torch.optim.Adam([z.requires_grad_()], lr=0.0003)

        for i in range(self.per_iter_step):
            generated = self.generator( z )
            l_c = content_loss(generated, corrupted, weight_mask)
            l_p = prior_loss(generated)
            loss = l_c + l_p

            if(math.isnan(loss) or math.isinf(loss)):
                print("NaN or INF!!!")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(l_c," ",l_p," ",loss)
            
        generated = self.generator( z )
        return generated, self.postprocess( corrupted, masks, generated )
