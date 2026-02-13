'''
The color network learns the color of the query point in  the
reference space. Given a query point x, deformation & from eq. 1, and color
latent vector z_col for the object i, the output is represented as
f_c_O(x+&, z_col) E R^3, which is the color at point x. Note that without
this separation of a reerence space, the ColorNet would  also have to take into
account information about object geometry, thus not being able to disentangle
shape and color. Note that the latent code for colornet, for a given 
identity and hairstyle, does not change with expressions. DeformNet finds the right
colors and geometry for different expressions by achieving dense correspondences.'

Architecture specifications:
    - 9 fully conected layers
    - 1024 hidden unit per layer
    - ReLU non-linearity after every layer except
    - input: deformed point (x + &) + color latent code z_col
    - Output: RGB color (3 values)

Latent code structure:
    z_col = (z_colId, z_colH)
    - z_colId: identity color code (same as geometry identity count: 58)
    - z_colH: Hairstyle color code (3 styles: nocap, cap1, cap2)
    (Note: hairstyle color has 3 codes vs geometry hairstyle has 4 codes)
'''

from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ref_net import PositionalEnconding

class ColorNet(nn.Module):
    '''
    Color network (ColorNet)

    this network learns the RGB color values in the reference space.
    it takes a point in the reference space (after deformation) and
    color latent codes, and predicts the color at that point

    key properties:
        1- operates in reference space (x + &) - not original space 
        2- Separate latent codes from geometry (z_col vs z_geo)
        3- Expression-independent: color doesnt change with expression
        4- Disentangled from geometry - same color code works for different expressions/poses of same person
    '''

    def __init__(self, 
                input_dim=3,               # deformed point dimension (x+&)
                latent_dim_identity=128,   # z_colId dimension
                latent_dim_hairstyle=16,   # z_colH dimension
                hidden_dim= 1024,          # hidden layer dimension
                num_layers= 9,             # number of FC layers
                num_enconding_frequencies=10, # Positional encoding frequencies
                output_dim=3):             # output: RGB color(3 valores)
        '''
        Args:        
            input_dim: Raw input dimension (3 for x, y, z)
            latent_dim_identity: Dimension of identity color latent code
            latent_dim_hairstyle: Dimension of hairstyle color latent code
            hidden_dim: Hidden layer dimension (1024)
            num_layers: Number of fC layers (9)
            num_enconding_frequencies: frequencies for positional encoding
            output_dim: output dimension (3 for RGB color)
        '''

        super().__init__()

        # store latent dimensions
        self.latent_dim_identity = latent_dim_identity
        self.latentt_dim_hairstyle = latent_dim_hairstyle
        self.total_latent_dim = latent_dim_hairstyle + latent_dim_identity

        # Positional encoding for deformed point (x + &)
        # 'The inputs to all our networks are encoded using sinusoidal positional encoding
        self.pos_encoding = PositionalEnconding(num_frequencies=num_enconding_frequencies, input_dim=input_dim)

        # input dimension after positional encoding
        encoded_dim = self.pos_encoding.output_dim

        # total input dimension: encoded deformed point + color latent code
        total_input_dim = encoded_dim + self.total_latent_dim

        # build 9-layer MLP
        layers = []

        # input layer: encoded_dim + latent -> hidden_dim
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU)

        # hidden layeers: 7 more hidden layers 
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU)

        
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        '''
        Xavier uniform initialization for linear layers
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def foward(self, x_deformed, z_col_id, z_col_h=None):
        '''
        Foward pass of ColorNet

        Args:
            x_deformed: (batch_size, 3) - points in reference space (x + &)
            z_col_id: (batch_size, latent_dim_identity) - identity color latent code
            z_col_h: (batch_size, latent_dim_hairstyle) - hairstyle color latent code
        Returns:
            RGB: (batch_size, 3) - RGB color values (tipically in [0,1])
        '''

        # apply positional encoding to deformed points
        x_encoded = self.pos_encoding(x_deformed) #(batch_size, encoded_dim)

        # concatenate latent codes
        latent_codes =  [z_col_id]

        if z_col_h is not None:
            latent_codes.append(z_col_h)

        z_col = torch.cat([x_encoded, z_col], dim= -1) # (batch_size, total_latent_dim)

        # Concatenate encoded point and latent code 
        network_input = torch.cat([x_encoded, z_col], dim=1)

        # predict RGB color 
        rgb = self.net(network_input)

        # optional: Apply sigmoid to ensure [0,1] range
        # not specified in paper, but common for color prediction 
        rgb = torch.sigmoid(rgb)

        return rgb

    def foward_with_dict(self, x_deformed, z_dict):
        '''
        Foward pass using dictionary
        Args:
            x_deformed: (batch_size, 3) - points in rerence space
            z_dict: Dictionary with keys 'id', 'hair'
        Returns:
            rgb: (batch_size, 3) - RGB color values
        ''' 
        z_id = z_dict.get('id', torch.zeros(x_deformed.shape[0], self.latent_dim_identity).to(x_deformed.device))
        z_hair = z_dict.get('hair', None)

        return self.foward(x_deformed, z_id, z_hair)
    
class ColorLatentCodes(nn.Module):
    '''
    Learnable latent codes for color
    'the color space includes two code vectors for identity and hairstyle. For
    all scans of the i-th subject we use the same latent variables for the geometry identity
    and color identity, z_geoId and z_colId. Similarly, for each expression and hairstyle, the same variables z_geoEx, z_geoH, 
    and z_colH are used all across all identities. By doing so, we able to learn disentangled latent variables without imposing 
    any explicit constraints
    Note: Hairstyle color has 3 codes(nocap, cap1, cap2) while geometry hairstyle has 4 codes(short, long, cap1, cap2).
    The caps share the same color codes across different hairstyles.
    '''

    def __init__(self,
                num_identities=58, # same as geometry identites
                num_hairstyles=3,  # 3: nocap, cap1, cap2
                latent_dim_identity=128,
                latent_dim_hairstyle=16):
        '''
        Docstring for __init__
        num_identities: number of unique identities 58
        num_hairstyles: number of hairstyle color categories
        latent_dim_identity: dimension of identity color latent code
        latent_dim_hairstyle: dimension of hairstyle color latent code
        '''
        super().__init__()
        # learnable latent code embeddings
        self.identity_codes = nn.Embedding(num_identities, latent_dim_identity)
        self.hairstyle_codes = nn.Embedding(num_hairstyles, latent_dim_hairstyle)

        # initialize with small random values
        nn.init.normal_(self.identity_codes.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.hairstyle_codes.weight, mean=0.0, std=0.01)

    def foward(self, identity_indices, hairstyle_indices=None):
        '''
        Docstring for foward
            dentity_indices: (batch_size,) - identity indices [0, num_identities-1]
            hairstyle_indices: (batch_size,) - Hairstyle color indices [0, num_hairstyles-1]
        Returns:
            z_col_id: (batch_size, latent_dim_identity)
            z_col_h: (batch_size, latent_dim_hairstyle) or None
        '''

        z_id = self.identity_codes(identity_indices)
        z_hair = self.hairstyle_codes(hairstyle_indices) if hairstyle_indices is not None else None

        return z_id,z_hair
    
    def get_all_codes(self):
        '''
        Returns all latent codes for visualization/ analysis
        '''
        return{
            'identity': self.identity_codes.weight.detach(),
            'hairstyle': self.hairstyle_codes.weight.detach()
        }
    
class ColorLoss(nn.Module):
    '''
    ColorLoss reconstruction losse

    'We use a similar 1 - loss for the color component, i.e. 
    L_col_O(x, z_col) = w_c * || f_c_O(x + &, z_col) - c_gt(x)||1'
    This is the L1 loss between predicted and ground truth colors.
    '''
    def __init__(self, weitght=1.0):
        '''
        Docstring for __init__
            weitght: w_c in paper - weighting factor for color loss
        '''
        super().__init__()
        self.weight = weitght
        self.l1_loss = nn.L1Loss()

    def forwad(self, pred_colors, gt_colors):
        '''
        Docstring for forwad
            pred_colors: (batch_size, 3) - Predicted RGB colors
            gt_colors: (batch_size, 3) - ground truth RGB colors
        Returns:
            loss: Scalar tensor - L1 color loss
        '''
        loss = self.weight * self.l1_loss(pred_colors, gt_colors)
        return loss
    
class TextureMapper:
    '''
    Helper class for mapping colors between scans using correspondences
    'We demonstrate these correspondences in fig.6, where the color is transferred
    from one scan to the other. The correspondences are also used in the applications of segmentation 
    and landmark transfer sec. 4.6'
    This classes uses the learned correspondences (via DeformNet) to transfer textures between different head scans
    '''

    def __init__(self, ref_net, deform_net, color_net):
        '''
        Docstring for __init__
            ref_net: reference shape network
            deform_net: shape deformation network
            color_net: color network
        '''
        self.ref_net = ref_net
        self.deform_net = deform_net
        self.color_net = color_net

    def transfer_texture(self,
                        source_points,     # points on source scan
                        source_z_geo,      # source geometry codes
                        source_z_col,      # source color codes 
                        target_z_geo):     # target geometry codes
        '''
        Transfer texture from source to target scan.

        The key insight: Both scans deform the same reference space.
        so we can:
        1- deform source points ti reference space using source_z_geo
        2- Query color in reference space using source_z_col 
        3- transfer thta color to target by deforming back using target_z_geo
        
        But careful: We need to find corresponding points. The paper 
        uses the reference space as the commom ground
        '''
        with torch.no_grad():
            # step 1: Deform source points to reference space
            # Using source geometry codes
            delta_source = self.deform_net.forward_with_dict(source_points, source_z_geo)
            ref_points = source_points + delta_source

            # step 2: Get colors in reference space using source color codes 
            colors_ref = self.color_net.foward_with_dict(ref_points, source_z_col)

            # step 3: to get colors on target, we need target points that 
            # correspond to the same reference points
            # this requires solving for target points p such that:
            # p + deform_net(p, target_z_geo) = ref_points
            # this is a root-finding problem. For demo, we´ll assume
            # we have corresponding points pre-computed

            return colors_ref
        
    def find_correspondence(self, source_point, source_z_geo, target_z_geo, num_iterations=100):
        '''
        Find point on target scan corresponding to source_point
        solves: find p such that p + &_target(p) = source_point + &_source(source_point)
        Args:        
            source_point: (1,3) - point on source scan
            source_z_geo: source geometry codes
            target_z_geo: target geometry codes
            num_iterations: number of optimization iterations
        Returns:
            target_point: (1,3) - Corresponding point on target scan
        '''
        # initialize with source point
        p = source_point.clone().detach().requires_grad_(True)

        # compute target reference point (source point deformed to reference)
        with torch.no_grad():
            delta_source = self.deform_net.foward_with_dict(source_point, source_z_geo)
            target_ref = source_point + delta_source

        # optimize to find p
        optimizer = torch.optim.Adam([p],lr = 0.01)

        for _ in range(num_iterations):
            optimizer.zero_grad()


            # deform current p to reference space using target codes
            delta_p = self.deform_net.foward_with_dict(p, target_ref)
            p_ref = p + delta_p

            # loss: p_ref should equal target_ref
            loss = torch.norm(p_ref - target_ref)

            loss.backward()
            optimizer.step()

        return p.detach()



