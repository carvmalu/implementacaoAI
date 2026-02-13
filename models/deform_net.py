'''
The shape network deforms the reference shape to represent
the shape of an individual head. The network takes the geometry latent
code z_geo fo an object i, and a query point x as input and produces the output,
f_s_O(x, z_geo) = & E R^3 , (1)

which is a displacement vector that deforms the query point
to the reference shape. Thus, the signed distance value at any
point x of the object i with geometry latent z_geo is
s(x, z_geo) = f_r_O(x+ &), (2)

where s(.) is the scalar signed distance value, and & is the deformation from eq.1
Architeture specifictions:
- 8 fully conected layers
- 1024 hidden units per layer
- ReLU non-linearity after every layer except output
- input: query point x (3D) + geometry latent code z_geo
- output: 3D displacment vector &

Latent code structure:
z_geo = (z_geoId, z_geoEx, z_geoH)
- z_geoId: identity code (58 identities in training)
- z_geoEx: Expression code (10 expressions)
- z_geoH: Hairstyle code (4 styles: short, long, cap1, cap2)
'''

from tkinter import NO
import torch
import torch.nn as nn
import torch.nn.funcitional as F
import numpy as np
from models.ref_net import PositionalEnconding

class DeformNet(nn.module):
    '''
    Shape Deformation Network (DeformNet)

    this network learns to deform the reference shape to match individual head scans.
    Given a query point x in 3D space and a geometry latent code z_geo, it predicts a displacement
    vector & that maps x to the corresponding point in the referece space.
    Instead of directly predicting SDF values we predict deformations to a canonical reference space
    which enables:
    1. learning dense correspondeces between shapes
    2. disentanglement of geometry and color
    3. efficient representation of large deformations (especially hair) 
    '''

    def __init__(self,
                 input_dim=3, # query poiny dimension (x,y,z)
                 latent_dim_identity= 128, # z_geoId dimension
                 latent_dim_expression= 32, # z_geoEx dimension
                 latent_dim_hairstyle= 16, # z_geoH dimension
                 hidden_dim= 1024, # hidden layer dimension 
                 num_layers = 8, # number of fc layers
                 num_enconding_frequencies = 10, # Postional enconding frequencies
                 output_dim=3): # output: 3D displacement vector
        '''
        Args:
            input_dim: Raw input dimension (3 for x,y,z)
            latent_dim_identity: Dimension of indentity latent code
            latent_dim_expression: Dimension of expression latent code
            latent_dim_hairstyle: Dimension of hairstyle latent code
            hidden_dim: hidden layer dimension
            num_layers: number of FC layers
            num_enconding_frequencies: frequencies for positional encoding
            output_dim: output dimension (3 for displacement vector)
        '''

        super().__init__()
        # store latent dimension for later use
        self.lantent_dim_identity = latent_dim_identity
        self.lantent_dim_expression = latent_dim_expression
        self.lantent_dim_hairstyle = latent_dim_hairstyle
        self.total_latent_dim = latent_dim_expression + latent_dim_hairstyle + latent_dim_identity

        # positional encoding for query point x
        # the inputs to all our networks are encoded using sinusoidal positional encoding
        self.pos_encoding = PositionalEnconding(num_frequencies=num_enconding_frequencies, input_dim=input_dim)

        # input dimension after positional encoding
        encoded_dim = self.pos_encoding.output_dim

        # total input dimension: encoded query point + geometry latent code
        total_input_dim = encoded_dim + self.total_latent_dim

        # build 8-layer MLP
        layers = []

        # input layer: encoded_dim + latent -> hidden_dim
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU)

        # Hidden layers: 6 more layers (total 8, including input e output)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU)

        # output layer: hidden_dim -> output_dim (3D displacement)
        layers.append(nn.Linear(hidden_dim, output_dim))
        # non activation - raw displacement vector

        self.net = nn.Sequential(*layers)

        # initalize weight
        self._init_weights()
    
    def _init_weights(self):
        '''
        Xavier uniform initialization for linear layers.
        '''

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forwad(self, x, z_geo_id, z_geo_ex = None, z_geo_h= None):
        '''
        Foward pass of DeformNet
        Args:
            x: (batch_size, 3) - query points in 3D space
            z_geo_id: (batch_size, latent_dim_identity) - identity latent code
            z_geo_ex: (batch_size, latent_dim_expression) - expression latent code
            z_geo_h: (batch_size, latent_dim_hairstyle) - hairstyle latent code
        Returns:
            delta: (batch_size, 3) - displacement vector & maps to query point to reference space
        '''

        # apply positional encoding to query points
        x_encoded = self.pos_encoding(x) # (batch_size, encoded_dim)

        # concatenate latent codes
        latent_codes = [z_geo_id]

        if z_geo_ex is not None:
            latent_codes.append(z_geo_ex)
        if z_geo_h is not None:
            latent_codes.append(z_geo_h)
        
        z_geo = torch.cat(latent_codes, dim=1) # (batch_size, total_latent_dim)

        # concatenate encoded point and latent code
        network_input = torch.cat([x_encoded, z_geo], dim=1)

        # predict displacement 
        delta = self.net(network_input)

        return delta

    def foward_with_dict(self, x, z_dict):
        '''
        Foward pass using dictionary of latent codes
        Args:        
            x: (batch_size, 3) - query points 
            z_dict: dictionary with keys 'id', 'ex', 'hair'
        Returns:
            delta: (batch_size, 3) - Displacement vector
        '''
        z_id = z_dict.get('id', torch.zeros(x.shape[0], self.latent_dim_identity).to(x.device))
        z_ex = z_dict.get('ex', None)
        z_hair = z_dict.get('hair',  None)

        return self.foward(x, z_id, z_ex, z_hair)
    
class GeometryLatentCodes(nn.Module):
    '''
    Learnable latent codes for geometry

    'The geometry space includes three code vectors for identity, 
    expression, and hairstyle. During training, the number of different
    identity code vectors is equal to the number of training identities, 58.
    The number of different expression vectors is fixed to 10 and hairstyle to 4 for geometry.'
    This module will manage all learnable latent codes for the training set.
    In autodecoder fashion, these codes are optimized jointly with network weights during training
    '''

    def __init__(self, num_identities=58, num_expressions=10, num_hairstyles=4, latent_dim_identity=128, latent_dim_expression=32, latent_dim_hairstyle=16):
        '''
        Docstring for __init__
        num_identities: Number of unique identities in training set
        num_expressions: Number of expression categories (10)
        num_hairstyles: Number of hairstyles categoris(4)
        latent_dim_identity: Dimension of identity latent code
        latent_dim_expression: Dimension of expression latent code
        latent_dim_hairstyle: Dimension of hairstyle latent code
        '''

        super().__init__()

        # learnable latent code embeddings 
        self.identity_codes = nn.Embedding(num_identities, latent_dim_identity)
        self.expression_codes = nn.Embedding(num_expressions, latent_dim_expression)
        self.hairstyle_codes = nn.Embedding(num_hairstyles, latent_dim_hairstyle)

        # initialize with small random values (as per DeepSDF)
        nn.init.normal_(self.identity_codes.weight, mean =0.0, std = 0.01)
        nn.init.normal_(self.expression_codes.weight, mean =0.0, std= 0.01)
        nn.init.nomral_(self.hairstyle_codes.weight, mean= 0.0, std = 0.01)
    
    def foward(self, identity_indices, expression_indices = None, hairstyle_indices = None):
        '''
        Args:
            identity_indices: (batch_size) - identity_indices [0, num_identities -1]
            expression_indices: (batch_size) - expression_indices [0, num_expressions -1]
            hairstyle_indices: (batch_size) - hairstyle_indices [0, num_hairstyle -1]
        Return:
            z_geo_id: (batch_size, latent_dim_identity) 
            z_geo_ex: (batch_size, latent_dim_expression) or None
            z_geo_h: (batch_size, latent_dim_hairstyle) or None
        '''

        z_id = self.identity_codes(identity_indices)
        z_ex = self.expression_codes(expression_indices) if expression_indices is not None else None
        z_hair = self.hairstyle_codes(hairstyle_indices) if hairstyle_indices is not None else None

        return z_id, z_ex, z_hair

    def get_all_codes(self):
        '''
        Returns: all latent codes for visualization/analysis
        '''
        return {
            'identity': self.identity_codes.weight.detach(),
            'expression': self.expression_codes.weight.detach(),
            'hairstyle': self.hairstyle_codes.weight.detach()
        }

class DeformationRegularizer:
    '''
    Deformation regularization loss

    'Further, to ensure regularized deformations to the reference shape,
    we impose a loss on the amount of deformation.
    We use L_def_O(x, z_geo) = ws * || f_s_O(x, z_geo) || 2'

    This prevents excessive/unnecessary deformations and keeps 
    the reference shape as the 'mean' shape.
    '''

    def __init__(self, weight=1.0):
        '''
        Docstring for __init__
            weight: ws in paper - weighting factor for derfomation loss
        '''

        self.weight = weight
    
    def __call__(self, delta):
        '''
        Docstring for __call__

        delta: (batch_size, 3) - predicted displacement vectors

        Returns:
            loss: Scalar tensor - L2 norm of displacements
        '''

        # L2 norm of displacement vectors
        deformation_magnitude = torch.norm(delta, dim=1)
        loss = self.weight * deformation_magnitude.mean()

        return loss

class PairwiseLandmarkLoss(nn.Module):
    '''
    Docstring for PairwiseLandmarkLoss
    Sparse pairwise landmark supervision loss

    'L_lm_O(z_geo_i, z_geo_j) = w_lm E || (x_l_i + &_l_i) - (x_l_j + &_l_j) ||2

    we enforce that the 16 landmarks of each shape i, deform to the same points 
    in the reference space using the pairwise loss

    This is the key loss that enables learning dense correspondences
    with only sparse supervision (16 landmarks per scan)
    '''

    def __init__(self, weight=10.0):
        ''' 
        Args:
            weight: w_lm in paper - weighting factor for landmark loss
        '''
        super().__init__()
        self.weight = weight

    def foward(self, 
               landmarks_i,  # (batch_size, num_landmarks, 3)
               delta_i, # (batch_size, num_landmarks, 3)
               landmarks_j, # (batch_size, num_landmarks, 3)
               delta_j): # (batch_size, num_landmarks, 3)
        '''
        Docstring for foward
        
            landmarks_i: 3D positions of landmarks for shape i
            delta_i: Predicted displacements at landmarks for shape i
            landmarks_j: 3D positions of landmarks for shape j
            delta_j: Predicted displacements at landmarks for shape j
        Returns:
            loss: Scalar tensor - Pairwise landmark correspondece loss
        '''

        # deform landmarks to reference space
        ref_points_i = landmarks_i + delta_i
        ref_points_j = landmarks_j + delta_j

        # L2 distance between corresponding points in reference space
        # they should be identical (same point on reference shape)
        distances = torch.norm(ref_points_i - ref_points_j, dim=1)

        loss = self.weight * distances.mean()

        return loss
    
    def foward_batch(self, batch_data, deform_net, ref_net= None):
        '''
        Compute pairwise loss for a batch of scans.
        Args:
            batch_data: Dictionary containing:
                - 'landmarks': (batch_size, num_landmarks, 3)
                - 'identity_idx': (batch_size,)
                - 'expression_idx': (batch_size,)
                - 'hairstyle': (batch_size,)
            deform_net: DeformNet instance
            ref_net: Optional RefNet (not used in loss, but for debbuging)
        Returns:
            loss: Scalar tensor
        '''

        landmarks = batch_data['landmarks']
        batch_size = landmarks.shape[0]

        # predict displacements for all landmarks in the batch
        all_deltas = []
        for i in range(batch_size):
            # get latent codes for this scan
            z_id = batch_data['z_id'][i:i+1]
            z_ex = batch_data.get('z_ex', None)
            z_hair = batch_data.get('z_hair', None)

            if z_ex is not None:
                z_ex = z_ex[i:i+1]
            if z_hair is not None:
                z_hair = z_hair[i:i+1]

            # predict displacements at landmark positions
            delta = deform_net(landmarks[i:i+1, z_id, z_ex, z_hair])
            all_deltas.append(delta)
        
        deltas = torch.cat(all_deltas, dim=0)

        # compute pairwise loss between all combinations
        total_loss = 0.0
        num_pairs = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                loss = self.foward(landmarks[i:i+1], deltas[i:i+1], landmarks[j:j+1], deltas[j:j+1])
                total_loss += loss
                num_pairs += 1

        if num_pairs > 0:
            total_loss /= num_pairs
        
        return total_loss
    
class EarPointHandler:
    '''
    Special handling for ears covered by hair.

    'for scans with the ears covered by hair, we do not have any ear annotations. 
    We would like to compress the ear region in the reference shape to a single point 
    in the reconstruction for these shapes. Thus, we additionally optimize for one point
    for each ear. We enforce pairwise constraints between the learnable ear points and the annotated
    ear points for the other shapes in the batch using eq. 6'
    This handles the case where ears are not visible (covered by hair)
    by learning a single point that represents the ear location in the reference space
    '''

    def __init__(self, num_ears=2):
        '''
        Docstring for __init__
            num_ears: number of ears (2: left and right)
        '''

        self.num_ears = num_ears

        # learnable ear points in reference space
        # these are optimized during training 
        self.ears = nn.Parameter(torch.zeros(num_ears, 3))
        nn.init.normal_(self.ear_points, mean=0.0, std= 0.1)

    def get_ear_points(self):
        '''
        Returns the learnable ear points
        '''
        return self.ear_points
    
    def compute_ear_loss(self,
                         landmarks_with_ears, # scans with visible ears
                         deltas_with_ears, # their displacements
                         ear_indices):  # indices of ear landmarks
        '''
        Compute loss for ear correspondence.
        Args:
            landmarks_with_ears: (batch_size, num_landmarks, 3)
            deltas_with_ears: (batch_size, num_landmarks, 3)
            ear_indices: list of landmark indices for ears
        Returns:
            loss: scalar tensor
        '''

        loss = 0.0
        for ear_idx in ear_indices:
            # deform ear landmarks to reference space 
            ear_landmarks = landmarks_with_ears[:, ear_idx:ear_idx+1]
            ear_deltas = deltas_with_ears[:, ear_idx:ear_idx+1]
            ref_ear_points = ear_landmarks + ear_deltas

            # these should map to the learnable ear point 
            ear_point = self.ear_points[ear_idx].view(1,1,3)
            loss += torch.norm(ref_ear_points - ear_point, dim =-1).mean()

        return loss


