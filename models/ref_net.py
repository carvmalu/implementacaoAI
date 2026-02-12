'''
MODEL: Reference Shape Network (RefNet)
Section: 3.3 training - Network architeture 

"The reference Shape network encodes a single reference shape, such that all individual
head shapes can be obtained by deforming this shape. The output at a query point X  is the signed 
distance value f_r_O(x)  E R. Note that for the reference shape there is no latent code input since only
a single reference shape is learned. This can be seen analogously to the mean (or neutral) shape used
in classical 3DMMs [5]. We use 3 fully conected layers for this network, where each hidden layer has 
dimensionality 512."
'''

from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEnconding(nn.Module):
    '''
    Sinusoidal positional encoding.
    "The inputs to all our networks are encoded using sinusoidal positional encoding [32]"
    Maps 3D coordinates (x,y,z) to a higher dimensional space using sine/cosine functions of 
    different frquencies. This helps the network learn high-frequency details (eyes, nose, mouth)
    reference[32] NeRF: representing scenes as neural radiance fields 
    '''
    def __init__(self, num_frequencies=10, input_dim=3):
        """
        Args:
            num_frequencies: number of frequency bands (L in NeRF paper)
            input_dim: input dimension (3 for x,y,z)
        Returns:
            output dimension: input_dim * (1 + 2 * num_frequencies)
        """

        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.output_dim = input_dim * (1 + 2 * num_frequencies)

        # frequencies 2^0, 2^1, 2^2, ..., 2^(L-1)
        self.register_buffer('frequencies', 2.0 ** torch.arange(num_frequencies))

    def forwad(self, x):
        '''
        Args:
            x: (batch_size, input_dim) - 3D query points
        Returns:
            encoded: (batch_size, output_dim) - Positional encoded features
        '''   
        # x shape: (batch_size, 3)
        batch_size = x.shape[0]

        # reshape for broadcasting: (1, num_frequencies, 1)
        freqs = self.frequencies.view(1, -1, 1)

        # reshape x: (batch_size, 1, input_dim)
        x_expanded = x.view(batch_size, 1, -1)

        # apply frequencies: (batch_size, num_frequencies, input_dim)
        sin_terms = torch.sin(freqs * x_expanded)
        cos_terms = torch.cos(freqs * x_expanded)

        # flatten: (batch_size, num_frequencies, input_dim)
        sin_flat = sin_terms.view(batch_size, -1)
        cos_flat = cos_terms.view(batch_size, -1)

        # concatenate [x, sin(x), cos(x), sin(2x), cos(2x), ...]
        encoded = torch.cat([x, sin_flat, cos_flat], dim = 1)

        return encoded

class RefNet(nn.Module):
    '''
    RefNet has:
    - 3 fully conected layers 
    - 512 hidden units per layer
    - ReLU non-linearity after every layer except output
    - no latent code input (unlike DeformNet/ColorNet)
    - Output: single scalar(signed distance value)
    This network represents the 'mean shape' or 'template' that all individual
    heads are deformed from. In classical 3DMMs, this would be the average face after procrustes 
    alignment.

    Analogy to classical 3DMMs [5]:
    Classical: S = S_mean + E a_i * PC_i 
    i3DMM: S = RefNet(x + &) where & = DeformNet(x, z_geo)
    '''

    def __init__(self, input_dim=3, hidden_dim = 512, num_layers = 3, num_encoding_frequencies = 10, output_dim = 1):
        '''
        Args:
            input_dim: Raw input dimension (3 for x,y,z)
            hidden_dim: Hidden layer dimension (512)
            num_layers: Number of FC layers (3)
            num_encoding_frequencies: Frequencies for positional encoding
            output_dim: output dimension (1 for SDF value) 
        '''
        super().__init__()

        # positional encoding - maps 3D to higher dimension
        self.pos_encoding = PositionalEncoding(num_frequencies = num_encoding_frequencies, input_dim = input_dim)

        # input dimension after positional encoding
        encoded_dim = self.pos_encoding.output_dim

        # build layers: [encoded_dim] -> [hidden_dim] -> [output_dim]
        layers = []

        # input layer: encoded_dim -> hidden_dim 
        layers.append(nn.Linear(encoded_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers: hidden_dim -> hidden_dim (num_layers - 2 times)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output layer:  hidden_dim -> output_layer
        # 'ReLU non-linearity after every layer, except the ouptut layer'
        layers.append(nn.Linear(hidden_dim, output_dim))
        # no activation on output layer - raw SDF value

        self.net = nn.Sequential(*layers)

        # initialize weigths using Xavier uniform initialization
        self._init_weights()

    def __init__weights(self):
        '''
        Initialize weights using Xavier uniform initialization.
        usual for deep networks with ReLU activations 
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def foward(self, x):
        """
        foward pass of RefNet.

        Args:
            x: (batch_size, 3) - query points in 3D space.
            Note: for training these are points + deformation
            s(x, z_geo) = f_r_O(x+ &)
        Returns:
            sdf: (batch_size, 1) - signed distance values
            Negative: inside surface
            Zero: on surface
            Positive: outside surface
        """

        # first ill apply positional encoding
        encoded  = self.pos_encoding(x)

        # pass through MLP
        sdf = self.net(encoded)

        return sdf

    def get_mean_shape_sdf(self, resolution=256, bbox=(-1.0, 1.0)):
        '''
        Utility function: Sample SDF of mean shape on grid. Used for visualization
        of the reference shape.
        Args: 
            resolution: Grid resolution (resolution^3 points)
            bbox: Bounding box coordinates (min, max)
        Returns:
            grid_points: (N, 3) - query points
            sdf_values: (N,1) - SDF values
        '''

        # create a 3D grid
        axis = torch.linspace(bbox[0], bbox[1], resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(axis, axis, axis, indexing='ij')

        # flatten grid
        grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        # compute SDF in batches to avoid OOM 
        batch_size = 10000
        sdf_values = []

        with torch.no_grad():
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                sdf_bacth = self.foward(batch)
                sdf_values.append(sdf_bacth)

        sdf_values = torch.cat(sdf_values, dim =0)

        return grid_points, sdf_values.reshape(resolution, resolution, resolution)

class RefNetPretrainer:
    '''
    Pretraining strategy for RefNet.

    'we initialize RefNet by pretraining it using only one mouth-open training sacn.'

    this class handles the pretraining of RefNet on a single scan beforee
    full joint training with DeformNet and ColorNet
    ''' 

    def __init__(self, ref_net, device= 'cuda'):
        self.ref_net = ref_net.to(device)
        self.device = device

    def pretrain_on_single_scan(self, points, sdf_values, colors=None, num_epochs= 100, learning_rate = 1e-3):
        # sampled points from scan
        # ground truth sdf
        # optional colors

        '''
        pretrain refnet on single scan to learn the mean shape
        Args:
            points: (N,3) - query points
            sdf_values: (N,1) - ground truth sdf values
            colors: (N,3) - optional ground truth colors
            num_epochs: Number of pretraining epochs
            learning_rate: learning rate for adam optimizer
        '''

        print('Pretraining RefNet on single scan')
        print('')
        print(f'Points: {points.shape}')
        print(f'Epochs: {num_epochs}')
        print(f'Learning rate: {learning_rate}')

        # convert to tensors
        points_tensor = torch.FloatTensor(points).to(self.device)
        sdf_tensor = torch.FloatTensor(sdf_values).to(self.device)

        # optimizer
        optimizer = torch.optim.Adam(self.ref_net.parameters(), lr= learning_rate)

        # loss function: L1 loss for SDF
        # paper uses L1 loss for geometry
        criterion = nn.L1Loss()

        # training loop 
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # foward pass
            pred_sdf = self.ref_net(points_tensor)

            # compute loss
            loss = criterion(pred_sdf, sdf_tensor)

            # backward pass
            loss.backward()
            optimizer.step()

            if(epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/ {num_epochs}] - Loss:{loss.item():.6f}')

        print('Pretraining complete')
        return self.ref_net