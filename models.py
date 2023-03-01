import torch
import torch.nn as nn
import numpy as np
from utils import set_saved_attrs
from configs import config

# Import only model classes with *
__all__ = ['Generator_dcgan', 'Discriminator_dcgan',
           'Generator_wgan', 'Discriminator_wgan',
           'Generator_PG', 'Discriminator_PG']

# Useful definitions
latent_dim = config.latent_dim  # Size of the latent space in the generator
image_size = config.image_size  # Dimension of the training images
N_colors = config.N_colors  # Number of colors in images
LeakyReLU_leak = config.LeakyReLU_leak  # Leak coefficient of LeakyReLU activation


# Original dcgan features
# Generator: the first 4 layers have the same number of features
# Discriminator: the first 4 layers have the same number of features
# N_gen_features = [1024, 512, 256, 128, 64, 32, 16, 8]
# N_dis_features = [16, 32, 64, 128, 128, 128, 128]

# N_gen_features = [512, 256, 128, 64, 32, 16, 8, 8]
# N_dis_features = [64, 128, 256, 256, 256, 128, 64]


# N_gen_features = [8, 8, 8, 8, 8, 8, 8, 8]
# N_dis_features = [8, 8, 8, 8, 8, 8, 8]

###################### PG-GAN ######################
# Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation.", 2018

# Function to initialize a layer's weight with kaiming draws (aka He initialization)
def kaiming_init(model: nn.Module):
    torch.nn.init.kaiming_normal_(model.weight, a=LeakyReLU_leak, mode='fan_in', nonlinearity='leaky_relu')
    if model.bias is not None:
        model.bias.data.zero_()


# Module version of the interpolate functional.
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

    def extra_repr(self):
        return 'scale_factor={}, mode={}, align_corners={}'.format(self.scale_factor, self.mode, self.align_corners)


# Module to perform pixel normalization
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        # Average the square values of the pixels over the feature dimension (dim=1)
        x_norm = torch.sqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + self.epsilon)
        return x / x_norm

    def extra_repr(self):
        return 'epsilon={}'.format(self.epsilon)


# 1x1 convolution to project feature maps onto color space
class ToImage(nn.Module):
    def __init__(self, in_channels, N_colors):
        super().__init__()
        self.in_channels = in_channels
        self.N_colors = N_colors

        # Add convolution layer
        self.layers = nn.Sequential()
        conv = nn.Conv2d(in_channels, N_colors, kernel_size=1, stride=1, padding=0, bias=False)
        kaiming_init(conv)
        self.layers.append(conv)

        # Add tanh activation to force pixel values in the range [-1,1]
        self.layers.append(nn.Tanh())

    def forward(self, x):
        return self.layers(x)

    def extra_repr(self):
        return 'in_channels={}, N_colors={}'.format(self.in_channels, self.N_colors)


# 1x1 convolution to project color space onto feature maps
class FromImage(nn.Module):
    def __init__(self, N_colors, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.N_colors = N_colors
        self.conv = nn.Conv2d(N_colors, out_channels, kernel_size=1, stride=1, padding=0)
        kaiming_init(self.conv)

    def forward(self, x):
        return self.conv(x)

    def extra_repr(self):
        return 'N_colors={}, out_channels={}'.format(self.N_colors, self.out_channels)


# Convolution with scaled weights
class Conv2d_normalized(nn.Conv2d):
    """ 2D convolution block with weight normalization
        Args:
            act_func ((str,float)): Parameters that define the activation function that follows the convolution.
                                    default=('leaky_relu',0.2)
    """

    def __init__(self, *args, **kwargs):
        # Remove the act_func kwarg before passing them to the parent class __init__
        act_func = kwargs['act_func']
        del kwargs['act_func']
        super().__init__(*args, **kwargs)

        # Initialize weights with kaiming draws and set bias to 0.
        kaiming_init(self)

        # Initialize weights with truncated normal when the weight scale is used.
        # torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)

        # Calculate the constant used to rescale weights.
        # Reference: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." 2015
        self.weight_scale_mode = 'fan_in'
        # self.weight_scale_mode = 'fan_out'
        if self.weight_scale_mode == 'fan_in':
            n_connections = self.weight.shape[1] * np.prod(self.kernel_size)
        elif self.weight_scale_mode == 'fan_out':
            n_connections = self.weight.shape[0] * np.prod(self.kernel_size)
        else:
            raise ValueError('{} is not a supported mode', self.weight_scale_mode)
        gain = torch.nn.init.calculate_gain(nonlinearity=act_func[0], param=act_func[1])
        self.weight_scale = nn.Parameter(torch.tensor(gain / np.sqrt(n_connections)), requires_grad=False)
        # self.weight_scale = torch.tensor(gain / np.sqrt(n_connections))

    def forward(self, x):
        return super().forward(self.weight_scale * x)


# Linear layer with scaled weights
class Linear_normalized(nn.Linear):
    """ Linear layer with weight normalization
        Args:
            act_func ((str,float)): Parameters that define the activation function that follows the convolution.
                                    default=('leaky_relu',0.2)
    """

    def __init__(self, *args, **kwargs):
        # Remove the act_func kwarg before passing them to the parent class __init__
        act_func = kwargs['act_func']
        del kwargs['act_func']
        super().__init__(*args, **kwargs)

        # Initialize weights with kaiming draws and set bias to 0.
        kaiming_init(self)

        # Initialize weights with truncated normal when the weight scale is used.
        # torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)

        # Calculate the constant used to rescale weights.
        # Reference: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." 2015
        self.weight_scale_mode = 'fan_in'
        # self.weight_scale_mode = 'fan_out'
        if self.weight_scale_mode == 'fan_in':
            n_connections = self.weight.shape[1]
        elif self.weight_scale_mode == 'fan_out':
            n_connections = self.weight.shape[0]
        else:
            raise ValueError('{} is not a supported mode', self.weight_scale_mode)
        if act_func is not None:
            gain = torch.nn.init.calculate_gain(nonlinearity=act_func[0], param=act_func[1])
        else:
            gain = 1
        self.weight_scale = nn.Parameter(torch.tensor(gain / np.sqrt(n_connections)), requires_grad=False)
        # self.weight_scale = torch.tensor(gain / np.sqrt(n_connections))

    def forward(self, x):
        return super().forward(torch.mul(self.weight_scale, x))


# Convolution block for PG-GAN
class Conv2d_scale_block(nn.Sequential):
    """ Scaling transformation followed by 2D convolution block"""

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, padding=1, scale_weights=True):
        super().__init__()
        if scale_factor < 1:
            # Use an average pooling layer to downsample by a fac
            kern_size = int(np.round(1 / scale_factor))
            self.append(nn.AvgPool2d(kernel_size=(kern_size, kern_size)))
        else:
            # Use interpolate to upsample
            self.append(Interpolate(scale_factor=scale_factor, mode='bilinear'))

        # Add two convolutions followed by activation and pixel norm
        self.append(Conv2d_normalized(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      padding_mode='replicate', bias=False, act_func=('leaky_relu', LeakyReLU_leak)))
        self.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        self.append(PixelNorm())
        self.append(Conv2d_normalized(out_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      padding_mode='replicate', bias=False, act_func=('leaky_relu', LeakyReLU_leak)))
        self.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        self.append(PixelNorm())

    def forward(self, x):
        return super().forward(x)


# Progressively growing generator network
class Generator_PG(nn.Module):
    def __init__(self, N_features_per_layer: list[int], image_size_init=4, latent_dim=latent_dim):
        # Ensure that the initial image size is a multiple of the final image size
        err_msg = 'The final image size ({}) is not divisible by the initial image size {}'.format(image_size,
                                                                                                   image_size_init)
        assert image_size % image_size_init == 0, err_msg

        # Ensure that there are enough layers to grow the network to full image resolution
        N_conv_layers = len(N_features_per_layer) - 1
        N_conv_layers_required = int(np.log2(image_size / image_size_init))
        err_msg = '{} convolution layers are required. {} were given.'.format(N_conv_layers_required, N_conv_layers)
        assert N_conv_layers_required == N_conv_layers, err_msg

        super().__init__()
        self.latent_dim = latent_dim
        self.N_features_per_layer = N_features_per_layer
        self.N_layers = 1
        self.image_size_init = image_size_init
        self.image_size = image_size_init
        self.image_size_max = 2 ** N_conv_layers * image_size_init
        self.alpha = 1.0

        # The initial layer is a fully connected layer.
        self.layers = nn.Sequential()
        self.layers.append(Linear_normalized(latent_dim, self.N_features_per_layer[0] * image_size_init ** 2,
                                             bias=False, act_func=('leaky_relu', LeakyReLU_leak)))
        # self.layers.append(nn.Linear(latent_dim, self.N_features_per_layer[0] * image_size_init ** 2, bias=False))
        self.layers.append(
            nn.Unflatten(dim=1, unflattened_size=(self.N_features_per_layer[0], image_size_init, image_size_init)))
        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        self.layers.append(PixelNorm())
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[0], self.N_features_per_layer[0],
                                             (3, 3), stride=1, padding=1, padding_mode='replicate',
                                             bias=False, act_func=('leaky_relu', LeakyReLU_leak)))
        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        self.layers.append(PixelNorm())

        # Initialize all subsequent convolution layers
        self.conv_block_list = nn.ModuleList([])
        for i in range(len(self.N_features_per_layer) - 1):
            in_channels = self.N_features_per_layer[i]
            out_channels = self.N_features_per_layer[i + 1]
            self.conv_block_list.append(Conv2d_scale_block(scale_factor=2, in_channels=in_channels,
                                                           out_channels=out_channels, kernel_size=(3, 3)))

        # Initialize all layers that transform from the feature space to color space
        self.ToIm_list = nn.ModuleList([])
        for i in range(len(self.N_features_per_layer)):
            self.ToIm_list.append(ToImage(self.N_features_per_layer[i], N_colors))

        # Initialize the last block that will be used to perform the soft resolution transition
        self.last_conv_block = None

        # Initialize the layer that converts from feature space to color space
        self.ToIm = self.ToIm_list[self.N_layers - 1]
        self.ToIm_prev = None

        # Define the attributes that will be saved/loaded.
        # Save all attributes except layers.
        saved_attrs = set(self.__dict__.keys()) - set(self.__class__.mro()[1].__dict__.keys())
        saved_attrs = sorted([attr for attr in saved_attrs if not attr.startswith('_') and attr != 'layers'])
        self.saved_attrs = saved_attrs

    def forward(self, x):
        if self.alpha < 1:
            # The network is in the middle of a transition.
            y_prev = self.layers(x)
            y = self.last_conv_block(y_prev)
            return (1 - self.alpha) * self.ToIm_prev(y_prev) + self.alpha * self.ToIm(y)
        else:
            return self.ToIm(self.layers(x))

    def increase_resolution(self, alpha_init=0.0):
        # Ensure that there is no ongoing transition.
        assert self.alpha >= 1, 'The previous transition has not ended.'

        # Start a resolution transition.
        assert 0 <= alpha_init < 1, 'The given alpha ({:.3f}) must be >= 0 and < 1'.format(alpha_init)
        self.alpha = alpha_init
        self.N_layers += 1
        self.image_size *= 2

        # During a transition self.layers contains the previous layers and self.last_block contains the new block
        # that will be added once the transition is complete.
        self.last_conv_block = self.conv_block_list[self.N_layers - 2]

        # Update the layers that transforms to color space.
        self.ToIm_prev = self.ToIm
        self.ToIm = self.ToIm_list[self.N_layers - 1]

        # Since ToIm_prev is at a lower resolution, we need to upsample its output.
        # Redefine self.ToIm_prev by adding an Interpolate layer.
        self.ToIm_prev = nn.Sequential(self.ToIm_prev, Interpolate(scale_factor=2, mode='nearest'))

    def advance_transition(self, alpha_step=0.1):
        # Increase alpha by the given step to advance the resolution transition
        self.alpha += alpha_step
        if self.alpha >= 1.0:
            # The transition has ended.
            # Concatenate the last block to the core layers
            self.layers.extend(self.last_conv_block)
            self.last_conv_block = None

            # Delete the weight and bias in the ToImg layer that is not used anymore.
            # del self.ToIm_prev[0].conv.weight
            # del self.ToIm_prev[0].conv.bias
            self.ToIm_prev = None

    # Method to increase the resolution to a given input size
    def set_resolution(self, res: int, alpha=1.0):
        assert res % self.image_size == 0, 'The resolution must be divisible by {}'.format(self.image_size)
        assert res <= self.image_size_max, 'The resolution must be smaller than {}'.format(self.image_size_max)

        # Restructure the layers to obtain the given resolution
        while self.image_size != res:
            self.increase_resolution()
            if self.image_size == res:
                self.advance_transition(alpha)
            else:
                self.advance_transition(1.0)

    @classmethod
    def from_state_dict(cls, filename, device=torch.device('cpu'), verbose=True):
        # Load the dict with saved vars
        Saved_dict = torch.load(filename, map_location=device)

        # Initialize the object
        N_features_per_layer = Saved_dict['Generator_attrs']['N_features_per_layer']
        image_size_init = Saved_dict['Generator_attrs']['image_size_init']
        latent_dim = Saved_dict['Generator_attrs']['latent_dim']
        obj = cls(N_features_per_layer, image_size_init, latent_dim)

        # Overwrite the network attributes that were saved.
        if 'Generator_attrs' in Saved_dict:
            # Adjust the resolution of the PGGAN networks before overwrite the attributes
            curr_res = Saved_dict['Generator_attrs']['image_size']
            curr_alpha = Saved_dict['Generator_attrs']['alpha']
            obj.set_resolution(curr_res, curr_alpha)
            set_saved_attrs(obj, Saved_dict['Generator_attrs'])

        # Overwrite the network parameters
        obj.load_state_dict(Saved_dict['Generator_state'])

        if verbose:
            print('Loaded training state from {}'.format(filename))

        return obj


# Progressively growing generator network
class Discriminator_PG(nn.Module):
    def __init__(self, N_features_per_layer: list[int], image_size_init=4):
        # Ensure that the initial image size is a multiple of the final image size
        err_msg = 'The final image size ({}) is not divisible by the initial image size {}'.format(image_size,
                                                                                                   image_size_init)
        assert image_size % image_size_init == 0, err_msg

        # Ensure that there are enough layers to grow the network to full image resolution
        N_conv_layers = len(N_features_per_layer) - 1
        N_conv_layers_required = int(np.log2(image_size / image_size_init))
        err_msg = '{} convolution layers are required. {} were given.'.format(N_conv_layers_required, N_conv_layers)
        assert N_conv_layers_required == N_conv_layers, err_msg

        super().__init__()
        self.N_features_per_layer = N_features_per_layer
        self.N_layers = 1
        self.image_size_init = image_size_init
        self.image_size = image_size_init
        self.image_size_max = 2 ** N_conv_layers * image_size_init
        self.alpha = 1.1

        # The final layer has two convolutions followed by a fully connected layer.
        self.layers = nn.Sequential()
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], self.N_features_per_layer[-1],
                                             (3, 3), stride=1, padding=1, padding_mode='replicate',
                                             act_func=('leaky_relu', LeakyReLU_leak)))
        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        self.layers.append(PixelNorm())

        ## Option 1 (using fully-connected at the end)
        # self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], self.N_features_per_layer[-1],
        #                                      (image_size_init, image_size_init), stride=1, padding=0,
        #                                      act_func=('leaky_relu', LeakyReLU_leak)))
        # self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        # self.layers.append(PixelNorm())
        # self.layers.append(nn.Flatten())
        # self.layers.append(Linear_normalized(self.N_features_per_layer[-1], 1, act_func=None))

        ## Option 2 (using only convolutions)
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], 1,
                                             (image_size_init, image_size_init), stride=1, padding=0,
                                             act_func=('leaky_relu', LeakyReLU_leak)))
        # self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_leak))
        # self.layers.append(PixelNorm())
        self.layers.append(nn.Flatten())

        # Initialize all subsequent convolution layers
        self.conv_block_list = nn.ModuleList([])
        for i in range(len(self.N_features_per_layer) - 1):
            in_channels = self.N_features_per_layer[i]
            out_channels = self.N_features_per_layer[i + 1]
            self.conv_block_list.append(Conv2d_scale_block(scale_factor=0.5, in_channels=in_channels,
                                                           out_channels=out_channels, kernel_size=(3, 3)))

        # Initialize all layers that transform from the color space to feature space
        self.FromIm_list = nn.ModuleList([])
        for i in range(len(self.N_features_per_layer)):
            self.FromIm_list.append(FromImage(N_colors, self.N_features_per_layer[i]))

        # Initialize the first block that will be used to perform the soft resolution transition
        self.first_conv_block = None

        # Initialize the layer that converts from feature space to color space
        self.FromIm = self.FromIm_list[-self.N_layers]
        self.FromIm_prev = None

        # Define the attributes that will be saved/loaded.
        # Save all attributes except 'layers'.
        saved_attrs = set(self.__dict__.keys()) - set(self.__class__.mro()[1].__dict__.keys())
        saved_attrs = sorted([attr for attr in saved_attrs if not attr.startswith('_') and attr != 'layers'])
        self.saved_attrs = saved_attrs

    def forward(self, x):
        if self.alpha < 1:
            # The network is in the middle of a transition.
            y_prev = self.FromIm_prev(x)
            y = self.FromIm(x)
            return (1 - self.alpha) * self.layers(y_prev) + self.alpha * self.layers(self.first_conv_block(y))
        else:
            return self.layers(self.FromIm(x))

    def increase_resolution(self, alpha_init=0.0):
        # Ensure that there is no ongoing transition.
        assert self.alpha > 1, 'The previous transition has not ended.'

        # Start a resolution transition.
        assert 0 <= alpha_init < 1, 'The given alpha ({:.3f}) must be >= 0 and < 1'.format(alpha_init)
        self.alpha = alpha_init
        self.N_layers += 1
        self.image_size *= 2

        # During a transition self.layers contains the previous layers and self.first_block contains the new block
        # that will be added once the transition is complete.
        self.first_conv_block = self.conv_block_list[-self.N_layers + 1]

        # Update the layers that transform from color space.
        self.FromIm_prev = self.FromIm
        self.FromIm = self.FromIm_list[-self.N_layers]

        # Since FromIm_prev is at a lower resolution, we need to downsample its input.
        # Redefine self.FromIm_prev by adding an Interpolate layer.
        self.FromIm_prev = nn.Sequential(Interpolate(scale_factor=0.5, mode='nearest'), self.FromIm_prev)

    def advance_transition(self, alpha_step=0.1):
        # Increase alpha by the given step to advance the resolution transition
        self.alpha += alpha_step
        if self.alpha >= 1.0:
            # The transition has ended.
            # Concatenate the core layers to the first block and redefine the layers
            layers_new = nn.Sequential()
            layers_new.extend(self.first_conv_block)
            layers_new.extend(self.layers)
            self.layers = layers_new
            self.first_conv_block = None

            # Delete the weight and bias in the ToImg layer that is not used anymore.
            # del self.FromIm_prev[1].conv.weight
            # del self.FromIm_prev[1].conv.bias
            self.FromIm_prev = None

    # Method to increase the resolution to a given input size
    def set_resolution(self, res: int, alpha=1.0):
        assert res % self.image_size == 0, 'The resolution must be divisible by {}'.format(self.image_size)
        assert res <= self.image_size_max, 'The resolution must be smaller than {}'.format(self.image_size_max)

        # Restructure the layers to obtain the given resolution
        while self.image_size != res:
            self.increase_resolution(0.1)
            if self.image_size == res:
                self.advance_transition(alpha)
            else:
                self.advance_transition(1.0)

    @classmethod
    def from_state_dict(cls, filename, device=torch.device('cpu'), verbose=True):
        # Load the dict with saved vars
        Saved_dict = torch.load(filename, map_location=device)

        # Initialize the object
        N_features_per_layer = Saved_dict['Discriminator_attrs']['N_features_per_layer']
        image_size_init = Saved_dict['Discriminator_attrs']['image_size_init']
        obj = cls(N_features_per_layer, image_size_init)

        # Overwrite the network attributes that were saved.
        if 'Discriminator_attrs' in Saved_dict:
            # Adjust the resolution of the PGGAN networks before overwrite the attributes
            curr_res = Saved_dict['Discriminator_attrs']['image_size']
            curr_alpha = Saved_dict['Discriminator_attrs']['alpha']
            obj.set_resolution(curr_res, curr_alpha)
            set_saved_attrs(obj, Saved_dict['Discriminator_attrs'])

        # Overwrite the network parameters
        obj.load_state_dict(Saved_dict['Discriminator_state'])

        if verbose:
            print('Loaded training state from {}'.format(filename))

        return obj


###################### Vanilla GAN ######################

# Generator network
class Generator_dcgan(nn.Module):

    def __init__(self, N_features, latent_dim=latent_dim):
        super(Generator_dcgan, self).__init__()
        self.latent_dim = latent_dim
        # Add 7 strided convolutional layers to bring the input from 4 x 4 to 512 x 512.
        self.layers = nn.Sequential(
            # Input
            nn.Unflatten(dim=1, unflattened_size=(latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, N_features[0], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(N_features[0]),
            nn.ReLU(True),

            # Conv1
            nn.ConvTranspose2d(N_features[0], N_features[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[1]),
            nn.ReLU(True),

            # Conv2
            nn.ConvTranspose2d(N_features[1], N_features[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[2]),
            nn.ReLU(True),

            # Conv3
            nn.ConvTranspose2d(N_features[2], N_features[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[3]),
            nn.ReLU(True),

            # Conv4
            nn.ConvTranspose2d(N_features[3], N_features[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[4]),
            nn.ReLU(True),

            # Conv5
            nn.ConvTranspose2d(N_features[4], N_features[5], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[5]),
            nn.ReLU(True),

            # Conv6
            nn.ConvTranspose2d(N_features[5], N_features[6], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[6]),
            nn.ReLU(True),

            # Conv7. No batch norm and tanh activation for the final layer.
            nn.ConvTranspose2d(N_features[6], N_colors, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


# Discriminator network with batch normalization post activation
class Discriminator_dcgan(nn.Module):
    def __init__(self, N_features):
        super(Discriminator_dcgan, self).__init__()
        self.layers = nn.Sequential(

            # Input. Input size: N_colors x 512 x 512
            nn.Conv2d(N_colors, N_features[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv1. Input size: N_features[0] x 256 x 256
            nn.Conv2d(N_features[0], N_features[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv2. Input size: N_features[1] x 128 x 128
            nn.Conv2d(N_features[1], N_features[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv3. Input size: N_features[2] x 64 x 64
            nn.Conv2d(N_features[2], N_features[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv4. Input size: N_features[3] x 32 x 32
            nn.Conv2d(N_features[3], N_features[4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[4]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv5. Input size: N_features[4] x 16 x 16
            nn.Conv2d(N_features[4], N_features[5], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[5]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv6. Input size: N_features[5] x 8 x 8
            nn.Conv2d(N_features[5], N_features[6], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(N_features[6]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Conv7. Input size: N_features[6] x 4 x 4
            nn.Conv2d(N_features[6], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)


###################### Wasserstein GAN ######################

# Generator network
# Generator_wgan = Generator_dcgan
class Generator_wgan(nn.Module):

    def __init__(self, N_features, latent_dim=latent_dim):
        super(Generator_wgan, self).__init__()
        self.latent_dim = latent_dim
        # Determine the initial size of the image.
        N_layers = len(N_features)
        Image_size_init = image_size // (2 ** N_layers)

        # The first layer initializes the image from the latent space.
        layers = list()
        layers.append(nn.Linear(latent_dim, N_features[0] * Image_size_init ** 2))
        layers.append(nn.Unflatten(dim=1, unflattened_size=(N_features[0], Image_size_init, Image_size_init)))
        layers.append(nn.BatchNorm2d(N_features[0]))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Add transpose convolution layers to upsample the image to its final size
        for i in range(N_layers - 1):
            layers.append(
                nn.ConvTranspose2d(N_features[i], N_features[i + 1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(N_features[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Add final layer without batch norm and tanh activation.
        layers.append(nn.ConvTranspose2d(N_features[-1], N_colors, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Discriminator network with batch normalization post activation
class Discriminator_wgan(nn.Module):
    def __init__(self, N_features):
        super(Discriminator_wgan, self).__init__()
        Layers = list()
        N_layers = len(N_features)

        # Define the first layer
        # Input. Input size: N_colors x 512 x 512
        Layers.append(nn.Conv2d(N_colors, N_features[0], kernel_size=4, stride=2, padding=1))
        Layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(N_layers - 1):
            Layers.append(nn.Conv2d(N_features[i], N_features[i + 1], kernel_size=4, stride=2, padding=1))
            Layers.append(nn.BatchNorm2d(N_features[i + 1]))
            Layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Add the final layer
        # Each convolution with a stride of 2 decreases the image size by a half.
        # Therefore, the final image size (in each dimension) is Size_init // 2**N_layers
        Image_size_final = image_size // (2 ** N_layers)

        # Flatten and output single number as the critic score
        Layers.append(nn.Flatten())
        Layers.append(nn.Linear(N_features[-1] * Image_size_final ** 2, 1))
        self.layers = nn.Sequential(*Layers)

    def forward(self, x):
        return self.layers(x)
