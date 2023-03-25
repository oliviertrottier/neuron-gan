import re
import torch
import torch.nn as nn
import numpy as np
import math
from configs import config
from collections import OrderedDict

# Import only model classes with *
__all__ = ['Generator_dcgan', 'Discriminator_dcgan',
           'Generator_wgan', 'Discriminator_wgan',
           'Generator_PG', 'Discriminator_PG']

# Useful definitions
latent_dim_default = config.latent_dim  # Size of the latent space in the generator
image_size_default = config.image_size  # Dimension of the training images
N_colors_default = config.N_colors  # Number of colors in images
LeakyReLU_neg_slope_default = config.LeakyReLU_leak  # Leak coefficient of LeakyReLU activation


# Original dcgan features
# Generator: the first 4 layers have the same number of features
# Discriminator: the first 4 layers have the same number of features
# N_gen_features = [1024, 512, 256, 128, 64, 32, 16, 8]
# N_dis_features = [16, 32, 64, 128, 128, 128, 128]

###################### PG-GAN ######################
# Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation.", 2018

# Function to initialize a layer's weight with kaiming draws (aka He initialization)
def kaiming_init(model: nn.Module, neg_slope=LeakyReLU_neg_slope_default):
    torch.nn.init.kaiming_normal_(model.weight, a=neg_slope, mode='fan_in', nonlinearity='leaky_relu')
    if model.bias is not None:
        model.bias.data.zero_()


# Function to remove modules in state dict before loading them
def pop_state_dict_modules(Network, state_dict, Module_prefix, N_layers_to_delete, from_start=True):
    Module_keys = [key for key in state_dict if key.startswith(Module_prefix)]
    Module_keys_ind = [int(re.search('\d', key)[0]) for key in Module_keys]
    N_layers_max = max(Module_keys_ind) + 1
    if N_layers_to_delete == 'all':
        N_layers_to_delete = N_layers_max
    assert N_layers_to_delete <= N_layers_max, 'Cannot remove more than {} layers'.format(N_layers_max)
    if from_start:
        Removed_ind = list(range(0, N_layers_to_delete))
    else:
        Removed_ind = list(range(N_layers_max - N_layers_to_delete, N_layers_max))
    Removed_keys = [k for i, k in enumerate(Module_keys) if Module_keys_ind[i] in Removed_ind]

    # Reconstruct the state dictionary
    New_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in Module_keys:
            if k not in Removed_keys:
                # Change the index in the key if layers are removed from the start
                if from_start:
                    ind_match = re.search('\d', k)
                    k = k[:ind_match.start()] + str(int(ind_match[0]) - N_layers_to_delete) + k[ind_match.end():]
                New_state_dict[k] = v
        else:
            New_state_dict[k] = v
    return New_state_dict


# Rename variables in the state dictionary
def rename_state_dict_modules(state_dict, New_names_dict):
    # Reconstruct the state dictionary
    New_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in New_names_dict:
            k = New_names_dict[k]
        New_state_dict[k] = v
    return New_state_dict


# Module version of the interpolate functional.
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                           align_corners=self.align_corners)

    def extra_repr(self):
        out_str = ''
        if self.size is not None:
            out_str = f'size={self.size}'
        else:
            out_str = f'scale_factor={self.scale_factor}'
        out_str += f', mode={self.mode}'
        if self.align_corners is not None:
            out_str += f', align_corners={self.align_corners}'
        return out_str


# Module to perform pixel normalization
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        # In the reference, epsilon defines the minimum value of the norm squared.
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Normalize the pixels over the feature dimension (dim=1).
        # This normalization is different than the one suggested in the PGGAN reference.

        # Option 1 (This call has memory leak. The expand() function is leaky).
        # return nn.functional.normalize(x, p=2, dim=1, eps=self.epsilon)

        # Option 2
        x_norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

        # Option 3
        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(self.epsilon)
        # x_norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True).clamp_min(self.epsilon)

        # Option 4
        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True) + self.epsilon
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

    def __init__(self, *args, scale_mode='fan_in', act_func=('leaky_relu', LeakyReLU_neg_slope_default), **kwargs):
        # Remove the act_func kwarg before passing them to the parent class __init__
        super().__init__(*args, **kwargs)

        # Initialize weights with kaiming draws and set bias to 0.
        kaiming_init(self)

        # Calculate the constant used to rescale weights.
        # Reference: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." 2015
        self.weight_scale_mode = scale_mode
        if self.weight_scale_mode == 'fan_in':
            n_connections = self.weight.shape[1] * np.prod(self.kernel_size)
        elif self.weight_scale_mode == 'fan_out':
            n_connections = self.weight.shape[0] * np.prod(self.kernel_size)
        else:
            raise ValueError('{} is not a supported mode', self.weight_scale_mode)

        # Determine the gain associated with the given activation function
        if act_func is not None:
            gain = torch.nn.init.calculate_gain(nonlinearity=act_func[0], param=act_func[1])
        else:
            gain = 1
        self.register_buffer('weight_scale', torch.tensor(gain / np.sqrt(n_connections)), persistent=False)

    def forward(self, x):
        return super().forward(self.weight_scale * x)


# Linear layer with scaled weights
class Linear_normalized(nn.Linear):
    """ Linear layer with weight normalization
        Args:
            act_func ((str,float)): Parameters that define the activation function that follows the convolution.
                                    default=('leaky_relu',0.2)
        Reference:
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." 2015
    """

    def __init__(self, *args, scale_mode='fan_in', act_func=('leaky_relu', LeakyReLU_neg_slope_default), **kwargs):
        # Remove the act_func kwarg before passing them to the parent class __init__
        super().__init__(*args, **kwargs)

        # Initialize weights with kaiming draws and set bias to 0.
        kaiming_init(self)

        # Calculate the number of connections.
        self.weight_scale_mode = scale_mode
        if self.weight_scale_mode == 'fan_in':
            n_connections = self.weight.shape[1]
        elif self.weight_scale_mode == 'fan_out':
            n_connections = self.weight.shape[0]
        else:
            raise ValueError('{} is not a supported mode', self.weight_scale_mode)

        # Determine the gain associated with the given activation function
        if act_func is not None:
            gain = torch.nn.init.calculate_gain(nonlinearity=act_func[0], param=act_func[1])
        else:
            gain = 1
        self.register_buffer('weight_scale', torch.tensor(gain / math.sqrt(n_connections)), persistent=False)

    def forward(self, x):
        return super().forward(self.weight_scale * x)


# Convolution block for PG-GAN
class Conv2d_scale_block(nn.Sequential):
    """ Scaling transformation followed by 2D convolution block"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, scale_factor=None,
                 LeakyReLU_neg_slope=LeakyReLU_neg_slope_default):
        super().__init__()

        if scale_factor < 1:
            # Use an average pooling layer to downsample
            self.append(nn.AvgPool2d(kernel_size=int(1 / scale_factor)))
        else:
            # Use interpolate to upsample
            self.append(Interpolate(scale_factor=scale_factor, mode='bilinear'))

        # Add two convolutions followed by activation and pixel norm
        activation_func = ('leaky_relu', LeakyReLU_neg_slope)
        self.append(Conv2d_normalized(in_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      padding_mode='zeros', bias=False, act_func=activation_func))
        self.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        self.append(PixelNorm())
        self.append(Conv2d_normalized(out_channels, out_channels, kernel_size, stride=1, padding=padding,
                                      padding_mode='zeros', bias=False, act_func=activation_func))
        self.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        self.append(PixelNorm())


# Progressively growing generator network
class Generator_PG(nn.Module):
    alpha: torch.Tensor

    def __init__(self, N_features_per_layer: list[int], image_size_init=4,
                 latent_dim=latent_dim_default, LeakyReLU_neg_slope=LeakyReLU_neg_slope_default,
                 N_colors=N_colors_default):

        # Define the total number of resolution scaling
        N_scaling = len(N_features_per_layer) - 1

        super().__init__()
        self.latent_dim = latent_dim
        self.N_features_per_layer = N_features_per_layer
        self.N_layers = 1
        self.N_layers_max = len(self.N_features_per_layer)
        self.N_colors = N_colors
        self.image_size_init = image_size_init
        self.image_size = image_size_init
        self.image_size_max = 2 ** N_scaling * image_size_init
        self.LeakyReLU_neg_slope = LeakyReLU_neg_slope
        self.register_buffer('alpha', torch.tensor(1.0), persistent=False)

        # The initial layer is a fully connected layer.
        self.layers = nn.Sequential()
        # self.layers.append(nn.Linear(latent_dim, self.N_features_per_layer[0] * image_size_init ** 2, bias=False))
        # self.layers.append(nn.Unflatten(dim=1, unflattened_size=(self.N_features_per_layer[0], image_size_init, image_size_init)))

        self.layers.append(Linear_normalized(latent_dim, self.N_features_per_layer[0] * image_size_init ** 2,
                                             bias=False, act_func=('leaky_relu', LeakyReLU_neg_slope)))
        self.layers.append(
            nn.Unflatten(dim=1, unflattened_size=(self.N_features_per_layer[0], image_size_init, image_size_init)))

        # self.layers.append(nn.Unflatten(dim=1, unflattened_size=(latent_dim, 1, 1)))
        # self.layers.append(Conv2d_normalized(latent_dim, self.N_features_per_layer[0],
        #                                      kernel_size=image_size_init, stride=1, padding=image_size_init - 1,
        #                                      padding_mode='zeros', bias=True,
        #                                      act_func=('leaky_relu', LeakyReLU_neg_slope)))

        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        self.layers.append(PixelNorm())
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[0], self.N_features_per_layer[0],
                                             kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                                             bias=False, act_func=('leaky_relu', LeakyReLU_neg_slope)))
        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        self.layers.append(PixelNorm())

        # Initialize all subsequent convolution layers
        self.conv_block_list = nn.ModuleList()
        for i in range(len(self.N_features_per_layer) - 1):
            in_channels = self.N_features_per_layer[i]
            out_channels = self.N_features_per_layer[i + 1]
            self.conv_block_list.append(
                Conv2d_scale_block(in_channels=in_channels, out_channels=out_channels, scale_factor=2, kernel_size=3))

        # Initialize all layers that transform from the feature space to color space
        self.ToIm_list = nn.ModuleList()
        for i in range(len(self.N_features_per_layer)):
            self.ToIm_list.append(ToImage(self.N_features_per_layer[i], N_colors))

        # Initialize the layer that converts from feature space to color space
        self.ToIm = self.ToIm_list.pop(0)

        # Define the upsampling method
        self.upsample = Interpolate(scale_factor=2, mode='bilinear')

        # Define attributes that will be saved to reconstruct the model. Save all attributes except modules.
        not_saved_attrs = ['layers', 'ToIm_list', 'ToIm', 'conv_block_list']
        saved_attrs = set(self.__dict__.keys()) - set(self.__class__.mro()[1].__dict__.keys())
        saved_attrs = sorted([attr for attr in saved_attrs if not attr.startswith('_') and attr not in not_saved_attrs])
        saved_attrs.append('alpha')
        self.saved_attrs = saved_attrs

    def forward(self, x):
        if self.alpha < 1:
            # The network is in the middle of a transition.
            x = self.layers(x)
            im_start = self.upsample(self.ToIm(x))
            im_end = self.ToIm_list[0](self.conv_block_list[0](x))
            x = im_start + self.alpha * (im_end - im_start)
            return x
        else:
            return self.ToIm(self.layers(x))

    def increase_resolution(self):
        # Ensure that there is no ongoing transition.
        assert self.alpha >= 1, 'The previous transition has not ended.'

        # Start a resolution transition.
        self.alpha = 0 * self.alpha
        self.N_layers += 1
        self.image_size *= 2

        # Check that the image size has not exceeded the maximum.
        assert self.image_size <= self.image_size_max, (
            f'The image size ({self.image_size}) is greater than the maximum ({self.image_size_max})')

    def advance_transition(self, alpha_step=0.1):
        # Increase alpha by the given step to advance the resolution transition
        self.alpha += alpha_step
        if self.alpha >= 1.0:
            # The transition has ended.
            # Concatenate the last conv block to the core layers.
            self.layers.append(self.conv_block_list.pop(0))

            # Overwrite the new ToImg layer from the list.
            self.ToIm = self.ToIm_list.pop(0)

    # Method to increase the resolution to a given input size.
    def set_resolution(self, res: int, alpha=1.0):
        assert res % self.image_size == 0, 'The resolution must be divisible by {}'.format(self.image_size)
        assert math.log2(res / self.image_size).is_integer(), (
            f'{res} cannot be attained by multiplying the initial resolution ({self.image_size}) by a power of 2.')
        assert res <= self.image_size_max, 'The resolution must be smaller than {}'.format(self.image_size_max)

        # Restructure the layers to obtain the given resolution.
        while self.image_size != res:
            self.increase_resolution()
            if self.image_size == res:
                self.advance_transition(alpha)
            else:
                self.advance_transition(1.0)

    @classmethod
    def from_state_dict(cls, filename, device=torch.device('cpu'), verbose=True):
        # Load the dict with saved vars.
        Saved_dict = torch.load(filename, map_location=device)
        Saved_attrs = Saved_dict['Generator_attrs']

        # Initialize the object with the constructor attributes
        construct_attrs_name = ['N_features_per_layer', 'image_size_init', 'LeakyReLU_neg_slope', 'N_colors']
        construct_attrs = {k: v for k, v in Saved_attrs.items() if k in construct_attrs_name}
        obj = cls(**construct_attrs)

        # Remove attributes that should not be loaded
        Saved_attrs = {k: v for k, v in Saved_attrs.items() if k in obj.saved_attrs}

        # Adjust the resolution of the network
        obj.set_resolution(Saved_attrs['image_size'], Saved_attrs['alpha'])

        # Check if the state dict is in an old format. If yes, remove older params before loading the state dict.
        State_dict = Saved_dict['Generator_state']
        ToIm_patt = re.compile('(?<=ToIm_list.)\d')
        ToIm_list_params_name = [key for key in State_dict if ToIm_patt.search(key)]
        if ToIm_list_params_name:
            N_ToIm_modules = max([int(ToIm_patt.findall(key)[0]) for key in ToIm_list_params_name]) + 1
        else:
            N_ToIm_modules = 0
        Conv_block_patt = re.compile('(?<=conv_block_list.)\d')
        conv_block_list_params_name = [key for key in State_dict if Conv_block_patt.search(key)]
        if conv_block_list_params_name:
            N_conv_block_list = max([int(Conv_block_patt.findall(key)[0]) for key in conv_block_list_params_name]) + 1
        else:
            N_conv_block_list = 0
        if N_ToIm_modules > len(obj.ToIm_list):
            # Warn user
            if verbose:
                print('Warning! Loaded state dict in old format. Keys will be removed to match the new format.')

            # Remove modules in the loaded state dict
            State_dict = pop_state_dict_modules(obj, State_dict, 'ToIm_list', N_ToIm_modules - len(obj.ToIm_list),
                                                from_start=True)
            State_dict = pop_state_dict_modules(obj, State_dict, 'conv_block_list',
                                                N_conv_block_list - len(obj.conv_block_list), from_start=True)
            State_dict = pop_state_dict_modules(obj, State_dict, 'ToIm_prev', 'all', from_start=True)
            State_dict = pop_state_dict_modules(obj, State_dict, 'last_conv_block', 'all', from_start=True)

        # Overwrite the network parameters
        obj.load_state_dict(State_dict)

        if verbose:
            print('Loaded training state from {}'.format(filename))

        return obj


# Progressively growing generator network
class Discriminator_PG(nn.Module):
    alpha: torch.Tensor

    def __init__(self, N_features_per_layer: list[int], image_size_init=4,
                 LeakyReLU_neg_slope=LeakyReLU_neg_slope_default, N_colors=N_colors_default):
        # Define the total number of resolution scaling
        N_scaling = len(N_features_per_layer) - 1

        super().__init__()
        self.N_features_per_layer = N_features_per_layer
        self.N_layers = 1
        self.N_layers_max = len(self.N_features_per_layer)
        self.N_colors = N_colors
        self.image_size_init = image_size_init
        self.image_size = image_size_init
        self.image_size_max = 2 ** N_scaling * image_size_init
        self.LeakyReLU_neg_slope = LeakyReLU_neg_slope
        self.register_buffer('alpha', torch.tensor(1.0), persistent=True)

        # The final layer has two convolutions followed by a fully connected layer.
        self.layers = nn.Sequential()
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], self.N_features_per_layer[-1],
                                             kernel_size=3, stride=1, padding=1, padding_mode='zeros',
                                             act_func=('leaky_relu', LeakyReLU_neg_slope)))
        self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        self.layers.append(PixelNorm())

        ## Option 1 (using fully-connected at the end)
        # self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], self.N_features_per_layer[-1],
        #                                      (image_size_init, image_size_init), stride=1, padding=0,
        #                                      act_func=('leaky_relu', LeakyReLU_neg_slope)))
        # self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        # self.layers.append(PixelNorm())
        # self.layers.append(nn.Flatten())
        # self.layers.append(Linear_normalized(self.N_features_per_layer[-1], 1, act_func=None))

        ## Option 2 (using only convolutions)
        self.layers.append(Conv2d_normalized(self.N_features_per_layer[-1], 1,
                                             (image_size_init, image_size_init), stride=1, padding=0,
                                             act_func=('leaky_relu', LeakyReLU_neg_slope)))
        # self.layers.append(nn.LeakyReLU(negative_slope=LeakyReLU_neg_slope))
        # self.layers.append(PixelNorm())
        self.layers.append(nn.Flatten())

        # Initialize all subsequent convolution layers
        self.conv_block_list = nn.ModuleList()
        for i in range(len(self.N_features_per_layer) - 1):
            in_channels = self.N_features_per_layer[i]
            out_channels = self.N_features_per_layer[i + 1]
            self.conv_block_list.append(
                Conv2d_scale_block(in_channels=in_channels, out_channels=out_channels, scale_factor=0.5, kernel_size=3))

        # Initialize all layers that transform from the color space to feature space
        self.FromIm_list = nn.ModuleList()
        for i in range(len(self.N_features_per_layer)):
            self.FromIm_list.append(FromImage(N_colors, self.N_features_per_layer[i]))

        # Initialize the layer that converts from feature space to color space
        self.FromIm = self.FromIm_list.pop(-1)
        self.downsample = Interpolate(scale_factor=0.5, mode='bilinear')

        # Define attributes that will be saved to reconstruct the model. Save all attributes except modules.
        not_saved_attrs = ['layers', 'FromIm_list', 'FromIm', 'conv_block_list']
        saved_attrs = set(self.__dict__.keys()) - set(self.__class__.mro()[1].__dict__.keys())
        saved_attrs = sorted([attr for attr in saved_attrs if not attr.startswith('_') and attr not in not_saved_attrs])
        saved_attrs.append('alpha')
        self.saved_attrs = saved_attrs

    def forward(self, x):
        if self.alpha < 1:
            # The network is in the middle of a transition.
            y_start = self.FromIm(self.downsample(x))
            y_end = self.conv_block_list[-1](self.FromIm_list[-1](x))
            y = y_start + self.alpha * (y_end - y_start)
            return self.layers(y)
        else:
            return self.layers(self.FromIm(x))

    def increase_resolution(self):
        # Ensure that there is no ongoing transition.
        assert self.alpha >= 1, 'The previous transition has not ended.'

        # Start a resolution transition.
        self.alpha = 0 * self.alpha
        self.N_layers += 1
        self.image_size *= 2

        # Check that the image size has not exceeded the maximum.
        assert self.image_size <= self.image_size_max, (
            f'The image size ({self.image_size}) is greater than the maximum ({self.image_size_max})')

    def advance_transition(self, alpha_step=0.1):
        # Increase alpha by the given step to advance the resolution transition
        self.alpha += alpha_step
        if self.alpha >= 1.0:
            # The transition has ended.

            # Append the new convolution at the beginning of the layers
            self.layers.insert(0, self.conv_block_list.pop(-1))

            # Overwrite the FromImg layer with the one from the list
            self.FromIm = self.FromIm_list.pop(-1)

    # Method to increase the resolution to a given input size
    def set_resolution(self, res: int, alpha=1.0):
        assert res % self.image_size == 0, 'The resolution must be divisible by {}'.format(self.image_size)
        assert math.log2(res / self.image_size).is_integer(), (
            f'{res} cannot be attained by multiplying the initial resolution ({self.image_size}) by a power of 2.')
        assert res <= self.image_size_max, 'The resolution must be smaller than {}'.format(self.image_size_max)

        # Increase the resolution until the given resolution is obtained
        while self.image_size < res:
            self.increase_resolution()
            if self.image_size == res:
                self.advance_transition(alpha)
            else:
                self.advance_transition(1.0)

    @classmethod
    def from_state_dict(cls, filename, device=torch.device('cpu'), verbose=True):
        # Load the dict with saved vars
        Saved_dict = torch.load(filename, map_location=device)
        Saved_attrs = Saved_dict['Discriminator_attrs']

        # Initialize the object with the constructor attributes
        construct_attrs_name = ['N_features_per_layer', 'image_size_init', 'LeakyReLU_neg_slope', 'N_colors']
        construct_attrs = {k: v for k, v in Saved_attrs.items() if k in construct_attrs_name}
        obj = cls(**construct_attrs)

        # Remove attributes that should not be loaded
        Saved_attrs = {k: v for k, v in Saved_attrs.items() if k in obj.saved_attrs}

        # Adjust the resolution of the network
        obj.set_resolution(Saved_attrs['image_size'], Saved_attrs['alpha'])

        # Check if the state dict is in an old format. If yes, remove older params before loading the state dict.
        State_dict = Saved_dict['Discriminator_state']
        FromIm_patt = re.compile('(?<=FromIm_list.)\d')
        FromIm_list_params_name = [key for key in State_dict if FromIm_patt.search(key)]
        if FromIm_list_params_name:
            N_FromIm_modules = max([int(FromIm_patt.findall(key)[0]) for key in FromIm_list_params_name]) + 1
        else:
            N_FromIm_modules = 0
        Conv_block_patt = re.compile('(?<=conv_block_list.)\d')
        conv_block_list_params_name = [key for key in State_dict if Conv_block_patt.search(key)]
        if FromIm_list_params_name:
            N_conv_block_list = max([int(Conv_block_patt.findall(key)[0]) for key in conv_block_list_params_name]) + 1
        else:
            N_conv_block_list = 0
        if N_FromIm_modules > len(obj.FromIm_list):
            # Warn user
            if verbose:
                print('Warning! Loaded state dict in old format. Keys will be removed to match the new format.')

            # Remove modules in state dict
            State_dict = pop_state_dict_modules(obj, State_dict, 'FromIm_list', N_FromIm_modules - len(obj.FromIm_list),
                                                from_start=False)
            State_dict = pop_state_dict_modules(obj, State_dict, 'conv_block_list',
                                                N_conv_block_list - len(obj.conv_block_list), from_start=False)
            State_dict = pop_state_dict_modules(obj, State_dict, 'FromIm_prev', 'all', from_start=False)
            State_dict = pop_state_dict_modules(obj, State_dict, 'first_conv_block', 'all', from_start=False)

        # Overwrite the network parameters
        obj.load_state_dict(State_dict)

        if verbose:
            print('Loaded training state from {}'.format(filename))

        return obj


###################### Vanilla GAN ######################

# Generator network
class Generator_dcgan(nn.Module):

    def __init__(self, N_features, latent_dim=latent_dim_default, N_colors=N_colors_default):
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
    def __init__(self, N_features, N_colors=N_colors_default):
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

    def __init__(self, N_features, latent_dim=latent_dim_default, image_size=image_size_default,
                 N_colors=N_colors_default):
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
    def __init__(self, N_features, image_size=image_size_default, N_colors=N_colors_default):
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
