import math
import os
import time
import sys
from parse import parse
import datetime
import random, string

import matplotlib
import matplotlib.pyplot as plt
import shelve, pickle
import torch
import torchvision
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import inspect
from collections import OrderedDict
from torchvision.utils import make_grid

import cv2
from PIL import Image


################################################## ARCHITECTURE ##################################################

# Cubic activation function
class CubicActivation(nn.Module):
    def __init__(self):
        """Cubic activation function with learnable parameters
        f(x) = a*x^3 + b*x^2 + c*x"""
        super(CubicActivation, self).__init__()
        self.a = nn.Parameter(torch.tensor([1]))
        self.b = nn.Parameter(torch.tensor([1]))
        self.c = nn.Parameter(torch.tensor([1]))

    def forward(self, x):
        # This operation is not in-place. It could potentially be made in-place with a custom functional that implements
        # its own backward() method.
        return self.a * torch.pow(x, 3) + self.b * torch.pow(x, 2) + self.c * torch.pow(x, 1)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'a={}, b={}, c={}'.format(self.a, self.b, self.c)


################################################## TRAINING ##################################################
# Function to sample a vector from the latent space
def sample_latent_vec(size: tuple, device='cpu', mode='randn'):
    if mode == 'rand':
        return 2 * torch.rand(*size, device=device) - 1
    elif mode == 'randn':
        # Generate uniform numbers on the hypersphere
        # Reference: "Choosing a Point from the Surface of a Sphere", 1972
        # Clamp the values to prevent numerical overflow
        z = torch.randn(*size, device=device).clamp(-5, 5)
        return z / z.norm(2, dim=1, keepdim=True)
    else:
        raise ValueError('{} is not supported'.format(mode))


# Initialize network parameters
def init_weights(m: nn.Module):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# Define a function to determine the number of discriminator training steps
def Calculate_D_steps(Loss_real, Loss_fake, N_min, N_max, Period):
    # Calculate the standard deviation of the discriminator loss over the last iterations
    # When the difference between the real and fake loss is big, the discriminator is well-trained.
    # In this case, there is no need to train the discriminator further as the generator must catch up.
    # The difference is compared to the standard deviation of the real loss over the last Period iterations
    if Loss_real and Loss_fake:
        D_loss_real_std = np.std(Loss_real[-Period:])
        D_loss_diff = np.mean(np.abs(np.subtract(Loss_fake[-Period:], Loss_real[-Period:])))
        N_steps = np.round(D_loss_real_std / D_loss_diff * N_max)
        N_steps = np.min([N_steps, N_max])
        N_steps = np.max([N_steps, N_min])
        N_steps = int(N_steps)
    else:
        # The loss arrays are empty. Use the maximum number of steps.
        N_steps = N_max
    return N_steps


# Functions to get all the saved attributes in a model
def get_saved_attrs(model):
    saved_attrs_dict = {}
    if hasattr(model, 'saved_attrs'):
        for attr in model.saved_attrs:
            saved_attrs_dict[attr] = getattr(model, attr)
    return saved_attrs_dict


def set_saved_attrs(model, saved_attrs_dict):
    for attr_name, attr_value in saved_attrs_dict.items():
        if hasattr(model, attr_name):
            setattr(model, attr_name, attr_value)
        else:
            raise ValueError('{} is not an attribute of {}', attr_name, model)
    return saved_attrs_dict


# Basic class to perform checkpoints of the training and load from previous checkpoints.
class Checkpointer:
    def __init__(self, Generator_net, Discriminator_net, lr: float, filename: str, verbose=True,
                 device=torch.device('cpu'), extra_checkpoint_period=50e3):
        self.Generator_net = Generator_net
        self.Discriminator_net = Discriminator_net
        self.lr = lr
        self.filename = filename
        self.epoch = 0
        self.Loss_real = []
        self.Loss_fake = []
        self.Loss_G = []
        self.Loss_D = []
        self.verbose = verbose
        self.device = device
        self.extra_checkpoint_period = extra_checkpoint_period

    def save_state(self, epoch, Loss_real, Loss_fake, Loss_G, Loss_D):
        self.epoch = epoch
        self.Loss_real = Loss_real
        self.Loss_fake = Loss_fake
        self.Loss_G = Loss_G
        self.Loss_D = Loss_D
        checkpoint_dict = {'epoch': self.epoch,
                           'Generator_state': self.Generator_net.state_dict(),
                           'Generator_attrs': get_saved_attrs(self.Generator_net),
                           'Discriminator_state': self.Discriminator_net.state_dict(),
                           'Discriminator_attrs': get_saved_attrs(self.Discriminator_net),
                           'lr': self.lr,
                           'Loss_real': self.Loss_real,
                           'Loss_fake': self.Loss_fake,
                           'Loss_G': self.Loss_G,
                           'Loss_D': self.Loss_D}
        torch.save(checkpoint_dict, self.filename)

        # Perform the extra checkpoint of the weights
        if epoch % self.extra_checkpoint_period == 0:
            filename_base, ext = os.path.splitext(self.filename)
            filename_temp = filename_base + '_{:d}k'.format(int(epoch / 1000)) + ext
            torch.save(checkpoint_dict, filename_temp)

        if self.verbose:
            print('Training state at epoch {} saved in {}.'.format(self.epoch, self.filename))

    def load_state(self, filename=None):
        if filename is None:
            # Use the filename set in the constructor
            last_checkpoint_dict = torch.load(self.filename, map_location=self.device)
            self.epoch = last_checkpoint_dict['epoch']
            self.Loss_real = last_checkpoint_dict['Loss_real']
            self.Loss_fake = last_checkpoint_dict['Loss_fake']
            self.Loss_G = last_checkpoint_dict['Loss_G']
            self.Loss_D = last_checkpoint_dict['Loss_D']
        else:
            # Use the given filename to load the weights only
            last_checkpoint_dict = torch.load(filename, map_location=self.device)

        # Overwrite the network attributes
        if 'Generator_attrs' in last_checkpoint_dict and 'Discriminator_attrs' in last_checkpoint_dict:
            # Adjust the resolution of the PGGAN networks.
            if hasattr(self.Generator_net, 'set_resolution'):
                curr_res = last_checkpoint_dict['Generator_attrs']['image_size']
                alpha = last_checkpoint_dict['Generator_attrs']['alpha']
                self.Generator_net.set_resolution(curr_res, alpha)
                self.Discriminator_net.set_resolution(curr_res, alpha)

            set_saved_attrs(self.Generator_net, last_checkpoint_dict['Generator_attrs'])
            set_saved_attrs(self.Discriminator_net, last_checkpoint_dict['Discriminator_attrs'])

        # Overwrite the network parameters
        self.Generator_net.load_state_dict(last_checkpoint_dict['Generator_state'], strict=False)
        self.Discriminator_net.load_state_dict(last_checkpoint_dict['Discriminator_state'], strict=False)

        if self.verbose and filename is None:
            print('Loaded training state from {}'.format(self.filename))
        elif self.verbose:
            print('Loaded weights from {}'.format(filename))


################################################## MISCELLANEOUS ##################################################
# Determine if the computer is remote
def is_computer_remote():
    import socket
    return 'Hephaistos' not in socket.gethostname()


# Get user input and validate answer
def ValidatedInput(prompt: str, validate_func, invalid_ans_msg='Invalid answer.'):
    # Ensure the prompt ends with a newline char
    if not prompt.endswith('\n'):
        prompt += '\n'

    while True:
        Ans = input(prompt)
        if validate_func(Ans):
            break
        else:
            print(invalid_ans_msg)
    return Ans


# Function to calculate the norm of all gradients in a neural net
def calculate_grad_norm_hist(model: torch.nn.Module, grad_min=-30, log_scale=True):
    # Calculate all gradient norms
    params_grad_norm = []
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            params_grad_norm.append(p.grad.detach().data.abs_().flatten())

    # Concatenate all data
    if params_grad_norm:
        params_grad_norm = torch.cat(params_grad_norm)
    params_grad_norm = np.array(params_grad_norm.cpu())

    # Apply log scale and apply minimum
    if log_scale:
        params_grad_norm = np.maximum(params_grad_norm, 10 ** grad_min)
        params_grad_norm = np.log10(params_grad_norm)
    else:
        params_grad_norm = np.maximum(params_grad_norm, grad_min)

    # Calculate basis statistics
    if params_grad_norm.size > 0:
        grad_mean = np.mean(params_grad_norm)
        grad_std = np.std(params_grad_norm)
    else:
        grad_mean = np.nan
        grad_std = np.nan
    return params_grad_norm, grad_mean, grad_std


# Print a set of monitored variables
def print_monitored_vals(monitor_dict: OrderedDict):
    strings = []
    for prop_str, prop_value in monitor_dict.items():
        if isinstance(prop_value, int):
            format_str = '{}'
        elif isinstance(prop_value, float):
            format_str = '{: >#7.4g}'
        else:
            format_str = '{}'
        strings.append(prop_str + ': ' + format_str.format(prop_value))

    # Remove the first comma.
    out_str = ', '.join(strings)
    print(out_str)


# Save variables in a given scope
def save_vars(caller_vars, verbose=True):
    # Determine the filename where all the vars will be saved.
    Name_of_caller = inspect.stack()[1][3]
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    rand_ID = ''.join(random.choices(string.ascii_uppercase, k=4))
    filename = f'saved_vars_{Name_of_caller}_{date_str}_{rand_ID}.pkl'
    # filename = 'test.pkl'

    # Save the dump in a particular folder that is under the same folder as the current module
    Saved_vars_dir = os.path.abspath('./saved_vars')
    os.makedirs(Saved_vars_dir, exist_ok=True)
    filepath = os.path.join(Saved_vars_dir, filename)

    Saved_vars = {}
    for var_name, var_val in caller_vars.items():
        try:
            # Test if the variable pickleable
            pickle.dumps(var_val)
            Saved_vars[var_name] = var_val
        except Exception:
            Saved_vars[var_name] = 'ERROR: variable cannot be saved'

    # Save variables
    with open(filepath, 'wb') as f:
        pickle.dump(Saved_vars, f)

    # Saved_vars = shelve.open(filepath)
    # try:
    #     Saved_vars[var_name] = caller_vars[var_name]
    # except Exception:
    #     Saved_vars[var_name] = 'ERROR: variable cannot be saved'
    # Saved_vars.close()

    if verbose:
        print(f'Variables saved in:\n{filepath}')


################################################## TESTING ##################################################
Latent_vecs_memo = {}


def gen_samples(Generator: nn.Module, N_images=16, seed=None):
    # Get the device of the generator from the device of its first parameter
    Generator_device = next(Generator.parameters()).device
    if seed is not None:
        # Check if the needed random vectors have generated before by checking the dict of latent vectors.
        Memo_key = (seed, N_images, Generator.latent_dim)
        if Memo_key in Latent_vecs_memo:
            Z_latent = Latent_vecs_memo[Memo_key]
        else:
            # The needed latent vectors are not memoized. Generate them and save them.
            # The random seed cannot be set on mps device.
            # In this case, generate the random vectors on cpu and move them to the training device.

            # Save the rng state and set the random seed with the input value
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

            # Sample the latent vectors
            Z_latent = sample_latent_vec((N_images, Generator.latent_dim))

            # Restore the rng state after sampling
            torch.set_rng_state(rng_state)

            # Save the vectors in the memo.
            Latent_vecs_memo[Memo_key] = Z_latent

        # Move the latent vector to the training device
        if Z_latent.device != Generator_device:
            Z_latent = Z_latent.to(Generator_device)
    else:
        Z_latent = sample_latent_vec((N_images, Generator.latent_dim), device=Generator_device)
    return Generator(Z_latent), Z_latent


################################################## PLOTTING ##################################################

# Simple progress bar
class ProgressBar:
    def __init__(self, N, update_msg='', complete_msg='Complete'):
        self.width = 20  # Width of the bar
        self.N = N  # Total number of iterations
        self.iter = 0  # Number of completed iterations
        self.progress = 0.0 # Progress fraction
        self.update_msg = update_msg # Message to print while the bar is updating
        self.complete_msg = complete_msg # Message to print when the process has completed
        self.start_time = time.time() # Time when the progress bar was initiated.

        # Initialize the bar
        self.print()

    def step(self, custom_update_msg=''):
        self.iter += 1
        assert self.iter <= self.N, 'The progress bar has exceeded the total number of iterations'
        self.print(custom_update_msg)

        # If the last step has been reached, print final message and bring the cursor to the next line.
        if self.iter == self.N:
            self.print(self.complete_msg)
            print()

    def calculate_remaining_time(self):
        time_elapsed = time.time() - self.start_time
        time_remaining_sec = time_elapsed * (1 / self.progress - 1)

        # Count the number of hours, minutes and seconds that remain.
        time_remaining = [0, 0, 0]
        time_remaining[0] = int(time_remaining_sec // 3600)
        time_remaining[1] = int((time_remaining_sec - 60 * time_remaining[0]) // 60)
        time_remaining[2] = int((time_remaining_sec - 60 * time_remaining[1] - 3600 * time_remaining[0]))
        return time_remaining

    def print(self, custom_update_msg=''):
        # Calculate progress
        self.progress = self.iter / self.N
        progress_perc = 100 * self.progress

        # Calculate the remaining time
        if self.iter > 0:
            time_remaining = self.calculate_remaining_time()
            time_remaining_str = '{0:02d}:{1:02d}:{2:02d}'.format(*time_remaining)
        else:
            time_remaining_str = '--:--:--'

        # Format the bar
        N_complete_tokens = int(self.progress * self.width)
        bar_tokens = N_complete_tokens * '*' + (self.width - N_complete_tokens) * ' '

        # Format the update message.
        if custom_update_msg:
            prefix = custom_update_msg
        elif self.update_msg:
            prefix = self.update_msg
        else:
            prefix = ''
        bar_str = f'{prefix}|{bar_tokens}| {progress_perc:3.0f}%, Time remaining:{time_remaining_str}'

        # Print with carriage return to prepare the cursor for the next print.
        sys.stdout.write("\033[K") # Clear entire line
        print(bar_str, end='\r')


# Plot an image (torch.tensor)
def plot_image(image: torch.tensor):
    PIL_image = torchvision.transforms.ToPILImage()(image)
    plt.clf()
    plt.imshow(PIL_image)
    plt.colorbar()


# Plot a random sample image of a dataset
def plot_sample(dataset: Dataset, ind=None):
    N_samples = len(dataset)
    if ind is None:
        ind = torch.randint(N_samples - 1, (1,)).item()
    elif ind > N_samples - 1:
        raise ValueError('ind must be smaller than {}'.format(N_samples))
    plot_image(dataset[ind])


# Plot a random sample images from a generator
def plot_gen_samples(Generator: nn.Module, N_images=16, seed=None, filename=None):
    # Generator images
    Gen_images, _ = gen_samples(Generator, N_images, seed=seed)
    N_rows = int(np.round(np.sqrt(N_images)))

    # Detach and move images to the cpu before saving them.
    # Not doing so creates a grid of repeated images.
    Gen_images = Gen_images.detach().to(torch.device('cpu'))

    # Upsample the eval images, if they don't the final resolution.
    if Gen_images.size(-1) != Generator.image_size_max:
        size_final = (Generator.image_size_max, Generator.image_size_max)
        Gen_images = nn.functional.interpolate(Gen_images, size=size_final)

    if filename is None:
        # Create the grid
        images_grid = make_grid(Gen_images, nrow=N_rows)
        # Make sure the color channel is last for matplotlib
        plt.imshow(images_grid.permute(1, 2, 0))
    else:
        # Save the grid
        torchvision.utils.save_image(Gen_images, filename, nrow=N_rows, normalize=True)


# Calculate the total number of parameters in a model
def N_params(model: torch.nn.Module):
    return sum([torch.prod(torch.tensor(param.size())).item() for param in model.parameters()])


# Plot the distribution of gradient norms over all parameters
def plot_grad_norm(generator_model: torch.nn, discriminator_model: torch.nn, filename: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # Plot the histogram of the generator gradients
    gen_params_grad_norm, gen_grad_mean, gen_grad_std = calculate_grad_norm_hist(generator_model)
    ax1.hist(gen_params_grad_norm, alpha=0.75)

    ax1.set_title('Generator, $\mu$={:.2}, $\sigma$={:.2}'.format(gen_grad_mean, gen_grad_std))
    ax1.set_xlabel('Parameter gradient norm (Logged)')
    ax1.set_ylabel('Counts')

    # Plot the histogram of the discriminator gradients
    disc_params_grad_norm, disc_grad_mean, disc_grad_std = calculate_grad_norm_hist(discriminator_model)
    ax2.hist(disc_params_grad_norm, alpha=0.75)

    ax2.set_title('Discriminator, $\mu$={:.2}, $\sigma$={:.2}'.format(disc_grad_mean, disc_grad_std))
    ax2.set_xlabel('Parameter gradient norm (Logged)')
    ax2.set_ylabel('Counts')

    # Tighten the layout
    fig.tight_layout()

    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)
        plt.close(fig)


# Plot scores and save it in a figure
def plot_scores(loss_real, loss_fake, filename, G_loss=None, D_loss=None):
    # Use Agg backend if the plot is saved.
    if filename != '':
        matplotlib.use('Agg')

    fig = plt.figure()
    plt.plot(loss_real, label="Real images (<D(x)>_x)")
    plt.plot(loss_fake, label="Fake images (<D(G(z))>_z)")
    if G_loss:
        plt.plot(G_loss, label="Generator")
    if D_loss:
        plt.plot(D_loss, label="Discriminator")
    plt.legend(loc="upper left")
    plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    plt.savefig(filename)
    plt.close(fig)


# Create a video of the training progress from a series of .png image
def make_samples_video(video_filename, frames_dir):
    # Determine the size of the video from one frame
    frames_filename = [f for f in os.listdir(frames_dir) if f.endswith('.png')]

    frame_ex_filename = frames_filename[0]
    frame_ex_filepath = os.path.join(frames_dir, frame_ex_filename)
    frame_ex_img = Image.open(frame_ex_filepath)

    # Sort the frames by the epoch number
    frame_filename_format = '{prefix}_{ID}_{Epoch:d}.{ext}'
    filename_get_epoch = lambda x: parse(frame_filename_format, x).named['Epoch']
    frames_filename.sort(key=filename_get_epoch)

    # Determine the number of frames to show.
    Video_length = 20  # seconds
    frame_rate = 30  # frames/sec
    N_frames_to_show = int(frame_rate * Video_length)

    # Reduce the list of frames.
    if len(frames_filename) > N_frames_to_show:
        ind = np.round(np.linspace(0, len(frames_filename) - 1, N_frames_to_show)).astype(int)
        frames_filename = [f for i, f in enumerate(frames_filename) if i in ind]

    # Timestamp settings
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 4
    thickness = 4  # px
    epoch_str_format = 'epoch:{:5.1f}e+03'
    format_epoch = lambda x: epoch_str_format.format(x / 1000)
    timestamp_size, timestamp_baseline = cv2.getTextSize(format_epoch(100), font, fontscale, thickness)

    # Pad image on top by the height of the timestamp
    top_pad = timestamp_size[1] + timestamp_baseline

    # Initialize the VideoWriter
    video_basename, video_ext = os.path.splitext(video_filename)
    assert video_ext == '.mp4', 'Only .mp4 format is supported'

    video_filepath_api = os.path.join(frames_dir, video_basename + '.avi')
    video_filepath_mp4 = os.path.join(frames_dir, video_filename)
    video_filename_gif = os.path.join(frames_dir, video_basename + '.gif')
    video_size = (frame_ex_img.size[0], frame_ex_img.size[1] + top_pad)
    output_video = cv2.VideoWriter(video_filepath_api, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, video_size)

    # Write .png frames into video
    bar = ProgressBar(len(frames_filename), 'Creating samples video', 'Video created.')
    for frame_filename in frames_filename:
        frame_filepath = os.path.join(frames_dir, frame_filename)
        img = cv2.imread(frame_filepath)

        # Pad image at the top to include timestamp
        img = cv2.copyMakeBorder(img, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Add epoch timestamp
        frame_epoch = filename_get_epoch(frame_filename)
        epoch_str = format_epoch(frame_epoch)

        # Determine the position of the timestamp
        timestamp_size, timestamp_baseline = cv2.getTextSize(epoch_str, font, fontscale, thickness)
        timestamp_pos = [video_size[0], 0]  # Position of the top-right corner
        horizontal_alignment = 'right'
        vertical_alignment = 'top'

        # Determine the position of the bottom left corner based on the chosen alignment.
        if horizontal_alignment == 'left':
            pass
        elif horizontal_alignment == 'right':
            timestamp_pos[0] = timestamp_pos[0] - timestamp_size[0]
        elif horizontal_alignment == 'center':
            timestamp_pos[0] = timestamp_pos[0] - timestamp_size[0] / 2
        else:
            raise ValueError('Horizontal alignment {} is not supported'.format(horizontal_alignment))

        if vertical_alignment == 'bottom':
            pass
        elif vertical_alignment == 'top':
            timestamp_pos[1] = timestamp_pos[1] + timestamp_size[1]
        elif vertical_alignment == 'center':
            timestamp_pos[1] = timestamp_pos[1] + (timestamp_size[1]) / 2
        else:
            raise ValueError('Horizontal alignment {} is not supported'.format(vertical_alignment))

        cv2.putText(img, epoch_str, tuple(timestamp_pos), font, fontscale, color, thickness, cv2.LINE_AA)

        # Write frame
        output_video.write(img)
        bar.step()
    output_video.release()

    # Convert and compress video to .mp4 using ffmpeg
    ffmpeg_cmd = 'ffmpeg -y -loglevel warning -i "{}" "{}"'.format(video_filepath_api, video_filepath_mp4)
    os.system(ffmpeg_cmd)

    # Create a .gif
    ffmpeg_cmd = 'ffmpeg -y -loglevel warning -i "{}" "{}" -filter_complex "[0:v] scale=720:-1"'.format(video_filepath_mp4, video_filename_gif)
    os.system(ffmpeg_cmd)

    # Remove the .avi file
    os.remove(video_filepath_api)
