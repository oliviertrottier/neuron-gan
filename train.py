# Basic imports
import os
import time
import sys
import shutil
import argparse
import uuid
import numpy as np
from collections import OrderedDict

# Torch imports
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# Utils imports
from parse import parse
from utils import init_weights, Calculate_D_steps, plot_scores, plot_grad_norm, Checkpointer, ValidatedInput, \
    sample_latent_vec, plot_gen_samples
from configs import config
from loss_functions import D_W_loss, G_W_loss, similarity_loss, D_grad_pen_loss

# Force fork start for multiprocessing to avoid freeze_method() error
torch.multiprocessing.set_start_method('fork', force=True)

# Defaults
ROOT_DIR_DEFAULT = os.path.dirname(__file__)
FILENAME_FORMAT = '{prefix}_{ID}.{ext}'
if torch.cuda.is_available():
    device_default = 'cuda'
elif torch.backends.mps.is_available():
    device_default = 'mps'
else:
    device_default = 'cpu'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='', help='Filename of configurations stored in ./configs')
parser.add_argument('--root_dir', type=str, default=ROOT_DIR_DEFAULT, help='Root directory')
parser.add_argument('--dataset_dir', type=str, default='./data/real_images',
                    help='Dataset directory where real images are stored')
parser.add_argument('--images_dir', type=str, default='./images',
                    help='Directory where sub-directory of sample images is located')
parser.add_argument('--weights_dir', type=str, default='./weights',
                    help='Weights directory where network is saved')
parser.add_argument('--plots_dir', type=str, default='./plots',
                    help='Plots directory')

# WGAN config
parser.add_argument('--wgan', action='store_true', help='Use the Wasserstein loss function and network')
parser.add_argument('--n_critic', type=int, default=5, help='Number of critic learning iterations in wgan training')
parser.add_argument('--adapt_critic', action='store_true', default=False, help='Adapt the number critic training steps')
parser.add_argument('--unroll_steps', type=int, default=0, help='Number of unrolled discriminator steps (Unrolled GAN)')
parser.add_argument('--weights_init', type=str, default='', help='Path to weights dict used to initialize networks')
parser.add_argument('--dis_weights', type=str, default='', help='Path to weights of discriminator')

# PGGAN config
parser.add_argument('--pggan', action='store_true', help='Use the Progressively Growing network')
parser.add_argument('--grad_pen_lambda', type=float, default=0.0,
                    help='weight of the gradient penalty in the loss function')
parser.add_argument('--transit_sch', type=float, default=[50, 100, 150, 200, 250, 300, 350], nargs='*',
                    help='Schedule where a resolution transition starts')
parser.add_argument('--transit_period', type=int, default=None,
                    help='Period at which a resolution transition occurs. Overwrites --transit_sch option')
parser.add_argument('--alpha_step', type=float, default=0.05,
                    help='Increment of alpha parameter during resolution transitions')

# Training
parser.add_argument('--RMSprop', action='store_true', default=False, help='Use RMSprop optimizer')
parser.add_argument('--learning_rate', type=float, default=0.00002, help='SGD learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images loaded per iteration')
parser.add_argument('--N_epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 parameter for Adam optimizer')
parser.add_argument('--sim_loss_lambda', type=float, default=0.0, help='Add similarity loss for generator')
parser.add_argument('--sim_loss_lambda_decay_rate', type=float, default=0.0, help='Decay factor of the similarity loss')
parser.add_argument('--drift_epsilon', type=float, default=0.001, help='Weight of the discriminator drift loss')

# Misc
parser.add_argument('--ID', type=str, default=uuid.uuid4().hex[:4], help='Training ID')
parser.add_argument('--resume', action='store_true', default=False, help='Resume training. --ID is required.')
parser.add_argument('--seed', type=int, default=1, help='Set random seed')
parser.add_argument('--checkpointing_period', type=int, default=100, help='Period at which a checkpoint is performed')
parser.add_argument('--translation', type=float, default=0.0, help='Percentage of translation in data augmentation')
parser.add_argument('--device', type=str, default=device_default, choices=['cpu', 'mps', 'cuda'],
                    help='Device used to train')

parser.add_argument('--N_workers', type=int, default=2, help='Number of workers to load data')
parser.add_argument('--pin_memory', action='store_true', default=False,
                    help='Pin data to device memory in data loader.')
options = parser.parse_args()

# Import the configurations given in the configs file
input_args = [arg[2:] for arg in sys.argv if arg.startswith('-') and arg != '--configs']
if options.configs:
    # Overwrite configs given as input
    overwritten_configs = {arg: getattr(options, arg) for arg in input_args}
    config.import_configs(options.configs, overwritten_configs=overwritten_configs)
else:
    for arg in input_args:
        setattr(config, arg, getattr(options, arg))
    config.define_ID_dependent_configs()
    config.validate_configs()

# Import datasets, models and loss functions. These imports depend on the global configs.
from data.NeuronDataset import NeuronDataset, DatasetIterator
from models import *

# Print configurations
config.print_configs()

# Fix seed
torch.manual_seed(config.seed)

# Ensure the training ID has not been used before by searching the weights folder
if not config.resume:
    Existent_files = [f for f in os.listdir(config.weights_dir) if f.endswith('.pth')]
    IDs_existent = {parse(FILENAME_FORMAT, file).named['ID']: 0 for file in Existent_files}
    ID_old = config.ID
    if config.ID in IDs_existent:
        Ans = ValidatedInput('ID={} already exists. Use a new ID(y/n)?'.format(config.ID), lambda x: x in ['y', 'n'])
        # Ask for new ID.
        if Ans == 'y':
            Ans = ValidatedInput('Type a new ID:', lambda x: x not in IDs_existent, 'ID already exists.')

# Setup device
if config.device == 'cuda':
    device = torch.device('cuda:0')
    cudnn.benchmark = True
elif config.device == 'mps':
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Determine datatype based on the device
if device.type == 'mps':
    # MPS only supports 32 bit floats.
    print('Setting default torch datatype to {} for {} device'.format(torch.float32, device))
    torch.set_default_dtype(torch.float32)
    datatype = torch.float32
else:
    # Otherwise, use the default.
    datatype = torch.get_default_dtype()

######################################## Dataset and dataloader ########################################
dataset = NeuronDataset(directory=config.dataset_dir, augmentations=True, im_translation=config.translation)

if config.image_preprocessing == 'device':
    # Perform the preprocessing (resize, augmentations, etc) on the device
    dataloader = DatasetIterator(dataset, batch_size=config.batch_size, device=device)
else:
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.N_workers, pin_memory=config.pin_memory)
N_real_images = len(dataset)

# Define the number of feature maps in the generator and discriminator
N_gen_features = config.N_gen_features
N_dis_features = config.N_dis_features

# Find the initial and final image size for PGGAN
if config.pggan:
    N_upsamples = len(N_gen_features) - 1
    Image_size_final = dataset.image_size
    Image_size_initial = Image_size_final // (2 ** N_upsamples)

######################################## Networks ########################################
# Generator
if config.wgan:
    Generator_net = Generator_wgan(N_gen_features)
elif config.pggan:
    Generator_net = Generator_PG(N_gen_features, image_size_init=Image_size_initial)
else:
    Generator_net = Generator_dcgan(N_gen_features)
Generator_net.to(device, datatype)

print('Generator Network:')
print(Generator_net)

# Discriminator
if config.wgan:
    Discriminator_net = Discriminator_wgan(N_dis_features)
elif config.pggan:
    Discriminator_net = Discriminator_PG(N_dis_features, image_size_init=Image_size_initial)
else:
    Discriminator_net = Discriminator_dcgan(N_dis_features)
Discriminator_net.to(device, datatype)

# Define the period at which the metrics for the adaptive discriminator learning are updated
Disc_adapt_update_period = 100

print('Discriminator Network:')
print(Discriminator_net)

# Initialize checkpointer and load previous state if training is resumed
Train_state_filename = os.path.join(config.weights_dir,
                                    FILENAME_FORMAT.format(prefix='GenDisc', ID=config.ID, ext='pth'))
checkpoint = Checkpointer(Generator_net, Discriminator_net, config.learning_rate, Train_state_filename,
                          N_epochs=config.N_epochs, device=device, extra_checkpoint_period=1e3)

if config.resume and os.path.exists(Train_state_filename):
    # Load the entire last training state
    checkpoint.load_state()
elif config.weights_init:
    # Initialize only the weights from the input weights_init filename
    checkpoint.load_state(os.path.join(config.weights_dir, config.weights_init))
elif not config.pggan:
    # Initialize the weights of the networks using the init_weights function
    Generator_net.apply(init_weights)
    Discriminator_net.apply(init_weights)

# For PGGAN, adjust the size of the image size output of the dataloader to the current network resolution.
if config.pggan:
    # Check that the discriminator and generator are at the same resolution
    err_msg = 'The generator and discriminator are at different resolution'
    assert Generator_net.image_size == Discriminator_net.image_size, err_msg
    dataset.set_image_size(Generator_net.image_size)

# Pre-training setup (loss, optimizer, input noise for generating fake samples, checkpoints)
if config.RMSprop:
    optimizer_dis = optim.RMSprop(Discriminator_net.parameters(), lr=config.learning_rate)
    optimizer_gen = optim.RMSprop(Generator_net.parameters(), lr=config.learning_rate)
else:
    optimizer_dis = optim.Adam(Discriminator_net.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_gen = optim.Adam(Generator_net.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

######################################## Loss functions ########################################
G_loss = G_W_loss(Generator_net, Discriminator_net)
D_loss = D_W_loss(Generator_net, Discriminator_net, drift_epsilon=config.drift_epsilon)
D_grad_loss = D_grad_pen_loss(Generator_net, Discriminator_net, Lambda=config.grad_pen_lambda)

######################################## Learning rate schedulers ########################################

# Calculate the number of resolution transition that has occured
N_res_transitions = int(np.round(np.log2(Generator_net.image_size / Generator_net.image_size_init)))

# Define the total decay of the learning rate over the course of a learning rate ramp down in each learning phase.
lr_transit_total_decay = 1 / 100

# Calculate the learning rate decay rate in each learning phase (determined by the resolution of the image)
lr_decay_rate = []
transitions_epoch_boundaries = [0] + config.transit_sch + [config.N_epochs]
for i in range(len(transitions_epoch_boundaries) - 1):
    Phase_N_Epochs = transitions_epoch_boundaries[i + 1] - transitions_epoch_boundaries[i]
    # Decay the learning rate in the first half
    lr_decay_rate.append(np.exp(np.log(lr_transit_total_decay) / (Phase_N_Epochs / 2)))


# Function to update the learning rate of an optimizer
def update_lr(optimizer, epoch):
    # If a new transition is starting, reset the learning rate to its initial value
    if epoch in transitions_epoch_boundaries:
        for param in optimizer.param_groups:
            param['lr'] = config.learning_rate
    else:
        # Determine the decay rate of the learning in this current learning phase.
        Phase_ind = sum([epoch > transit_epoch for transit_epoch in config.transit_sch])
        Phase_N_Epochs = transitions_epoch_boundaries[Phase_ind + 1] - transitions_epoch_boundaries[Phase_ind]
        epoch_since_last_transit = epoch - transitions_epoch_boundaries[Phase_ind]
        if epoch_since_last_transit <= Phase_N_Epochs / 2:
            gamma = lr_decay_rate[Phase_ind]

            # Decrease the learning rates.
            for param in optimizer.param_groups:
                param['lr'] = config.learning_rate * (gamma ** epoch_since_last_transit)


# Samples
eval_noise = sample_latent_vec((16, Generator_net.latent_dim), device=device)
training_summary_filename = os.path.join(config.plots_dir, 'Training_summary_{}.png'.format(config.ID))

# Generate a sample of test images with the noise
# eval_filename = os.path.join(config.samples_sub_dir, 'Test_images_{}_{}.png'.format(config.ID, checkpoint.epoch))
# plot_gen_samples(Generator_net, eval_noise=eval_noise, N_images=16, seed=0, filename=eval_filename)

# Initialize timeseries
Score_real_series = checkpoint.Loss_real
Score_fake_series = checkpoint.Loss_fake
G_loss_series = checkpoint.Loss_G
D_loss_series = checkpoint.Loss_D
epoch_init = checkpoint.epoch + 1  # The first epoch starts at 1
if config.N_epochs_session:
    epoch_final = epoch_init + config.N_epochs_session
else:
    epoch_final = config.N_epochs + 1

# Initialize the learning rates of the optimizers
update_lr(optimizer_dis, epoch_init - 1)
update_lr(optimizer_gen, epoch_init - 1)

# Monitoring
Monitored_values = OrderedDict()
Monitoring_period = 10
epoch_stats = {'score_real': 0, 'score_fake': 0, 'D_loss': 0, 'G_loss': 0, 'D_grad_pen': 0, 'G_sim_loss': 0}


# Training functions
def pggan_train():
    # Initialize similarity loss param
    Sim_loss_lambda = config.sim_loss_lambda

    # Initialize local monitoring variables
    Score_real = None
    Score_fake = None
    D_loss_val = torch.tensor(0)
    G_loss_val = torch.tensor(0)
    D_grad_pen = torch.tensor(0)
    G_sim_loss = torch.tensor(0)

    start_time = time.time()

    for epoch in range(epoch_init, epoch_final):
        # Initialize epoch stats
        for stat, val in epoch_stats.items():
            epoch_stats[stat] = 0.0
        lr = optimizer_gen.param_groups[0]['lr']

        # Advance resolution transition, if one is ongoing.
        if Generator_net.alpha < 1 and Discriminator_net.alpha < 1:
            Generator_net.advance_transition(config.alpha_step)
            Discriminator_net.advance_transition(config.alpha_step)
        elif Generator_net.alpha < 1:
            err_msg = 'The networks are not synchronized. Gen_alpha={:.3f}, Disc_alpha={:.3f}'.format(
                Generator_net.alpha, Discriminator_net.alpha)
            raise Exception(err_msg)

        # Start resolution transition if the epoch is in the transition schedule
        if epoch in config.transit_sch:
            Generator_net.increase_resolution()
            Discriminator_net.increase_resolution()

            # Increase the size of the image outputted by the dataloader.
            dataset.set_image_size(Generator_net.image_size)

        # Determine the number of discriminator training steps
        if config.adapt_critic and len(Score_real_series) > Disc_adapt_update_period:
            N_D_steps = Calculate_D_steps(Score_real_series, Score_fake_series, 0, config.n_critic,
                                          Period=Disc_adapt_update_period)
        else:
            N_D_steps = config.n_critic

        # Reduce the similarity loss lambda factor.
        if config.sim_loss_lambda_decay_rate > 0 and Sim_loss_lambda > 0:
            if Sim_loss_lambda > 1e-5:
                Sim_loss_lambda = config.sim_loss_lambda * (1 - config.sim_loss_lambda_decay_rate) ** (epoch - 1)
            else:
                # Set lambda to 0 for the rest of the epoch.
                Sim_loss_lambda = 0

        for i, images in enumerate(dataloader):
            # Move images to the GPU or mps device
            if device.type != images.device.type:
                images = images.to(device)

            # Train discriminator for many iterations
            for j in range(N_D_steps):
                Discriminator_net.zero_grad()
                D_loss_val, Score_real, Score_fake = D_loss(images)

                # Add gradient penalty
                D_grad_pen = D_grad_loss(images)
                D_loss_val += D_grad_pen

                # Backprop
                D_loss_val.backward()
                optimizer_dis.step()

            # If no discriminator steps were taken, calculate the loss for monitoring.
            if N_D_steps == 0:
                D_loss_val, Score_real, Score_fake = D_loss(images)
                D_grad_pen = D_grad_loss(images)
                D_loss_val += D_grad_pen

            # Train generator
            Generator_net.zero_grad()
            G_loss_val, Z_latent = G_loss(images)

            # Add batch similarity loss.
            if Sim_loss_lambda > 0:
                G_sim_loss = similarity_loss(images, Z_latent, Sim_loss_lambda)
                G_loss_val += G_sim_loss

            # Backprop
            G_loss_val.backward()
            optimizer_gen.step()

            # Accumulate the losses of the current epoch
            batch_size_curr = images.size(0)
            epoch_stats['score_real'] += batch_size_curr * Score_real.detach().item()
            epoch_stats['score_fake'] += batch_size_curr * Score_fake.detach().item()
            epoch_stats['D_loss'] += batch_size_curr * D_loss_val.detach().item()  # Only last discriminator iteration
            epoch_stats['G_loss'] += batch_size_curr * G_loss_val.detach().item()
            epoch_stats['D_grad_pen'] += batch_size_curr * D_grad_pen.detach().item()
            epoch_stats['G_sim_loss'] += batch_size_curr * G_sim_loss.detach().item()

        # Normalize the epoch stats for the entire dataset
        for stat, val in epoch_stats.items():
            epoch_stats[stat] /= len(dataset)

        # Monitoring
        if epoch % Monitoring_period == 0:
            Monitored_values['Epoch'] = '{}'.format(epoch)
            N_completed_epochs = epoch - epoch_init
            if N_completed_epochs > 0:
                Monitored_values['time(s)/iter'] = '{:.1f}'.format((time.time() - start_time) / N_completed_epochs)
            else:
                Monitored_values['time(s)/iter'] = '----'
            Monitored_values['lr'] = '{:.4g}'.format(lr)
            if config.adapt_critic:
                Monitored_values['N_D_steps'] = '{}'.format(N_D_steps)
            Monitored_values['alpha'] = '{: >5.3f}'.format(Generator_net.alpha)
            Monitored_values['Res'] = '{}x{}'.format(Generator_net.image_size, Generator_net.image_size)
            Monitored_values['Loss_real (<D(x)>_x)'] = '{: >#7.4g}'.format(epoch_stats['score_real'])
            Monitored_values['Loss_fake (<D(G(z))>)'] = '{: >#7.4g}'.format(epoch_stats['score_fake'])
            Monitored_values['G_loss'] = '{: >#7.4g}'.format(epoch_stats['G_loss'])
            Monitored_values['D_loss'] = '{: >#7.4g}'.format(epoch_stats['D_loss'])
            if epoch_stats['D_grad_pen'] != 0:
                Monitored_values['D_grad_pen'] = '{: >#7.4g}'.format(epoch_stats['D_grad_pen'])
            if epoch_stats['G_sim_loss'] != 0:
                Monitored_values['G_sim_loss'] = '{: >#7.4g}'.format(epoch_stats['G_sim_loss'])
            msg = ', '.join([s + ':' + v for s, v in Monitored_values.items()])
            print(msg)

        # Update the learning rates of the optimizers
        update_lr(optimizer_dis, epoch)
        update_lr(optimizer_gen, epoch)

        # Save epoch losses in timeseries
        Score_real_series[epoch - 1] = epoch_stats['score_real']
        Score_fake_series[epoch - 1] = epoch_stats['score_fake']
        G_loss_series[epoch - 1] = epoch_stats['G_loss']
        D_loss_series[epoch - 1] = epoch_stats['D_loss']

        # Checkpoint
        if epoch % config.checkpointing_period == 0:
            # Save weights
            checkpoint.save_state(epoch)

            # Create a set of fake images with the generator.
            # Set the seed to always use the same random latent vector
            Generator_net.train(False)
            Fake_samples_filepath = os.path.join(config.samples_sub_dir, 'Samples_{}_{:d}.png'.format(config.ID, epoch))
            plot_gen_samples(Generator_net, N_images=16, seed=0, filename=Fake_samples_filepath)
            Generator_net.train(True)

            # Update the training summary plot.
            plot_scores(Score_real_series[:epoch], Score_fake_series[:epoch], training_summary_filename)

            # Plot a histogram of the norm of the parameter gradients.
            grad_norm_hist_filename = os.path.join(config.plots_dir, 'Gradient_norms_{}.png'.format(config.ID))
            plot_grad_norm(Generator_net, Discriminator_net, grad_norm_hist_filename)


def wgan_train():
    Score_real = None
    Score_fake = None
    D_loss_val = None
    for epoch in range(epoch_init, config.N_epochs + 1):
        # Initialize epoch losses
        epoch_score_real = 0
        epoch_score_fake = 0
        epoch_D_loss = 0
        epoch_G_loss = 0

        # Determine the number of discriminator training steps
        if config.adapt_critic:
            N_D_steps = Calculate_D_steps(Score_real_series, Score_fake_series, 1, config.n_critic, 10)
        else:
            N_D_steps = config.n_critic

        for (i, images) in enumerate(dataloader):
            # Move images to the GPU or mps device
            if device.type != 'cpu':
                images = images.to(device)

            # Train discriminator for many iterations
            for j in range(N_D_steps):
                # Zero gradients and setup labels
                D_loss_val, Score_real, Score_fake = D_loss(images)

                Discriminator_net.zero_grad()
                D_loss_val.backward()
                optimizer_dis.step()

                # Clamp discriminator weights to ensure Lipshitz condition.
                for param in Discriminator_net.parameters():
                    param.data.clamp_(-0.01, 0.01)

            # Train generator
            Generator_net.zero_grad()
            G_loss_val, Z_latent = G_loss(images)

            # Add batch similarity loss.
            if config.sim_loss_lambda > 0:
                sim_loss_lambda = similarity_loss(images, Z_latent, config.sim_loss_lambda)
                G_loss_val += sim_loss_lambda

            # Gradient descent on generator
            G_loss_val.backward()
            optimizer_gen.step()

            # Accumulate the losses of the current epoch
            epoch_score_real += Score_real.item()
            epoch_score_fake += Score_fake.item()
            epoch_D_loss += D_loss_val.item()  # Only accumulate the loss of the last discriminator iteration
            epoch_G_loss += G_loss_val.item()

        # Monitoring
        msg = 'Epoch: %d, N_D_steps: %d | Loss_real (<D(x)>_x): %.4f, Loss_fake (<D(G(z))>): %.4f, G_loss: %.4f, D_loss: %.4f'
        print(msg % (epoch, N_D_steps, epoch_score_real, epoch_score_fake, epoch_G_loss, epoch_D_loss))

        # Save epoch losses in timeseries
        Score_real_series[epoch - 1] = epoch_score_real
        Score_fake_series[epoch - 1] = epoch_score_fake
        G_loss_series[epoch - 1] = epoch_G_loss
        D_loss_series[epoch - 1] = epoch_D_loss

        # Checkpoint
        if epoch % config.checkpointing_period == 0:
            # Save weights
            checkpoint.save_state(epoch)

            # Create the set of fake images with the predefined evaluation noise
            Generator_net.train(False)
            eval_fake = Generator_net(eval_noise)
            Fake_samples_filepath = os.path.join(config.images_dir,
                                                 'Samples_{}_{:d}.png'.format(config.ID, epoch))
            torchvision.utils.save_image(eval_fake.detach(), Fake_samples_filepath, nrow=4, normalize=True)
            Generator_net.train(True)

            # Update the training summary plot
            plot_scores(Score_real_series, Score_fake_series, training_summary_filename)

            # Plot a histogram of the norm of the parameter gradients.
            grad_norm_hist_filename = os.path.join(config.plots_dir,
                                                   'Gradient_norms_{}_{}.png'.format(config.ID, epoch))
            plot_grad_norm(Generator_net, Discriminator_net, grad_norm_hist_filename)


def dcgan_train():
    LATENT_DIM = 100  # Dimension of the latent space of the generator
    FAKE_LABEL = 0
    REAL_LABEL = 1
    BCE_loss = torch.nn.BCELoss()

    for epoch in range(1, config.N_epochs + 1):
        epoch_D_loss_real = 0
        epoch_D_loss_fake = 0
        epoch_G_loss = 0
        epoch_D_x = 0
        epoch_D_G = 0
        for (i, images) in enumerate(dataloader, start=0):
            # Move images to the GPU
            if device.type != images.device.type:
                images = images.to(device)

            # Train generator
            # Zero gradients and setup labels
            Discriminator_net.zero_grad()
            real_labels = torch.full((images.size(0), 1), fill_value=REAL_LABEL, device=device)
            fake_labels = torch.full((images.size(0), 1), fill_value=FAKE_LABEL, device=device)

            # Generate fake images with the generator
            Z = torch.randn((images.size(0), LATENT_DIM, 1, 1), device=device)
            fake_images = Generator_net(Z)

            # Calculate discriminator loss on real images (# - <log(D(x))>_x)
            dis_output_real = Discriminator_net(images)
            Dis_loss_real = BCE_loss(dis_output_real, real_labels)
            Dis_loss_real.backward()
            epoch_D_x += dis_output_real.sum().item()

            # Calculate discriminator loss on fake images (- <log(1-D(G(z)))>_z)
            # detach fake_images to avoid computing grads on generator
            dis_output_fake = Discriminator_net(fake_images.detach())
            Dis_loss_fake = BCE_loss(dis_output_fake, fake_labels)
            Dis_loss_fake.backward()
            epoch_D_G += dis_output_fake.sum().item()

            # Gradient descent on discriminator
            optimizer_dis.step()

            # Train generator
            # Calculate generator loss on fake images (- <log(D(G(z)))>_z)
            Generator_net.zero_grad()
            output = Discriminator_net(fake_images)
            G_loss = BCE_loss(output, real_labels)
            G_loss.backward()

            # Gradient descent on generator
            optimizer_gen.step()

            # Monitoring
            epoch_D_loss_real += Dis_loss_real.item()
            epoch_D_loss_fake += Dis_loss_fake.item()
            epoch_G_loss += G_loss.item()

        print('Epoch: %d | D_loss_real: %.4f, D_loss_fake: %.4f | G_loss: %.4f | <D(x)>= %.2f <D(G(z))>=%.2f'
              % (epoch, epoch_D_loss_real, epoch_D_loss_fake, epoch_G_loss, epoch_D_x / N_real_images,
                 epoch_D_G / N_real_images))

        Score_real_series[epoch - 1] = epoch_D_loss_real
        Score_fake_series[epoch - 1] = epoch_D_loss_fake
        G_loss_series[epoch - 1] = epoch_G_loss

        if epoch % config.checkpointing_period == 0:
            # Save weights
            Weights_G_filepath = os.path.join(config.weights_dir, 'Generator_{:d}.pth'.format(epoch))
            Weights_D_filepath = os.path.join(config.weights_dir, 'Discriminator_{:d}.pth'.format(epoch))
            torch.save(Generator_net.state_dict(), Weights_G_filepath)
            torch.save(Discriminator_net.state_dict(), Weights_D_filepath)

            # Create a set of fake images with a constant amount of noise
            eval_fake = Generator_net(eval_noise)
            Fake_samples_filepath = os.path.join(config.images_dir, 'Fake_samples_{:d}.png'.format(epoch))
            torchvision.utils.save_image(eval_fake.detach(), Fake_samples_filepath, nrow=4, normalize=True)

            # Update the training summary plot
            plot_scores(Score_real_series, Score_fake_series, G_loss_series, training_summary_filename)


# Start training
if __name__ == '__main__':
    if config.pggan:
        pggan_train()
    elif config.wgan:
        wgan_train()
    else:
        raise Exception('dcgan training is not implemented with new configs')
        dcgan_train()
