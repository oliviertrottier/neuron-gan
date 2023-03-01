import os

# Directories
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
configs_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
images_dir = os.path.join(root_dir, 'images')
weights_dir = os.path.join(root_dir, 'weights')
plots_dir = os.path.join(root_dir, 'plots')

# WGAN config
wgan = False  # Use the Wasserstein GAN (WGAN) architecture and loss function. See models.py
n_critic = 1  # Number of training steps of the critic for each training step of the generator
adapt_critic = False  # Adapt the number critic training steps based on the loss difference between the critic and generator
weights_init = ''  # Filename of the initial weights of the discriminator and generator
unroll_steps = 0  # Number of unrolled discriminator steps (Unrolled GAN)

# PGGAN config
pggan = True  # Use the progressively growing GAN (PGGAN) architecture. See models.py
grad_pen_lambda = 10  # Weight parameter of the gradient penalty term in the loss function
transit_sch = [25000, 50000, 75000, 100000, 125000]  # Schedule where a resolution transition starts
transit_period = None  # Period at which a resolution transition occurs. Overwrites transit_sch option
alpha_step = 0.0001  # Increment of alpha parameter during a resolution transition

# Training
ID = ''  # ID of the training run
samples_sub_dir = os.path.join(images_dir, '{}'.format(ID))  # Sub-directory where samples are saved
RMSprop = False  # Use RMSprop optimizer
learning_rate = 0.0001  # Learning rate of stochastic gradient descent (SGD)
batch_size = 8  # Number of dataset images used per training iteration
N_epochs = 150000  # Number of training epochs
beta1 = 0.5  # Beta_1 parameter of the Adam optimizer
sim_loss_lambda = 0.0  # Weight of the similarity term in the loss function
sim_loss_lambda_decay_rate = 0.0  # Decay (in %) of sim_loss_lambda at each epoch
drift_epsilon = 0.001  # Weight of the discriminator drift loss
resume = True  # Resume training
N_workers = 0  # Number of workers to load the data onto the device
seed = 1  # Random seed used by torch
checkpointing_period = 100  # Number of epochs between each checkpoint of the model weights and sample generator
device = 'cpu'  # Device used for training. Can be one of ['cpu', 'mps', 'cuda'].
pin_memory = False  # Pin data to device memory in data loader

# Dataset
dataset_name = 'science_2022'  # Name of the dataset
dataset_dir = os.path.join(data_dir, dataset_name)  # Directory where the dataset image are located
translation = 0.05  # Amount of translation (in % of the image size) used in data augmentation.

# Architecture
latent_dim = 64  # Dimension of the latent space of the generator network
image_size = 512  # Largest size of the square image in the dataset. Must be a power of 2.
N_colors = 1  # Number of color channels in the images
LeakyReLU_leak = 0.2  # Leak coefficient of the LeakyReLU activation function
N_gen_features = [128, 64, 32, 32, 16, 16]  # Number of feature maps in each layer of the generator.
N_dis_features = [16, 16, 32, 32, 64, 128]  # Number of feature maps in each layer of the discriminator.