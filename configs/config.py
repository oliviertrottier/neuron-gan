import os
import sys
import uuid
import numpy as np

##################### Default configurations #####################
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
grad_lambda = 10  # Weight parameter of the gradient penalty term in the loss function
transit_sch = [50000, 100000, 150000, 200000, 250000]  # Schedule where a resolution transition starts
transit_period = None  # Period at which a resolution transition occurs. Overwrites transit_sch option
alpha_step = 0.0001  # Increment of alpha parameter during a resolution transition

# Training
ID = uuid.uuid4().hex[:4]  # ID of the training run
samples_sub_dir = os.path.join(images_dir, '{}'.format(ID))  # Sub-directory where samples are saved
RMSprop = False  # Use RMSprop optimizer
learning_rate = 0.0001  # Learning rate of stochastic gradient descent (SGD)
batch_size = 16  # Number of dataset images used per training iteration
N_epochs = 250000  # Number of training epochs
beta1 = 0.5  # Beta_1 parameter of the Adam optimizer
sim_loss_lambda = 0.0  # Weight of the similarity term in the loss function
sim_loss_lambda_decay_rate = 0.0  # Decay (in %) of sim_loss_lambda at each epoch
drift_epsilon = 0.001  # Weight of the discriminator drift loss
resume = False  # Resume training
N_workers = 2  # Number of workers to load the data onto the device
seed = 1  # Random seed used by torch
checkpointing_period = 100  # Number of epochs between each checkpoint of the model weights and sample generator
device = 'cpu'  # Device used for training. Can be one of ['cpu', 'mps', 'cuda'].
pin_memory = True  # Pin data to device memory in data loader

# Dataset
dataset_name = 'science_2022'  # Name of the dataset
dataset_dir = os.path.join(data_dir, dataset_name)  # Directory where the dataset image are located
translation = 0.05  # Amount of translation (in % of the image size) used in data augmentation.

# Architecture
latent_dim = 512  # Dimension of the latent space of the generator network
image_size = 512  # Largest size of the square image in the dataset. Must be a power of 2.
N_colors = 1  # Number of color channels in the images
LeakyReLU_leak = 0.2  # Leak coefficient of the LeakyReLU activation
N_gen_features = [128, 64, 32, 32, 16, 16]  # Number of feature maps in each layer of the generator.
N_dis_features = [16, 16, 32, 32, 64, 128]  # Number of feature maps in each layer of the discriminator.

# Determine the local variables. Used to get the name of all configurations.
local_vars = locals()


# Determine the number of feature maps in each layer of the discriminator and generator
def define_ID_dependent_configs():
    global N_gen_features, N_dis_features, samples_sub_dir
    if ID in ['0004', '0005']:
        N_gen_features = [1024, 512, 256, 128, 64, 32, 16, 8]
        N_dis_features = [16, 32, 64, 128, 128, 128, 128]
    elif ID in ['0006']:
        N_gen_features = [512, 256, 128, 64, 32, 16, 8, 8]
        N_dis_features = [64, 128, 256, 256, 256, 128, 64]
    elif ID in ['0007']:
        N_gen_features = [512, 256, 128, 64, 32, 16]
        N_dis_features = [16, 32, 64, 128, 256, 512]
    elif ID in ['0008']:
        N_gen_features = [512, 256, 128, 64]
        N_dis_features = [64, 128, 256, 512]
    elif ID in ['0009']:
        # PGGAN network.
        # N_gen_features = [512, 512, 512, 512, 256]
        # N_dis_features = [256, 512, 512, 512, 512]
        N_gen_features = [32, 32, 32, 32, 16, 16]
        N_dis_features = [16, 16, 32, 32, 32, 32]
    else:
        # 0010 and onwards
        # PGGAN network.
        N_gen_features = [128, 64, 32, 32, 16, 16]
        N_dis_features = [16, 16, 32, 32, 64, 128]

    # Initialize the sub samples directory
    samples_sub_dir = os.path.join(images_dir, '{}'.format(ID))  # Sub-directory where samples are saved


# Determines if a local variable is a config
def is_var_a_config(x: str):
    x_repr = repr(local_vars[x])
    is_var_config = x in local_vars
    is_var_config &= not x.startswith('__')  # Remove builtins
    is_var_config &= 'module' not in x_repr  # Remove modules
    is_var_config &= 'function' not in x_repr  # Remove functions
    is_var_config &= x != 'local_vars'  # Remove "local_vars" variable
    return is_var_config


# Gather the name of all configurations
Configs_name = {var: val for var, val in local_vars.items() if is_var_a_config(var)}


# Print the value of the configurations
def print_configs():
    print('Configurations:')
    for name in Configs_name:
        print(f'{name}:', eval(name))


# Validates values of configs
def validate_configs():
    # Ensure directories are absolute
    global dataset_dir, images_dir, samples_sub_dir, weights_dir, plots_dir
    dataset_dir = os.path.abspath(dataset_dir)
    images_dir = os.path.abspath(images_dir)
    samples_sub_dir = os.path.abspath(samples_sub_dir)
    weights_dir = os.path.abspath(weights_dir)
    plots_dir = os.path.abspath(plots_dir)

    # Create the directories, if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(samples_sub_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Checks
    ## Ensure root directory is correctly defined
    root_dir_exp = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    root_dir_err_msg = 'The root directory is expected to be:\n{}\n{} was given'.format(root_dir_exp, root_dir)
    assert root_dir == root_dir_exp, root_dir_err_msg

    image_size_log = np.round(np.log2(image_size))
    assert image_size == 2 ** image_size_log, 'Image size must be a power of 2.'
    assert device in ['cpu', 'cuda', 'mps'], f'device:{device} is not supported.'
    assert ID != '', 'The training ID is undefined.'

    ## Check PGGAN configs
    if pggan:
        global transit_sch, transit_period

        # Check that the number of layers are the same in the discriminator and generator
        err_msg = 'The number of layers in the generator and discriminator must match.'
        assert len(N_gen_features) == len(N_dis_features), err_msg

        # Check that the lowest resolution is no smaller than 4x4
        N_upsamples = len(N_gen_features) - 1
        Image_size_final = image_size
        Image_size_initial = Image_size_final // (2 ** N_upsamples)
        assert Image_size_initial >= 4, 'The initial image size must be >= 4. Reduce the number of layers'

        # Check if a transition period is set and overwrite the transit_sch option if it has a value
        if transit_period is not None:
            transit_sch = [i * transit_period for i in range(1, N_upsamples + 1)]

        # Check that the number of transitions match the number of convolution layers in the network
        err_msg = 'The number of transitions ({}) does not match the number of convolution layers ({})'.format(
            len(transit_sch), N_upsamples)
        assert N_upsamples == len(transit_sch), err_msg

        # Check that the transition schedule is sufficiently separated
        # to allow one transition to finish before the next one starts.
        N_transition_epochs = np.ceil(1 / alpha_step)
        err_msg = 'The transitions must be separated by at least {} epochs'.format(N_transition_epochs)
        assert np.all(np.diff(transit_sch) > N_transition_epochs), err_msg


# Import the feature maps
define_ID_dependent_configs()


# Overwrites configurations from input file
def import_configs(filename):
    # Ensure the config file is a .py file
    base_name, ext = os.path.splitext(filename)
    if ext == '':
        # Add .py extension to filename
        filename += '.py'
    elif ext != '.py':
        raise ValueError('Filename must be a .py file')

    # Ensure the file exists
    config_filepath = os.path.join(configs_dir, filename)
    assert os.path.exists(config_filepath), f'config {filename} does not exist in {configs_dir}'

    # Import python config module dynamically with its filename
    import importlib.util
    global imported_configs
    spec = importlib.util.spec_from_file_location('user.config', config_filepath)
    imported_configs = importlib.util.module_from_spec(spec)
    sys.modules["user.config"] = imported_configs
    spec.loader.exec_module(imported_configs)

    # Overwrite each input config
    for config_in_name in dir(imported_configs):
        config_repr = repr(getattr(imported_configs, config_in_name))
        if config_in_name in Configs_name:
            # Declare the variable as global to ensure that exec modifies the caller scope (i.e. the module scope)
            exec(f'global {config_in_name}', globals())
            exec(f'{config_in_name} = imported_configs.{config_in_name}', globals())
        elif not config_in_name.startswith('__') and 'module' not in config_repr and 'function' not in config_repr:
            print(f'{config_in_name} is not a supported configuration and will be omitted.')

    # Redefine the samples sub-directory based on the ID
    global samples_sub_dir
    samples_sub_dir = os.path.join(images_dir, '{}'.format(ID))

    # Import the feature maps based on the ID
    define_ID_dependent_configs()

    # Validate configs
    validate_configs()


if __name__ == '__main__':
    import_configs('config_test.py')
    print(N_epochs)
