import os
import argparse

from configs import config
from models import Generator_PG
from utils import plot_gen_samples

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=16, help='Number of samples created')
parser.add_argument('-output', type=str, default='samples_default.png',
                    help='Filename of the output image file stored in ./samples')
parser.add_argument('-weights', type=str, default='gen_dis_default.pth',
                    help='Filename of the weights stored in ./weights')
options = parser.parse_args()

Weights_filepath = os.path.join(config.weights_dir, options.weights)
Output_filepath = os.path.join(config.images_dir, options.output)
if not os.path.exists(Weights_filepath):
    raise FileExistsError(f'{Weights_filepath} does not exist. Run setup.py.')

# Initialize the generator with the weights
Generator_net = Generator_PG.from_state_dict(Weights_filepath)

# Generate and plot the samples
plot_gen_samples(Generator_net, options.n, filename=Output_filepath)
