import os
import copy
import numpy as np
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from skimage.filters import threshold_multiotsu
from configs import config


# User-defined transform to replace low pixel values with white noise
def replace_zero_with_noise(image: Image.Image, noise_mean, noise_std):
    # Convert image to numpy array, find zero pixels and add noise to them
    image_array = np.array(image)
    image_is_zero = image_array == 0
    Noise_vals = noise_std * np.random.randn(*image_array.shape) + noise_mean
    image_array[image_is_zero] = Noise_vals[image_is_zero]
    image.paste(Image.fromarray(image_array))
    return image


# Transform to renormalize tensor values to a given range.
class Renormalize:
    def __init__(self, range_new: tuple, range_curr: tuple = None):
        self.range_new = range_new
        self.range_curr = range_curr

    def __call__(self, tensor: torch.Tensor):
        # Find the current range of tensor values
        if self.range_curr is None:
            # The current range is not defined. Use the min and max of the tensor
            Range_curr = (torch.min(tensor), torch.max(tensor))
        else:
            Range_curr = self.range_curr

        tensor.subtract_(Range_curr[0])
        tensor.divide_(Range_curr[1] - Range_curr[0])
        tensor.mul_(self.range_new[1] - self.range_new[0])
        tensor.add_(self.range_new[0])
        return tensor


# Dataset class for neuron images
class NeuronDataset(Dataset):
    def __init__(self, directory: str, image_size=config.image_size, augmentations: bool = True,
                 im_translation: float = 0.0, load_all=None):
        # To avoid memory issues, use serialized lists only in __getitem__.
        # See these references for more info:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        # Check if the dataset directory exists
        if not os.path.exists(directory):
            raise ValueError('The dataset path {} does not exist.'.format(directory))

        self.dir = directory
        self.image_size = image_size  # Size of the square images output by the dataset
        self.image_size_max = image_size  # Maximum size of the square images output by the dataset

        # Find all files in the dataset directory.
        self.filenames = []
        for file in os.listdir(self.dir):
            self.filenames.append(os.path.join(self.dir, file))

        # Convert filenames to serialized numpy array.
        self.filenames = np.array(self.filenames)

        # Define transforms to pad images with noisy pixels and convert to tensors
        Pad_size = self.image_size // 4
        self.pad_transform = torchvision.transforms.Pad(Pad_size, fill=0, padding_mode='constant')
        self.ToTensor_transform = torchvision.transforms.ToTensor()

        # Preload the images if there are less than 100MB (useful when dataset is small)
        self.filesizes = []
        self.images = []
        for filename in self.filenames:
            self.filesizes.append(os.path.getsize(filename))
        if load_all is None:
            self.load_all = sum(self.filesizes) < 100e6
        else:
            self.load_all = load_all

        # Load all images and calculate the average and std of the noise.
        N_images = len(self.filenames)
        self.images_noise_mean = np.empty(N_images, dtype=np.double)
        self.images_noise_std = np.empty(N_images, dtype=np.double)
        for i, filename in enumerate(self.filenames):
            img = Image.open(filename)

            # Use Otsu's thresholding to find the noise pixels.
            img_array = np.array(img)
            otsu_thresh = threshold_multiotsu(img_array, classes=4)
            img_is_noise = np.logical_and(img_array > 0.0, img_array < otsu_thresh[0])
            noise_val = img_array[img_is_noise]
            self.images_noise_mean[i] = np.mean(noise_val)
            self.images_noise_std[i] = np.std(noise_val)

            if self.load_all:
                # Pad image
                img = self.pad_transform(img)

                # Add noise to zero pixels
                img = replace_zero_with_noise(img, self.images_noise_mean[i], self.images_noise_std[i])

                # Save the new image as a torch.tensor
                self.images.append(self.ToTensor_transform(img))

            # Close PIl image
            img.close()

        # Initialize some basic augmentations.
        if augmentations:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomAffine(degrees=180, translate=(im_translation, im_translation), fill=0),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25)
            ])
        else:
            self.transforms = torchvision.transforms.Compose([])

        # Add transforms to fix image size and pixel range.
        self.transforms.transforms.extend([
            torchvision.transforms.CenterCrop(size=self.image_size),
            Renormalize(range_new=(-1, 1), range_curr=(0, 1))
        ])

    def __getitem__(self, i):
        # Load image and apply transforms.
        if self.load_all:
            # If all images have been loaded, simply index the images array.
            # This is faster than opening images each time since the loaders only to copy from RAM.
            Tensor_image = copy.deepcopy(self.images[i])
        else:
            # Otherwise, open the image with its filename.
            with Image.open(self.filenames[i]) as PIL_image:
                # Pad image
                PIL_image = self.pad_transform(PIL_image)

                # Add noise to zero pixels
                PIL_image = replace_zero_with_noise(PIL_image, self.images_noise_mean[i], self.images_noise_std[i])

                # Convert to torch.tensor
                Tensor_image = self.ToTensor_transform(PIL_image)

        return self.transforms(Tensor_image)

    # Method to change the size of the output image and add a resize transform accordingly
    def set_image_size(self, size: int):
        assert size <= self.image_size_max, 'The image size ({}) must be < {}.'.format(size, self.image_size_max)
        self.image_size = size
        Res_trans = torchvision.transforms.Resize(self.image_size, antialias=True)

        # Check if the last transform is a Resize transform.
        # If yes, replace it. If not, add one.
        if isinstance(self.transforms.transforms[-1], torchvision.transforms.Resize):
            if self.image_size < self.image_size_max:
                self.transforms.transforms[-1] = Res_trans
            else:
                # Remove the Resize transform since the output size is the same as the maximum.
                self.transforms.transforms.pop()
        elif self.image_size < self.image_size_max:
            # Add resize transform only if the input size is strictly less than the max image size.
            self.transforms.transforms.append(Res_trans)

    def __len__(self):
        return len(self.filenames)


class DatasetIterator:
    def __init__(self, dataset: NeuronDataset, batch_size: int, device: torch.device):
        if not dataset.load_all:
            raise Exception('On-device iteration is only possible when all images are loaded.')
        self.dataset = dataset
        self.images = dataset.images
        self.N_images = len(self.images)
        self.batch_size = batch_size
        self.device = device
        self.img_dtype = dataset.images[0].dtype
        self.output_size = (batch_size, 1, dataset.image_size, dataset.image_size)

        # Move images to the device
        for i in range(len(dataset)):
            self.images[i] = self.images[i].to(device).unsqueeze(dim=0)

        self.images_tensor = torch.zeros(self.output_size, device=self.device, dtype=self.img_dtype)
        self.transforms = dataset.transforms

    def __iter__(self):
        # Check if the images have changed size. If yes, initialize a tensor output
        if self.output_size[-1] != self.dataset.image_size:
            self.output_size = (self.batch_size, 1, self.dataset.image_size, self.dataset.image_size)
            self.images_tensor = torch.zeros(self.output_size, device=self.device, dtype=self.img_dtype)

        # Reset image counter
        self.image_ind = 0
        return self

    def __next__(self):
        if self.image_ind < self.N_images:
            for i in range(min([self.batch_size, self.N_images - self.image_ind])):
                self.images_tensor[i, :, :, :] = self.transforms(self.images[self.image_ind])
                self.image_ind += 1
            return self.images_tensor[:i + 1, :, :, :]
        raise StopIteration


# Plot dataset at various resolutions
def plot_dataset(dataset: NeuronDataset, resolutions: list[int], filename_prefix: str = 'Dataset',
                 directory=config.images_dir):
    size_final = (dataset.image_size_max, dataset.image_size_max)
    for res in resolutions:
        dataset.set_image_size(res)
        images = torch.stack([img for img in dataset], dim=0)

        # Upsample the eval images to ensure that they all have the same size.
        if images.size(-1) != dataset.image_size_max:
            images = torch.nn.functional.interpolate(images, size=size_final)
        Samples_filepath = os.path.join(directory, '{}_{}x{}.png'.format(filename_prefix, res, res))
        torchvision.utils.save_image(images, Samples_filepath, nrow=4, normalize=True)


if __name__ == "__main__":
    dataset_name = 'science_2022'
    dataset_dir = os.path.abspath(os.path.join(config.data_dir, dataset_name))
    images_dir = config.images_dir

    Main_dataset = NeuronDataset(dataset_dir, augmentations=True)
    Main_dataset_no_aug = NeuronDataset(dataset_dir, augmentations=False)

    # Plot examples of the images
    # Main_dataset.set_image_size(16)
    # plot_sample(Main_dataset)

    # Plot the dataset without augmentations at different resolutions.
    resolutions = [2 ** i for i in range(4, 10)]
    plot_dataset(Main_dataset_no_aug, resolutions, filename_prefix=f'{dataset_name}_no_aug')
    plot_dataset(Main_dataset, resolutions, filename_prefix=f'{dataset_name}_aug')
