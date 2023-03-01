import torch
import torch.nn as nn
from utils import save_vars, sample_latent_vec


# Function to calculate the Wasserstein loss of the discriminator network
class D_W_loss(nn.Module):
    def __init__(self, generator_net, discriminator_net, drift_epsilon=0.0):
        super().__init__()
        self.generator_net = generator_net
        self.discriminator_net = discriminator_net
        self.drift_epsilon = drift_epsilon

    def forward(self, real_images):
        # Get basic info
        batch_size = real_images.size(0)
        device = real_images.device
        im_size = real_images.size(-1)

        # Calculate score on real images (<D(x)>_x)
        real_images_score = self.discriminator_net(real_images)
        score_real = real_images_score.mean()

        # Generate fake images with the generator
        z = sample_latent_vec((batch_size, self.generator_net.latent_dim), device=device)
        fake_images = self.generator_net(z).detach()  # detach so that generator gradients are not computed

        # Calculate score on fake images (<D(G(z))>_z)
        score_fake = self.discriminator_net(fake_images).mean()

        # Calculate the total loss for the discriminator (- <D(x)>_x + <D(G(z))>_z)
        D_loss = - score_real + score_fake

        # Check for nans
        if torch.isnan(score_real):
            save_vars(locals())
            raise ValueError('Real loss is nan.')
        if torch.isnan(score_fake):
            # print(f'z_max: {torch.min(z):.5f}, z_min: {torch.max(z):.5f}')
            save_vars(locals())
            raise ValueError('Fake loss is nan.')

        # Add drift loss.
        if self.drift_epsilon > 0:
            D_loss += self.drift_epsilon * torch.square(real_images_score).mean()

        return D_loss, score_real, score_fake


# Function to calculate the Wasserstein loss of the generator network
class G_W_loss(nn.Module):
    def __init__(self, generator_net, discriminator_net):
        super().__init__()

        # Store the networks to use them in the forward pass
        self.generator_net = generator_net
        self.discriminator_net = discriminator_net

    def forward(self, real_images_batch):
        # Generate fake images with the generator
        batch_size = real_images_batch.size(0)
        device = real_images_batch.device
        z_latent = sample_latent_vec((batch_size, self.generator_net.latent_dim), device=device)
        fake_images = self.generator_net(z_latent)

        # Calculate generator loss on fake images (- <D(G(z))>_z)
        G_loss = - self.discriminator_net(fake_images).mean()

        # Check for nan.
        if torch.isnan(G_loss):
            save_vars(locals())
            raise ValueError('Generator loss is nan.')

        return G_loss, z_latent


# Function to calculate the least squares loss of the discriminator network
# Reference: "Least Squares Generative Adversarial Networks." 2017, http://arxiv.org/abs/1611.04076
def D_LS_loss(generator_net, discriminator_net, real_images_batch: torch.Tensor, Lambda: float = None):
    # Get basic info
    latent_dim = generator_net.latent_dim
    batch_size = real_images_batch.size(0)
    device = real_images_batch.device

    # Generate fake images with the generator
    z = sample_latent_vec((batch_size, latent_dim), device=device)
    fake_images = generator_net(z).detach()  # detach so that the generator gradients are not computed (faster)

    # Calculate score on real images (<D(x)>_x)
    Score_real_batch = discriminator_net(real_images_batch)
    Score_real_avg = Score_real_batch.mean()

    # Calculate score on fake images (<D(G(z))>_z)
    # detach fake_images to avoid computing grads on generator
    Score_fake_batch = discriminator_net(fake_images)
    Score_fake_avg = Score_fake_batch.mean()

    # Calculate the total loss for the discriminator (<(D(x) - 1)^2>_x + <D(G(z))^2>_z)
    D_loss = torch.mean((Score_real_batch - 1) ** 2) + torch.mean(Score_fake_batch ** 2)

    # Check for nans
    if torch.isnan(Score_real_avg):
        save_vars(locals())
        raise ValueError('Real loss is nan.')
    if torch.isnan(Score_fake_avg):
        save_vars(locals())
        raise ValueError('Fake loss is nan.')
    if torch.isnan(D_loss):
        save_vars(locals())
        raise ValueError('D loss is nan.')

    return D_loss, Score_real_avg, Score_fake_avg


# Function to calculate the least squares loss of the generator network
# Reference: "Least Squares Generative Adversarial Networks." 2017, http://arxiv.org/abs/1611.04076
def G_LS_loss(generator_net, discriminator_net, real_images_batch: torch.Tensor, Lambda: float = None):
    # Get basic info
    latent_dim = generator_net.latent_dim
    batch_size = real_images_batch.size(0)
    device = real_images_batch.device

    # Generate fake images with the generator
    z = sample_latent_vec((batch_size, latent_dim), device=device)
    fake_images = generator_net(z).detach()  # detach so that the generator gradients are not computed (faster)

    # Calculate score on fake images (<D(G(z))>_z)
    # detach fake_images to avoid computing grads on generator
    Score_fake_batch = discriminator_net(fake_images)
    Score_fake_avg = Score_fake_batch.mean()

    # Calculate the total loss for the generator (<(D(G(z)) - 1)^2>_z)
    G_loss = torch.mean((Score_fake_batch - 1) ** 2)

    # Check for nans
    if torch.isnan(Score_fake_avg):
        save_vars(locals())
        raise ValueError('Fake loss is nan.')
    if torch.isnan(G_loss):
        save_vars(locals())
        raise ValueError('D loss is nan.')

    return G_loss, Score_fake_avg


# Calculate gradient penalty loss of the discriminator
# Reference: See Algorithm 1 in "Improved Training of Wasserstein GANs.", 2017. http://arxiv.org/abs/1704.00028
class D_grad_pen_loss(nn.Module):
    def __init__(self, generator_net, discriminator_net, Lambda):
        super().__init__()

        # Store the networks to use them in the forward pass
        self.generator_net = generator_net
        self.discriminator_net = discriminator_net
        self.Lambda = Lambda

    def forward(self, real_images):
        # Get basic info
        if self.Lambda > 0:
            latent_dim = self.generator_net.latent_dim
            real_im_size = real_images.size()
            batch_size, im_size = real_im_size[0], real_im_size[-2:]
            device = real_images.device

            # Generate fake images.
            z_latent = sample_latent_vec((batch_size, latent_dim), device=device)
            x_tilde = self.generator_net(z_latent).detach()

            # Randomly interpolate between real and fake image (x_tilde)
            epsilon = torch.rand((batch_size, 1, 1, 1), device=device)
            x_hat = epsilon * real_images + (1 - epsilon) * x_tilde

            x_hat.requires_grad_()
            output = self.discriminator_net(x_hat)
            Disc_grad = torch.autograd.grad(outputs=output.sum(), inputs=x_hat, create_graph=True)[0]
            Gradient_penalty_loss = self.Lambda * torch.mean((Disc_grad.norm(2, dim=(1, 2, 3)) - 1) ** 2)
            # Gradient_penalty_loss = Lambda * torch.mean(torch.square(Disc_grad.norm(2, dim=1) - 1))
        else:
            Gradient_penalty_loss = torch.tensor(0)
        return Gradient_penalty_loss


# Define a loss function that discourages mode collapse by
# comparing the cosine similarity of the latent vectors with the images.
def similarity_loss(images_batch: torch.FloatTensor, Z_batch: torch.FloatTensor, Lambda: float = 1.0):
    # Change the images_batch and Z_batch into matrices where dim=0 is the batch dimension.
    batch_size = images_batch.size(0)
    images_mat = images_batch.view(batch_size, -1)
    Z_mat = Z_batch.view(batch_size, -1)

    # Normalize the rows of the matrices.
    images_mat = images_mat / images_mat.norm(2, dim=1, keepdim=True)
    Z_mat = Z_mat / Z_mat.norm(2, dim=1, keepdim=True)

    # Calculate cosine similarity between all pairs of Z vectors in the latent space.
    Z_cos_sim = torch.matmul(Z_mat, Z_mat.t())

    # Calculate cosine similarity between all pairs of images.
    images_cos_sim = torch.matmul(images_mat, images_mat.t())

    # Calculate the averaged squared difference between the latent and image cosine similarity
    # Divide the difference by 2 because each pair appears twice in images_cos_sim (symmetric matrix)
    N_pairs = batch_size * (batch_size - 1)
    Sim_loss = Lambda * torch.pow((Z_cos_sim - images_cos_sim), 2).sum() / N_pairs
    return Sim_loss
