import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, latent_dim, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_dim, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class Improvement(nn.Module):
    def __init__(self, autoencoder, bcr_loss, gui_loss, lambda_val=1.0, gamma_val=1.0):
        super(Improvement, self).__init__()
        self.autoencoder = autoencoder
        self.bcr_loss = bcr_loss
        self.gui_loss = gui_loss
        self.lambda_val = lambda_val
        self.gamma_val = gamma_val

    def forward(self, x):
        x_recon, z = self.autoencoder(x)

        bcr_loss = self.bcr_loss(z)
        gui_loss = self.gui_loss(x, z)
        recon_loss = self.loss_and_similarity(x, x_recon)

        total_loss = bcr_loss + self.lambda_val * gui_loss + self.gamma_val * recon_loss
        return total_loss, x_recon

    def loss_and_similarity(self, x, x_recon):
        # Reconstruction loss using Mean Squared Error
        recon_loss = F.mse_loss(x_recon, x)
        # Similarity score between original image and its reconstruction
        similarity = 1 - recon_loss
        # Combine the reconstruction loss and similarity score
        recon_loss += self.gamma_val * (1 - similarity)
        return recon_loss
