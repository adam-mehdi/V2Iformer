import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from v2iformer.model import V2Iformer
from fast_timesformer.model import FastTimeSformer

class GAN(LightningModule):

    def __init__(
        self,
        channels,
        frames,
        height,
        width,
        latent_dim: int = 128,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs
    ):
        """
        GAN that produces an image given a video clip or 3d image.

        Batch is composed of an x of shape (b, c, f, h, w) and a y of shape
            (b, c, 1, h, w)
        """
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, frames, width, height)

        self.G = V2Iformer(num_frames=frames, window_size=2, stages=2)

        self.D = nn.Sequential(
            FastTimeSformer(frames+1, channels, height, width, 1, frames_first=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.G(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch # (b, c, f, h, w)

        # generator
        if optimizer_idx == 0:
            # define targets
            one_targets = torch.ones(b, 1, device = self.device, dtype = torch.float)

            # maximize discriminator's predictions on fake images (include input clip for context)
            y_hat = self.G(x).unsqueeze(2)
            cat_y_hat = torch.cat((x, y_hat), dim = 2)
            g_loss = F.binary_cross_entropy(self.D(cat_y_hat), one_targets)

            # log the generator loss
            if logger is not None:
                self.logger.experiment['g_loss'].log(g_loss)

            return g_loss

        # discriminator
        
        if optimizer_idx == 1:
            # define targets 
            one_targets = torch.ones(b, 1, device = self.device, dtype = torch.float)
            zero_targets = torch.zeros(b, 1, device = self.device, dtype = torch.float)

            # maximize discriminator's predictions on real images
            cat_y = torch.cat((x, y), dim = 2)
            real_loss = F.binary_cross_entropy(self.D(cat_y), one_targets)

            # minimize discriminator's predictions on fake images
            y_hat = self.G(x).unsqueeze(2)
            cat_y_hat = torch.cat((x, y_hat), dim = 2)
            fake_loss = F.binary_cross_entropy(self.D(cat_y_hat), zero_targets)

            # average real and fake loss
            d_loss = (real_loss + fake_loss) / 2

            # log motley metrics using the new Neptune integration 
            ## <delete this code block if you are not using Neptune>
            if logger is not None:
                sample_fake = y_hat[:4]
                sample_real = y[:4]
                samples_comb = torch.stack([sample_fake, sample_real])
                samples_grid = make_grid(samples_comb, nrow=4, normalize=True, scale_each=True).permute(1,2,0).cpu()
                
                self.logger.experiment['d_loss'].log(d_loss)
                self.logger.experiment['image_grid'].log(File.as_image(grid_fake))

            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


      
# test      
def test():
    b, c, f, h, w = 8, 3, 3, 32, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(b, c, f, h, w).to(device)
    y = torch.randn(b, c, 1, h, w).to(device)
    model = GAN(c, f, h, w).to(device)
    
