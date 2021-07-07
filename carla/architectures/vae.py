"""
Created by Hamid Eghbal-zadeh at 25.01.21
Johannes Kepler University of Linz
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
import numpy as np

Tensor = TypeVar('torch.tensor')

# base class adopted from:
# https://github.com/AntixK/PyTorch-VAE/
class BasicBetaVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = [32, 32, 64, 64],
                 beta: int = 1.,
                 gamma: float = 1.,
                 max_capacity: int = 25,
                 vae_c_stop_iter: int = 1e5,
                 loss_type: str = 'B',
                 nonlin: str = 'relu',
                 enc_bn: bool = True,
                 dec_bn: bool = True,
                 kernel_sizes: List = [4, 4, 4, 4],
                 img_size: int = 84,
                 output_padding_lst: List = [0, 1, 1, 1, 0],
                 n_context: int = 1,
                 num_samples: int = 5,
                 dec_out_nonlin: str = 'tanh',
                 prior: str = 'gauss',
                 soft_clip: bool = False,
                 strides: List = [0, 1, 1, 1, 0],
                 paddings: List = [0, 1, 1, 1, 0],
                 batch_size: int = 144,
                 geco_goal: float = 0.5,
                 reduction: str = 'mean',
                 **kwargs) -> None:

        super(BasicBetaVAE, self).__init__()

        self.geco = GECO(goal=geco_goal, step_size=3e-4)
        self.latent_dim = latent_dim
        self.geco_goal = geco_goal
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = vae_c_stop_iter
        self.nonlin = nonlin
        self.enc_bn = enc_bn
        self.dec_bn = dec_bn
        self.kernel_sizes = kernel_sizes
        self.num_samples = num_samples
        self.prior = prior
        self.dec_out_nonlin = dec_out_nonlin
        self.soft_clip = soft_clip
        self.batch_size = batch_size
        self.img_size = img_size
        self.reduction = reduction

        if hidden_dims is None:
            hidden_dims = [32, 32, 64, 64]
        self.hidden_dims = hidden_dims

        if nonlin == 'relu':
            NonLinearity = nn.ReLU
        elif nonlin == 'lrelu':
            NonLinearity = nn.LeakyReLU
        elif nonlin == 'elu':
            NonLinearity = nn.ELU

        encoder_modules = []
        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            encoder_mlist = []
            encoder_mlist.append(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=kernel_sizes[i], stride=strides[i],
                          padding=paddings[i]))
            if enc_bn:
                encoder_mlist.append(nn.BatchNorm2d(h_dim))
            encoder_mlist.append(NonLinearity())
            encoder_modules.append(
                nn.Sequential(*encoder_mlist)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)
        dummy_in = torch.rand(1, 3, img_size, img_size)
        print(dummy_in.shape)
        dummy_out = self.encoder(dummy_in)
        print(dummy_out.shape)
        self.conv_feat_shape = dummy_out.shape
        self.fc_mu = nn.Linear(np.prod(self.conv_feat_shape), latent_dim)
        self.fc_var = nn.Linear(np.prod(self.conv_feat_shape), latent_dim)

        self.context_predictor = nn.Linear(latent_dim, n_context)

        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.fc_var.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.context_predictor.weight, mode='fan_out')

        # Build Decoder
        decoder_modules = []

        self.decoder_input = nn.Linear(latent_dim, np.prod(self.conv_feat_shape))
        nn.init.kaiming_normal_(self.decoder_input.weight, mode='fan_out')
        hidden_dims.reverse()
        kernel_sizes.reverse()
        paddings.reverse()
        strides.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_mlist = []
            decoder_mlist.append(nn.ConvTranspose2d(hidden_dims[i],
                                                    hidden_dims[i + 1],
                                                    kernel_size=kernel_sizes[i],
                                                    stride=strides[i],
                                                    padding=paddings[i],
                                                    output_padding=output_padding_lst[i]))
            if dec_bn:
                decoder_mlist.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            decoder_mlist.append(NonLinearity())
            decoder_modules.append(
                nn.Sequential(*decoder_mlist)
            )

        self.decoder = nn.Sequential(*decoder_modules)

        dummy_out = self.decoder(
            torch.rand(*self.conv_feat_shape))
        print(dummy_out.shape)

        final_layer_modules = []
        final_layer_modules.append(nn.ConvTranspose2d(hidden_dims[-1],
                                                      3,
                                                      kernel_size=kernel_sizes[-1],
                                                      stride=strides[-1],
                                                      padding=paddings[-1],
                                                      output_padding=output_padding_lst[-1]
                                                      ))
        if dec_out_nonlin == 'tanh':
            final_layer_modules.append(nn.Tanh())
        elif dec_out_nonlin == 'sig':
            final_layer_modules.append(nn.Sigmoid())

        self.final_layer = nn.Sequential(*final_layer_modules)
        dummy_out = self.final_layer(self.decoder(
            torch.rand(*self.conv_feat_shape)))
        print(dummy_out.shape)
        print('done!')

    def init_weights(self, init_fn):
        self.encoder.apply(init_fn)
        self.fc_mu.apply(init_fn)
        self.fc_var.apply(init_fn)
        self.decoder_input.apply(init_fn)
        self.decoder.apply(init_fn)
        self.final_layer.apply(init_fn)
        self.context_predictor.apply(init_fn)

    def softclip(self, tensor, min):
        """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
        result_tensor = min + F.softplus(tensor - min)

        return result_tensor

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        if self.soft_clip:
            log_var = self.softclip(log_var, -6)

        # remove and replaces NaNs and infs
        mu[torch.isnan(mu)] = 0.
        log_var[torch.isnan(log_var)] = 0.

        mu[torch.isinf(mu)] = 0.
        log_var[torch.isinf(log_var)] = 0.

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        if self.loss_type in ['beta', 'annealed', 'geco']:
            return self.beta_decode(z)

    def beta_decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.conv_feat_shape[1], self.conv_feat_shape[2], self.conv_feat_shape[3])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """

        # make logvar numerically stable again!
        logvar = torch.clamp(logvar, min=logvar.min().item() - 1e-7, max=100.)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_posterior = eps * std + mu

        return sampled_posterior

    def predict_context(self, disentangled_feat: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        predicted_context = self.context_predictor(disentangled_feat)
        predicted_context = F.softmax(predicted_context)
        return predicted_context

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if self.loss_type in ['beta', 'annealed', 'geco']:
            return self.beta_forward(input, **kwargs)

    def beta_forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        if self.loss_type in ['beta', 'annealed']:
            return self.beta_loss_function(*args, **kwargs)
        elif self.loss_type == 'geco':
            return self.geco_loss_function(*args, **kwargs)

    def kl_div(self, log_var, mu):
        log_var = torch.clamp(log_var, min=log_var.min().item() - 1e-7, max=50.)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kld

    def get_kld(self, log_var, mu, shapes):
        b, ch, w, h = shapes

        raw_kld_loss = self.kl_div(log_var, mu)

        if self.reduction == 'mean':
            # average over batch
            kld_loss = raw_kld_loss.mean(dim=0)
        elif self.reduction == 'norm_batch':
            normalizing_weight = 1. / b
            # average over batch
            kld_loss = normalizing_weight * raw_kld_loss.sum(0)
        elif self.reduction == 'norm_pixel':
            normalizing_weight = 1. / (b * ch * w * h)
            # average over batch
            kld_loss = normalizing_weight * raw_kld_loss.sum(0)
        elif self.reduction == 'norm_dim':
            normalizing_weight = 1. / (b * self.latent_dim)
            # average over batch
            kld_loss = normalizing_weight * raw_kld_loss.sum(0)
        elif self.reduction == 'norm_rel':
            normalizing_weight = 1. / b
            # average over batch
            kld_loss = normalizing_weight * raw_kld_loss.sum(0)
        elif self.reduction == 'sum':
            kld_loss = raw_kld_loss.sum(dim=0)
        return kld_loss

    def get_reconstruction(self, recons, input):
        b, ch, w, h = input.shape
        if self.prior == 'gauss':
            if self.reduction == 'mean':
                # average over batch
                recons_loss = F.mse_loss(recons, input, reduction='mean')
            elif self.reduction == 'norm_batch':
                normalizing_weight = 1. / b
                recons_loss = F.mse_loss(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_pixel':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.mse_loss(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_dim':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.mse_loss(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_rel':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.mse_loss(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'sum':
                recons_loss = F.mse_loss(recons, input, reduction='sum')

        elif self.prior == 'bernoulli':
            if self.reduction == 'mean':
                # average over batch
                recons_loss = F.binary_cross_entropy(recons, input, reduction='mean')
            elif self.reduction == 'norm_batch':
                normalizing_weight = 1. / b
                recons_loss = F.binary_cross_entropy(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_pixel':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.binary_cross_entropy(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_dim':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.binary_cross_entropy(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'norm_rel':
                normalizing_weight = 1. / (b * ch * w * h)
                recons_loss = F.binary_cross_entropy(recons, input, reduction='none')
                recons_loss = normalizing_weight * recons_loss.sum()
            elif self.reduction == 'sum':
                recons_loss = F.binary_cross_entropy(recons, input, reduction='sum')

        return recons_loss

    def beta_loss_function(self,
                           *args,
                           **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = self.get_reconstruction(recons, input)
        kld_loss = self.get_kld(log_var, mu, input.shape)

        if self.loss_type == 'beta':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + (self.beta * kld_loss)
        elif self.loss_type == 'annealed':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + (self.gamma * (kld_loss - C).abs())
        else:
            raise ValueError('Undefined loss type.')

        return [loss, {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}]

    def geco_loss_function(self,
                           *args,
                           **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = self.get_reconstruction(recons, input)
        kld_loss = self.get_kld(log_var, mu, input.shape)

        loss = self.geco.loss(recons_loss, kld_loss)

        return [loss, {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}]

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

# base class adopted from:
# https://github.com/applied-ai-lab/genesis/blob/master/train.py
class GECO():

    def __init__(self, goal, step_size, alpha=0.99, beta_init=1.0,
                 beta_min=1e-10, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.speedup = speedup

    def to_cuda(self):
        self.beta = self.beta.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0 - self.alpha) * err + self.alpha * self.err_ema
            constraint = (self.goal - self.err_ema)
            if self.speedup is not None and constraint.item() > 0:
                # for numerical stability
                step_size_constraint = self.speedup * self.step_size * constraint
                step_size_constraint = torch.where(step_size_constraint > 50, torch.scalar_tensor(50.).to(err.device),
                                                   step_size_constraint)
                factor = torch.exp(step_size_constraint)
            else:
                # for numerical stability
                step_size_constraint = self.step_size * constraint
                step_size_constraint = torch.where(step_size_constraint > 50, torch.scalar_tensor(50.).to(err.device),
                                                   step_size_constraint)
                factor = torch.exp(step_size_constraint)

            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        return loss
