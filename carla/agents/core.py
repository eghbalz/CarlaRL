


import os
import pickle
import scipy.signal
import torch

import torch.nn as nn
import numpy as np

from carla.architectures.utils import get_model, cnn, mlp
from carla.architectures.contextualizer import get_contextualizer_model, MODELS_DICT
from gym.spaces import Discrete
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: vector x, [x0, x1, x2]
    output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class StateContextNet(nn.Module):
    def __init__(self, obs_space, h_sizes_state_net, h_sizes_context_net, bottleneck_dim, conditioning,
                 contextual, activation, kernel_size, stride, use_posterior=False, context_encoder_model_path='',
                 state_encoder_model_path='', normalize_context_vector=False, vae_bottleneck=0,
                 vae_train=False, state_encoder_freeze=False, joint_training=False, vae_params={}):
        super().__init__()

        obs_shape = obs_space['image'].shape
        self.use_posterior = use_posterior
        self.conditioning = conditioning
        self.contextual = contextual
        self.h_sizes_context_net = list(h_sizes_context_net)
        self.h_sizes_state_net = list(h_sizes_state_net)
        self.vae_model_path = context_encoder_model_path
        self.state_encoder_model_path = state_encoder_model_path
        self.normalize_context_vector = normalize_context_vector
        self.h_bottleneck = bottleneck_dim
        self.vae_bottleneck = vae_bottleneck
        self.vae_train = vae_train
        self.state_encoder_freeze = state_encoder_freeze
        self.vae_params = vae_params
        self.joint_training=joint_training

        self.gt = False

        assert len(h_sizes_state_net) >= 1

        if conditioning in MODELS_DICT['unsup']:
            # skip the last linear layer
            bottleneck = False
        else:
            # use the last linear layer as bottleneck
            print('Using linear bottleneck layer after CNN')
            bottleneck = True

        if len(obs_shape) == 1:
            self.state_net = mlp([obs_shape[0]] + self.h_sizes_state_net, activation, output_activation=activation)

            # estimate the output shape of state_net
            dummy_out = self.state_net(torch.rand(2, *obs_shape))
            self.state_encoding_shape = dummy_out.shape
        else:
            self.state_net = cnn([3] + self.h_sizes_state_net, obs_shape, self.h_bottleneck, activation, kernel_size,
                                 stride, output_activation=activation, bottleneck=bottleneck)

            # estimate the output shape of state_net
            dummy_out = self.state_net(torch.rand(2, obs_shape[2], obs_shape[0], obs_shape[1]))
            self.state_encoding_shape = dummy_out.shape

        # setting up context net
        if contextual:
            self.state_feat_dim = np.prod(self.state_encoding_shape[1:])
            if conditioning in MODELS_DICT['unsup']:

                # self.use_context_encoder = False
                params = {'state_encoding_shape': self.state_encoding_shape, 'vae_bottleneck': self.vae_bottleneck,
                          'h_bottleneck': self.h_bottleneck, 'state_feat_dim': self.state_feat_dim}

                self.contextualizer = get_contextualizer_model(conditioning, params)

            elif conditioning in MODELS_DICT['gt']:
                self.contextual = False
                self.gt = True

                hidden_dims = [obs_space['context'].shape[0]] + self.h_sizes_context_net
                self.context_net = mlp(hidden_dims, activation, output_activation=nn.Identity,
                                       normalize_input=normalize_context_vector)
                self.contextualizer = get_contextualizer_model(conditioning,
                                                               {'context_net': self.context_net})
            else:
                raise NotImplementedError(f'Conditioning "{conditioning}" not implemented')

        else:
            # self.use_context_encoder = False
            self.h_sizes_context_net = [0]

        # this function loads and sets the weights if the vae model path is set.
        self.set_vae_weights()

        # infer output dimensionality
        dummy_input = {}
        for key in obs_space:
            dummy_input[key] = torch.rand(2, *obs_space[key].shape)

        self.output_dim = self.forward(dummy_input).shape[-1]

    def set_vae_weights(self):

        if self.contextual and self.vae_model_path != '':
            model_path = os.path.join(self.vae_model_path, 'model.pt')
            args_path = os.path.join(self.vae_model_path, 'args.pkl')

            with open(args_path, 'rb') as handle:
                args_dict = pickle.load(handle)

            self.context_encoder = get_model(**args_dict)
            self.context_encoder.load_state_dict(torch.load(model_path))
        elif self.contextual and self.joint_training:
            args_dict = self.vae_params
            self.context_encoder = get_model(**args_dict)

        if self.state_encoder_model_path != '':
            model_path = os.path.join(self.state_encoder_model_path, 'model.pt')
            args_path = os.path.join(self.state_encoder_model_path, 'args.pkl')

            with open(args_path, 'rb') as handle:
                args_dict = pickle.load(handle)
            assert 'vae_beta' in args_dict.keys()
            self.state_net = get_model(**args_dict)
            self.state_net.load_state_dict(torch.load(model_path))

        self.freeze_vae()
        self.set_eval_vae()

    def freeze_vae(self):
        if self.contextual and hasattr(self, "context_encoder") and not self.vae_train:
            for param in self.context_encoder.parameters():
                param.requires_grad = False

        if self.state_encoder_model_path != '' and self.state_encoder_freeze:
            for param in self.state_net.parameters():
                param.requires_grad = False

    def set_eval_vae(self):
        if self.contextual and hasattr(self, "context_encoder") and not self.vae_train:
            self.context_encoder.eval()

        if self.state_encoder_model_path != '' and self.state_encoder_freeze:
            self.state_net.eval()

    def unfreeze_vae(self):
        print('Unfreezing context encoder weights!')
        for param in self.context_encoder.parameters():
            param.requires_grad = True

    def forward(self, obs):

        inp = obs['image'].permute(0, 3, 1, 2)
        state_features = self.state_net(inp)

        if self.contextual:
            hcontext_mu, hcontext_log_var = self.context_encoder.encode(inp)

            if self.use_posterior:
                hcontext = self.context_encoder.reparameterize(hcontext_mu, hcontext_log_var)
            else:
                hcontext = hcontext_mu

            state_features = self.contextualizer(state_features, hcontext)
            state_features = state_features.reshape(state_features.shape[0],
                                                    -1)  # flattening might not be necessary anymore

        if hasattr(self, 'gt') and self.gt:
            state_features = self.contextualizer(state_features, obs['context'])

        return state_features


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes_policy_net=(64), hidden_sizes_state_net=(64),
                 hidden_sizes_context_net=(), hidden_bottleneck=256, conditioning='none', contextual=False,
                 activation=nn.Tanh, kernel_size=4, stride=2, use_posterior=False,
                 context_encoder_model_path='', state_encoder_model_path='', normalize_context_vector=False,
                 vae_bottleneck=0, vae_train=False, state_encoder_freeze=False, joint_training=False, vae_params={}):
        super().__init__()

        # only allow discrete action space (for now)
        assert isinstance(action_space, Discrete)

        self.joint_training = joint_training

        # channels are assumed to be in the last dimension
        self.state_context_net = StateContextNet(observation_space, hidden_sizes_state_net, hidden_sizes_context_net,
                                                 hidden_bottleneck, conditioning, contextual, activation, kernel_size,
                                                 stride, use_posterior, context_encoder_model_path,
                                                 state_encoder_model_path, normalize_context_vector, vae_bottleneck,
                                                 vae_train, state_encoder_freeze, joint_training=joint_training,
                                                 vae_params=vae_params)

        self.policy_logits_net = mlp(
            [self.state_context_net.output_dim] + list(hidden_sizes_policy_net) + [action_space.n],
            activation)
        self.policy_v_net = mlp([self.state_context_net.output_dim] + list(hidden_sizes_policy_net) + [1], activation)

    def init_weights(self, init_fn):
        self.apply(init_fn)
        self.state_context_net.set_vae_weights()

    def set_eval_context_net(self):
        self.state_context_net.set_eval_vae()

    def freeze_context_net(self):
        self.state_context_net.freeze_vae()

    def step(self, obs):

        with torch.no_grad():
            encoded_state = self.state_encoding(obs)
            pi = self._distribution(encoded_state)
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)
            v = self.v(encoded_state)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def act_and_val(self, obs):
        return self.step(obs)[:2]

    def state_encoding(self, obs):
        return self.state_context_net(obs)

    def _distribution(self, encoded_state):

        logits = self.policy_logits_net(encoded_state)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def pi(self, encoded_state, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(encoded_state)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def v(self, encoded_state):
        val = torch.squeeze(self.policy_v_net(encoded_state), -1)  # Critical to ensure v has right shape.
        return val

    def forward(self, obs, act=None):

        encoded_state = self.state_encoding(obs)
        pi, logp_a = self.pi(encoded_state, act)
        v = self.v(encoded_state)

        return pi, logp_a, v

    def set_gamma(self, gamma):
        if hasattr(self.state_context_net, 'contextualizer'):
            if hasattr(self.state_context_net.contextualizer, 'set_gamma'):
                self.state_context_net.contextualizer.set_gamma(gamma)
