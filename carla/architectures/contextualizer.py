import torch
from torch import nn


MODELS_DICT = {
    'unsup': ['carlac', 'carlaf', 'darla'],
    'gt': ['gt_encode_concat']
}


def get_contextualizer_model(conditioning, params):

    if "carlac" in conditioning:
        print('This model is used in {} mode'.format(conditioning))
        contextualizer = CARLAC(params['state_encoding_shape'][1], nn.ReLU)

    elif "carlaf" in conditioning:
        print('This model is used in {} mode'.format(conditioning))
        contextualizer = CARLAF(params['state_encoding_shape'][1], params['vae_bottleneck'], nn.ReLU)

    elif "darla" in conditioning:
        print('This model is used in {} mode'.format(conditioning))
        contextualizer = DARLA()

    elif "gt" in conditioning:
        contextualizer = ContextualizerGT(conditioning, params['context_net'])

    else:
        raise NotImplementedError(f'Coniditioning {conditioning} does not exist!')

    return contextualizer


class ContextualizerGT(nn.Module):
    """ Contextualizer using ground truth information"""

    def __init__(self, conditioning, context_net):
        super(ContextualizerGT, self).__init__()
        self.context_net = context_net
        self.conditioning = conditioning
        if 'encode' in conditioning:
            self.use_context_encoder = True
        else:
            self.use_context_encoder = False

    def forward(self, x, context):
        """

        :param x:
        :param context:
        :return:
        """
        if self.use_context_encoder:
            context = self.context_net(context)
        hconcat = torch.cat((x, context), -1)

        return hconcat


class CARLAC(nn.Module):
    """
    CARLA with Concat
    """

    def __init__(self, in_dim, activation):
        super(CARLAC, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

    def forward(self, x, c=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                c : not used (just for interface purposes)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        b, ch, width, height = x.size()
        out = torch.cat((x.reshape(b, -1), c), 1)
        return out


class CARLAF(nn.Module):
    """ Carla with FiLM layer"""

    def __init__(self, in_dim, n_factor, activation):
        super(CARLAF, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.norm = nn.BatchNorm2d(in_dim)
        self.weight = nn.Linear(n_factor, in_dim)
        self.bias = nn.Linear(n_factor, in_dim)

    def forward(self, x, c):
        b, ch, width, height = x.size()
        weight = self.weight(c).unsqueeze(2).unsqueeze(3)
        bias = self.bias(c).unsqueeze(2).unsqueeze(3)
        out = self.norm(x * weight + bias)
        out = out.reshape(b, -1)
        return out


class DARLA(nn.Module):
    """ DARLA"""

    def __init__(self):
        super(DARLA, self).__init__()

    def forward(self, x, c):
        """
            inputs :
                x : input feature maps ( B X C X W X H)
                c: input disentangled features (B X F)
            returns :
                out : disentangled features (B X F)
        """
        return c