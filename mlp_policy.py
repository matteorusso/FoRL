import numpy as np
import torch
import torch.nn.functional as F
from distributions import make_pdtype

from utils import (
    small_convnet,
    flatten_dims,
    unflatten_first_dim,
    small_mlp,
    flatten_two_dims,
)


class MLPPolicy(object):
    def __init__(
        self,
        ob_space,
        ac_space,
        hidsize,
        ob_mean,
        ob_std,
        feat_dim,
        layernormalize,
        nl,
        scope="policy",
        use_oh=True,
    ):
        if layernormalize:
            print(
                "Warning: policy is operating on top of layer-normed features. It might slow down the training."
            )
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.ac_pdtype = make_pdtype(ac_space)

        self.pd = self.vpred = None
        self.hidsize = hidsize
        self.feat_dim = feat_dim
        self.scope = scope
        pdparamsize = self.ac_pdtype.param_shape()[0]

        self.features_model = small_mlp(
            self.ob_space,
            nl=self.nl,
            feat_dim=self.feat_dim,
            last_nl=None,
            layernormalize=self.layernormalize,
        )

        self.pd_hidden = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidsize),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidsize, hidsize),
            # torch.nn.ReLU(),
        )
        self.pd_head = torch.nn.Linear(hidsize, pdparamsize)
        self.vf_head = torch.nn.Linear(hidsize, 1)

        self.param_list = [
            dict(params=self.features_model.parameters()),
            dict(params=self.pd_hidden.parameters()),
            dict(params=self.pd_head.parameters()),
            dict(params=self.vf_head.parameters()),
        ]

        self.flat_features = None
        self.pd = None
        self.vpred = None
        self.ac = None
        self.ob = None

        self.use_oh = use_oh

    def update_features(self, ob, ac):
        sh = (
            ob.shape
        )  # ob.shape = [nenvs, timestep, H, W, C]. Can timestep > 1 ?
        x = flatten_dims(
            ob, len(self.ob_space.shape)
        )  # flat first two dims of ob.shape and get a shape of [N, H, W, C].
        flat_features = self.get_features(x)  # [N, feat_dim]
        self.flat_features = flat_features
        hidden = self.pd_hidden(flat_features)
        pdparam = self.pd_head(hidden)
        vpred = self.vf_head(hidden)
        self.vpred = unflatten_first_dim(vpred, sh)  # [nenvs, tiemstep, v]
        self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
        self.ac = ac
        self.ob = ob

    def get_features(self, x):
        x_has_timesteps = len(x.shape) == 2

        if x_has_timesteps:
            sh = x.shape
            x = flatten_two_dims(x)
        if self.use_oh:
            x = torch.tensor(x).squeeze()
            x = F.one_hot(x, num_classes=self.ob_space.n).float()
        else:
            x = (x - self.ob_mean) / self.ob_std
            x = x[:, None]
            x = torch.tensor(x).float()
            x = self.features_model(x)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob):
        self.update_features(ob, None)
        a_samp = self.pd.sample()
        entropy = self.pd.entropy()
        nlp_samp = self.pd.neglogp(a_samp)
        return (
            a_samp.squeeze().data.numpy(),
            self.vpred.squeeze().data.numpy(),
            nlp_samp.squeeze().data.numpy(),
        )
