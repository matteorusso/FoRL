import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from distributions import make_pdtype

from utils import (
    small_convnet,
    flatten_dims,
    unflatten_first_dim,
    flatten_two_dims,
    small_mlp,
)


class NSIPolicyMLP(object):
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

        self.nsn = small_mlp(
            self.ob_space.n,
            nl=self.nl,
            hidsize=self.hidsize,
            last_nl=F.leaky_relu,
            layernormalize=self.layernormalize,
        )
        self.idn = small_mlp(
            2 * self.ob_space.n,
            nl=self.nl,
            hidsize=self.hidsize,
            last_nl=F.leaky_relu,
            layernormalize=self.layernormalize,
        )
        self.vfn = small_mlp(
            self.ob_space.n,
            nl=self.nl,
            hidsize=self.hidsize,
            last_nl=F.leaky_relu,
            layernormalize=self.layernormalize,
        )

        self.nsn_head = torch.nn.Linear(hidsize, feat_dim)
        self.idn_head = torch.nn.Linear(hidsize, pdparamsize)
        self.vfn_head = torch.nn.Linear(hidsize, 2)

        self.param_list_NSN = [
            dict(params=self.nsn.parameters()),
            dict(params=self.nsn_head.parameters()),
        ]
        self.param_list_IDN = [
            dict(params=self.idn.parameters()),
            dict(params=self.idn_head.parameters()),
        ]
        self.param_list_VFN = [
            dict(params=self.vfn.parameters()),
            dict(params=self.vfn_head.parameters()),
        ]

        self.flat_features = None
        self.pd = None
        self.vpred = None
        self.ac = None
        self.ob = None

        self.use_oh = True

    def update_features(self, ob, ac):
        sh = (
            ob.shape
        )  # ob.shape = [nenvs, timestep, H, W, C]. Can timestep > 1 ?
        x = flatten_dims(
            ob, len(self.ob_space.shape)
        )  # flat first two dims of ob.shape and get a shape of [N, H, W, C].
        flat_features = self.get_features(x)  # [N, feat_dim]
        self.flat_features = flat_features

        # Next state predicition <-- NSN
        nsp_logit = self.nsn_head(
            self.nsn(flat_features)
        )  # to be saved for loss
        nsp = F.softmax(nsp_logit, dim=-1)
        # Get current dist. over actions that brought us there <-- IDN
        ac_distr = self.idn_head(
            self.idn(torch.cat([nsp, flat_features], dim=-1))
        )
        # Get current value function prediction <-- VFN
        vfp = self.vfn_head(self.vfn(flat_features))

        self.nsp_logit = nsp_logit
        self.vpred = unflatten_first_dim(vfp, sh)  # [nenvs, timestep, v]
        self.pd = pd = self.ac_pdtype.pdfromflat(ac_distr)
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

    def get_loss_IDN(self):
        ac = self.ac
        sh = ac.shape
        ac = flatten_dims(ac, len(self.ac_space.shape))
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(
            1, torch.tensor(ac).unsqueeze(1), 1
        )  # one_hot(self.ac, self.ac_space.n, axis=2)
        ac = unflatten_first_dim(ac, sh)

        features = self.features.detach()
        next_features = self.next_features
        assert features.shape[:-1] == ac.shape[:-1]
        sh = features.shape

        x = unflatten_first_dim(self.nsp_logit, sh)

        ac_distr = self.idn_head(
            self.idn(
                torch.cat(
                    [
                        flatten_dims(next_features, 1),
                        flatten_dims(features, 1),
                    ],
                    dim=-1,
                )
            )
        )
        # if np.random.rand() > 0.99:
        #     moves = unflatten_first_dim(ac_distr, sh)
        #     print("test")
        #     print(features[0, 0, :].reshape(4, 4))
        #     print(features[0, 1, :].reshape(4, 4))
        #     print(F.softmax(moves[0, 0, :], dim=0))

        sh = ac.shape
        ac = flatten_dims(ac, 1)
        pd = self.ac_pdtype.pdfromflat(ac_distr)
        entropy = pd.entropy().mean()

        ce = F.cross_entropy(
            ac_distr,
            torch.argmax(ac, dim=-1),
            reduction="none",
        )
        return ce.reshape(sh[:-1])

    def get_loss(self):
        ac = self.ac
        sh = ac.shape
        ac = flatten_dims(ac, len(self.ac_space.shape))
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(
            1, torch.tensor(ac).unsqueeze(1), 1
        )  # one_hot(self.ac, self.ac_space.n, axis=2)
        ac = unflatten_first_dim(ac, sh)

        features = self.features.detach()
        next_features = self.next_features
        assert features.shape[:-1] == ac.shape[:-1]
        sh = features.shape

        x = unflatten_first_dim(self.nsp_logit, sh)

        # if np.random.rand() > 0.999:
        #     print("test")
        #     print(next_features[0, 0, :].reshape(4, 4))
        #     print(F.softmax(x, dim=-1)[0, 0, :].reshape(4, 4))
        #     print(next_features[0, 1, :].reshape(4, 4))
        #     print(F.softmax(x, dim=-1)[0, 1, :].reshape(4, 4))
        #     print(next_features[0, 2, :].reshape(4, 4))
        #     print(F.softmax(x, dim=-1)[0, 2, :].reshape(4, 4))
        #     print(next_features[0, 3, :].reshape(4, 4))
        #     print(F.softmax(x, dim=-1)[0, 3, :].reshape(4, 4))
        sh = x.shape
        ce = F.cross_entropy(
            flatten_dims(x, 1),
            torch.argmax(flatten_dims(next_features, 1), dim=-1),
            reduction="none",
        )
        # ce = torch.sum((F.softmax(x, dim=-1) - next_features) ** 2, dim=-1)
        return ce.reshape(sh[:-1])

    def calculate_loss(self, obs, last_obs, acs):
        n_chunks = 4
        n = obs.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        losses = None
        for i in range(n_chunks):
            ob = obs[sli(i)]
            last_ob = last_obs[sli(i)]
            ac = acs[sli(i)]
            self.update_features(ob, ac)
            features = self.get_features(ob)
            self.features = features

            last_features = self.get_features(last_ob)

            self.next_features = torch.cat(
                [features[:, 1:, :], last_features], 1
            )

            loss = self.get_loss()
            if losses is None:
                losses = loss
            else:
                losses = torch.cat((losses, loss), 0)
        return losses.data.numpy()

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
