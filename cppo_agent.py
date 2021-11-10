import time
from utils import flatten_dims
import torch
import numpy as np
from math_util import explained_variance
from mpi_moments import mpi_moments
from running_mean_std import RunningMeanStd

from rollouts import Rollout
from utils import get_mean_and_std
from vec_env import ShmemVecEnv as VecEnv

class NSIOptimizer(object):
    envs = None

    def __init__(self, *, scope, ob_space, ac_space, stochpol,
                 ent_coef, gamma, lam, nepochs, lr, cliprange,
                 nminibatches,
                 normrew, normadv, use_news, ext_coeff, int_coeff,
                 nsteps_per_seg, nsegs_per_env):
        self.use_recorder = True
        self.n_updates = 0
        self.scope = scope
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.stochpol = stochpol
        self.nepochs = nepochs
        self.lr = lr
        self.cliprange = cliprange
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nminibatches = nminibatches
        self.gamma = gamma
        self.lam = lam
        self.normrew = normrew
        self.normadv = normadv
        self.use_news = use_news
        self.ent_coef = ent_coef
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff

    def start_interaction(self, env_fns, nlump=2):
        self.optimizer_NSN = torch.optim.Adam(self.stochpol.param_list_NSN, lr=self.lr)
        self.optimizer_NSN.zero_grad()

        self.optimizer_IDN = torch.optim.Adam(self.stochpol.param_list_IDN, lr=self.lr)
        self.optimizer_IDN.zero_grad()

        self.optimizer_VFN = torch.optim.Adam(self.stochpol.param_list_VFN, lr=self.lr)
        self.optimizer_VFN.zero_grad()

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        self.envs = [VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for l in range(self.nlump)]

        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=None)

        self.buf_advs_NSN = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets_NSN = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        self.buf_advs_IDN = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets_IDN = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()

    def calculate_advantages(self, is_NSN, rews, use_news, gamma, lam):
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
            if not use_news:
                nextnew = 0

            if is_NSN:
                nextvals = self.rollout.buf_vpreds[:, t + 1, 0] if t + 1 < nsteps else self.rollout.buf_vpred_last[:, 0]
                nextnotnew = 1 - nextnew

                delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t, 0]
                self.buf_advs_NSN[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
            else:
                nextvals = self.rollout.buf_vpreds[:, t + 1, 1] if t + 1 < nsteps else self.rollout.buf_vpred_last[:, 1]
                nextnotnew = 1 - nextnew
                delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t, 1]
                self.buf_advs_IDN[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        if is_NSN:
            self.buf_rets_NSN[:] = self.buf_advs_NSN + self.rollout.buf_vpreds[:, :, 0]
        else:
            self.buf_rets_IDN[:] = self.buf_advs_IDN + self.rollout.buf_vpreds[:, :, 1]

    def update(self):
        # When normalizing fix in RewardForwardFilter
        assert not self.normrew
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews_NSN = np.copy(self.rollout.buf_rews_NSN)
            rews_IDN = np.copy(self.rollout.buf_rews_IDN)

        self.calculate_advantages(rews=rews_NSN, is_NSN = True, use_news=self.use_news, gamma=self.gamma, lam=self.lam)
        self.calculate_advantages(rews=rews_IDN, is_NSN = False, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        info = dict(
            advmean_NSN=self.buf_advs_NSN.mean(),
            advstd_NSN=self.buf_advs_NSN.std(),
            retmean_NSN=self.buf_rets_NSN.mean(),
            retstd_NSN=self.buf_rets_NSN.std(),
            vpredmean_NSN=self.rollout.buf_vpreds[:, :, 0].mean(),
            vpredstd_NSN=self.rollout.buf_vpreds[:, :, 0].std(),
            rew_mean_NSN=np.mean(self.rollout.buf_rews_NSN),
            advmean_IDN=self.buf_advs_IDN.mean(),
            advstd_IDN=self.buf_advs_IDN.std(),
            retmean_IDN=self.buf_rets_IDN.mean(),
            retstd_IDN=self.buf_rets_IDN.std(),
            vpredmean_IDN=self.rollout.buf_vpreds[:, :, 1].mean(),
            vpredstd_IDN=self.rollout.buf_vpreds[:, :, 1].std(),
            rew_mean_IDN=np.mean(self.rollout.buf_rews_IDN)
        )

        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        to_report = {'nsn_loss': 0.0, 'idn_loss': 0.0, 'vfn_loss': 0.0, 'metric': 0.0,}

        # normalize advantages
        if self.normadv:
            # Do it for NSN
            m, s = get_mean_and_std(self.buf_advs_NSN)
            self.buf_advs_NSN = (self.buf_advs_NSN - m) / (s + 1e-7)
            # Do it for IDN
            m, s = get_mean_and_std(self.buf_advs_IDN)
            self.buf_advs_IDN = (self.buf_advs_IDN - m) / (s + 1e-7)
        
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]

                acs = self.rollout.buf_acs[mbenvinds]
                rews = self.rollout.buf_rews[mbenvinds]
                metric = self.rollout.metric[mbenvinds]
                # if np.random.rand() >= 0.95:
                #     print("No. steps to goal/OPT =", np.mean(rews)*6)
                vpreds = self.rollout.buf_vpreds[mbenvinds]
                nlps = self.rollout.buf_nlps[mbenvinds]
                obs = self.rollout.buf_obs[mbenvinds]
                rets_NSN = self.buf_rets_NSN[mbenvinds]
                rets_IDN = self.buf_rets_IDN[mbenvinds]
                advs_NSN = self.buf_advs_NSN[mbenvinds]
                advs_IDN = self.buf_advs_IDN[mbenvinds]
                last_obs = self.rollout.buf_obs_last[mbenvinds]

                lr = self.lr
                cliprange = self.cliprange

                self.stochpol.update_features(obs, acs)
                features = self.stochpol.get_features(obs)
                self.stochpol.features = features
                last_features = self.stochpol.get_features(last_obs)
                self.stochpol.next_features = torch.cat(
                    [features[:, 1:, :], last_features], 1
                )

                acs = torch.tensor(flatten_dims(acs, len(self.ac_space.shape)))
                neglogpac = self.stochpol.pd.neglogp(acs)
                entropy = torch.mean(self.stochpol.pd.entropy())
                vpred = self.stochpol.vpred
                vf_nsn_loss = 0.5 * torch.mean((vpred[:, :, 0].squeeze() - torch.tensor(rets_NSN)) ** 2)
                vf_idn_loss = 0.5 * torch.mean((vpred[:, :, 1].squeeze() - torch.tensor(rets_IDN)) ** 2)
                VFN_loss = vf_nsn_loss + vf_idn_loss


                nlps = torch.tensor(flatten_dims(nlps, 0))
                ratio = torch.exp(nlps - neglogpac.squeeze())

                # Advantages for NSN and IDN
                advs_NSN = flatten_dims(advs_NSN, 0)
                advs_IDN = flatten_dims(advs_IDN, 0)
                negadv_NSN = torch.tensor(- advs_NSN)
                negadv_IDN = torch.tensor(- advs_IDN)

                # Losses for NSN
                pg_losses1_NSN = negadv_NSN * ratio
                pg_losses2_NSN = negadv_NSN * torch.clamp(ratio, min = 1.0 - cliprange, max = 1.0 + cliprange)
                pg_loss_surr_NSN = torch.max(pg_losses1_NSN, pg_losses2_NSN)
                pg_loss_NSN = torch.mean(pg_loss_surr_NSN)

                # Losses for IDN
                pg_losses1_IDN = negadv_IDN * ratio
                pg_losses2_IDN = negadv_IDN * torch.clamp(ratio, min = 1.0 - cliprange, max = 1.0 + cliprange)
                pg_loss_surr_IDN = torch.max(pg_losses1_IDN, pg_losses2_IDN)
                pg_loss_IDN = torch.mean(pg_loss_surr_IDN)

                ent_loss = (- self.ent_coef) * entropy

                approxkl = 0.5 * torch.mean((neglogpac - nlps) ** 2)

                NSN_loss = pg_loss_NSN #+ ent_loss
                self.optimizer_NSN.zero_grad()
                NSN_loss.backward(retain_graph=True) # otherwise we cannot compute gradients for IDN
                self.optimizer_NSN.step()

                idn_params = list(self.stochpol.idn.parameters()) + list(self.stochpol.idn_head.parameters())
                IDN_loss = pg_loss_IDN + ent_loss
                self.optimizer_IDN.zero_grad()
                IDN_loss.backward(inputs=idn_params)
                self.optimizer_IDN.step()

                self.optimizer_VFN.zero_grad()
                VFN_loss.backward()
                self.optimizer_VFN.step()

                to_report['nsn_loss'] += NSN_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['idn_loss'] += IDN_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['vfn_loss'] += VFN_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['metric'] += np.mean(metric) / (self.nminibatches * self.nepochs)

        info.update(to_report)
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update) # MPI.COMM_WORLD.Get_size() * 
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.get_var_values()

    def set_var_values(self, vv):
        self.stochpol.set_var_values(vv)

class PpoOptimizer(object):
    envs = None

    def __init__(self, *, scope, ob_space, ac_space, stochpol,
                 ent_coef, gamma, lam, nepochs, lr, cliprange,
                 nminibatches,
                 normrew, normadv, use_news, ext_coeff, int_coeff,
                 nsteps_per_seg, nsegs_per_env, dynamics):
        self.dynamics = dynamics
        self.use_recorder = True
        self.n_updates = 0
        self.scope = scope
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.stochpol = stochpol
        self.nepochs = nepochs
        self.lr = lr
        self.cliprange = cliprange
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nminibatches = nminibatches
        self.gamma = gamma
        self.lam = lam
        self.normrew = normrew
        self.normadv = normadv
        self.use_news = use_news
        self.ent_coef = ent_coef
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff

    def start_interaction(self, env_fns, dynamics, nlump=2):
        param_list = self.stochpol.param_list + self.dynamics.param_list + self.dynamics.auxiliary_task.param_list # copy a link, not deepcopy.
        self.optimizer = torch.optim.Adam(param_list, lr=self.lr)
        self.optimizer.zero_grad()

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        self.envs = [VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for l in range(self.nlump)]

        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=dynamics)

        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_news, gamma, lam):
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
            if not use_news:
                nextnew = 0
            nextvals = self.rollout.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_last
            nextnotnew = 1 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t]
            self.buf_advs[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds

    def update(self):
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.rollout.buf_rews)
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        info = dict(
            advmean=self.buf_advs.mean(),
            advstd=self.buf_advs.std(),
            retmean=self.buf_rets.mean(),
            retstd=self.buf_rets.std(),
            vpredmean=self.rollout.buf_vpreds.mean(),
            vpredstd=self.rollout.buf_vpreds.std(),
            ev=explained_variance(self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel()),
            rew_mean=np.mean(self.rollout.buf_rews),
            recent_best_ext_ret= self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        to_report = {'total': 0.0, 'pg': 0.0, 'vf': 0.0, 'ent': 0.0, 'approxkl': 0.0, 'clipfrac': 0.0, 'aux': 0.0, 'dyn_loss': 0.0, 'feat_var': 0.0}

        # normalize advantages
        if self.normadv:
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]

                acs = self.rollout.buf_acs[mbenvinds]
                rews = self.rollout.buf_rews[mbenvinds]
                # if np.random.rand() >= 0.95:
                #     print("No. steps to goal/OPT =", np.mean(rews)*6)
                vpreds = self.rollout.buf_vpreds[mbenvinds]
                nlps = self.rollout.buf_nlps[mbenvinds]
                obs = self.rollout.buf_obs[mbenvinds]
                rets = self.buf_rets[mbenvinds]
                advs = self.buf_advs[mbenvinds]
                last_obs = self.rollout.buf_obs_last[mbenvinds]

                lr = self.lr
                cliprange = self.cliprange

                self.stochpol.update_features(obs, acs)
                self.dynamics.auxiliary_task.update_features(obs, last_obs)
                self.dynamics.update_features(obs, last_obs)

                feat_loss = torch.mean(self.dynamics.auxiliary_task.get_loss())
                dyn_loss = torch.mean(self.dynamics.get_loss())

                acs = torch.tensor(flatten_dims(acs, len(self.ac_space.shape)))
                neglogpac = self.stochpol.pd.neglogp(acs)
                entropy = torch.mean(self.stochpol.pd.entropy())
                vpred = self.stochpol.vpred
                vf_loss = 0.5 * torch.mean((vpred.squeeze() - torch.tensor(rets)) ** 2)

                nlps = torch.tensor(flatten_dims(nlps, 0))
                ratio = torch.exp(nlps - neglogpac.squeeze())

                advs = flatten_dims(advs, 0)
                negadv = torch.tensor(- advs)
                pg_losses1 = negadv * ratio
                pg_losses2 = negadv * torch.clamp(ratio, min = 1.0 - cliprange, max = 1.0 + cliprange)
                pg_loss_surr = torch.max(pg_losses1, pg_losses2)
                pg_loss = torch.mean(pg_loss_surr)
                ent_loss = (- self.ent_coef) * entropy

                approxkl = 0.5 * torch.mean((neglogpac - nlps) ** 2)
                clipfrac = torch.mean((torch.abs(pg_losses2 - pg_loss_surr) > 1e-6).float())
                feat_var = torch.std(self.dynamics.auxiliary_task.features.float())

                total_loss = pg_loss + ent_loss + vf_loss + feat_loss + dyn_loss

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                to_report['total'] += total_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['pg'] += pg_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['vf'] += vf_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['ent'] += ent_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['approxkl'] += approxkl.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['clipfrac'] += clipfrac.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['feat_var'] += feat_var.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['aux'] += feat_loss.data.numpy() / (self.nminibatches * self.nepochs)
                to_report['dyn_loss'] += dyn_loss.data.numpy() / (self.nminibatches * self.nepochs)

        info.update(to_report)
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update) # MPI.COMM_WORLD.Get_size() * 
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.get_var_values()

    def set_var_values(self, vv):
        self.stochpol.set_var_values(vv)


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
