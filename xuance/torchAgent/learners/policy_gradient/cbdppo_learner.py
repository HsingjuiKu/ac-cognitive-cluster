from xuance.torchAgent.learners import *
from xuance.state_categorizer import StateCategorizer
import numpy as np
import torch


class CBDPPO_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_range: float = 0.25,
                 clip_grad_norm: float = 0.25,
                 use_grad_clip: bool = True,
                 ):
        super(CBDPPO_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.clip_grad_norm = clip_grad_norm
        self.use_grad_clip = use_grad_clip

    def update(self, obs_batch, act_batch, ret_batch, value_batch, adv_batch, old_logp,index):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        value_batch = torch.as_tensor(value_batch, device=self.device)
        adv_batch = torch.as_tensor(adv_batch, device=self.device)
        old_logp_batch = torch.as_tensor(old_logp, device=self.device)

        outputs, a_dist, v_pred = self.policy([obs_batch,0])
        log_prob = a_dist.log_prob(act_batch)

        # 创建与 obs_batch 大小相同的零张量，用于存储 v_pred 值
        v_pred_subcritic = torch.zeros(obs_batch.shape[0],device=self.device)
        
        # 获取 index 中的唯一类别
        unique_indices = torch.unique(index)
        
        # 对每个唯一类别进行迭代
        for i in unique_indices:
            # 获取对应类别的子集
            sub_obs = obs_batch[index == i, :]
            if sub_obs.shape[0] != 0:
                # 计算 v_pred
                _, _, v_pred = self.policy([sub_obs, int(i + 1)])
                # 将 v_pred 赋值到对应位置
                v_pred_subcritic[index == i] = v_pred.squeeze()
                # print(v_pred_subcritic)
        beta_dynamic = min(0 + 1/2500000 * self.iterations, 1)
 
        v_pred_combined = beta_dynamic * v_pred_subcritic + (1-beta_dynamic) * v_pred_original
        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -torch.minimum(surrogate1, surrogate2).mean()

        c_loss = F.mse_loss(v_pred_combined, ret_batch)

        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]
        
        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred.mean().item(),
            "clip_ratio": cr
        }

        return info
