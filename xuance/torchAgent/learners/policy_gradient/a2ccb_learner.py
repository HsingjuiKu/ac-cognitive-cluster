from xuance.torchAgent.learners import *
from xuance.state_categorizer import StateCategorizer
import numpy as np
import torch

class A2CCB_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 clip_grad: Optional[float] = None):
        super(A2CCB_Learner, self).__init__(policy, optimizer, scheduler, device, model_dir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_grad = clip_grad

    def update(self, obs_batch, act_batch, ret_batch, adv_batch,index):
        self.iterations += 1
        
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        adv_batch = torch.as_tensor(adv_batch, device=self.device)
        index = torch.as_tensor(index, device = self.device)
        obs_batch = torch.as_tensor(obs_batch, device = self.device)

        # Get the predict v from original critic net
        outputs, a_dist, v_pred_original = self.policy([obs_batch,0])
        log_prob = a_dist.log_prob(act_batch)
        v_pred_subcritic = []
        times = 0
        
        # if state_categorizer.initialized:
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
        beta_dynamic = min(0 + 0.00001 * times, 1)
        times += 1
        v_pred_combined = beta_dynamic * v_pred_subcritic + (1-beta_dynamic) * v_pred_original
        # else:
        #     v_pred_combined = v_pred_original
            
        a_loss = -(adv_batch * log_prob).mean()
        c_loss = F.mse_loss(v_pred_combined, ret_batch)
        e_loss = a_dist.entropy().mean()

        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "actor-loss": a_loss.item(),
            "critic-loss": c_loss.item(),
            "entropy": e_loss.item(),
            "learning_rate": lr,
            "predict_value": v_pred_combined.mean().item()
        }

        return info

