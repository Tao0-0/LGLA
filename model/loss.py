from math import cos
from numpy.core.fromnumeric import sort
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
eps = 1e-7 

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
  
 
class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None, num_experts=3, s=30, tau=3.0):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau
        self.num_experts = num_experts
        if self.num_experts <= 2:
            self.tau = 1.0
        
        print('loss num_experts: ', self.num_experts)
        print("cls_num_list: ", cls_num_list)
        print('self.tau: ', self.tau)
        self.cls_num_list = torch.tensor(cls_num_list).cuda()
        
        mask_cls = []
        if self.num_experts == 2:
            mask_cls.append(torch.ones_like(self.cls_num_list).bool())
            print(mask_cls[0])
        else:
            self.region_points = self.get_region_points(self.cls_num_list)
            for i in range(len(self.region_points)):
                mask_cls.append((self.cls_num_list> self.region_points[i][0]) & (self.cls_num_list <= self.region_points[i][1]))
                print('i: ', i)
                print(self.region_points[i][0], self.region_points[i][1])
                print(sum(mask_cls[i]), sum(self.cls_num_list[mask_cls[i]]))
        
        self.mask_cls = mask_cls
        

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def get_region_points(self, cls_num_list): # Divide data sets equally according to the number of samples
        region_num = sum(cls_num_list) // (self.num_experts - 1)
        sort_list, _ = torch.sort(cls_num_list)
        region_points = []
        now_sum = 0
        for i in range(len(sort_list)):
            now_sum += sort_list[i]
            if now_sum > region_num:
                region_points.append(sort_list[i])
                now_sum = 0
        region_points = list(reversed(region_points))
        # assert len(region_points) == self.num_experts - 2

        region_left_right = []

        for i in range(len(region_points)):
            if i == 0:
                region_left_right.append([region_points[i], cls_num_list.max()])
            else:
                region_left_right.append([region_points[i], region_points[i-1]])
        region_left_right.append([0, region_points[len(region_points)-1]])
        # assert len(region_left_right) == self.num_experts - 1

        return region_left_right

    def get_region_points_cls(self, cls_num_list): # Divide data sets equally according to the number of categories
        cnt = self.C_number // (self.num_experts - 1)
        sort_list, _ = torch.sort(cls_num_list)

        region_points = []
        for i in range(1, self.num_experts - 1):
            region_points.append(sort_list[cnt*i-1])
        
        region_points = list(reversed(region_points))
        assert len(region_points) == self.num_experts - 2

        region_left_right = []

        for i in range(len(region_points)):
            if i == 0:
                region_left_right.append([region_points[i], cls_num_list.max()])
            else:
                region_left_right.append([region_points[i], region_points[i-1]])
        region_left_right.append([0, region_points[len(region_points)-1]])
        assert len(region_left_right) == self.num_experts - 1

        return region_left_right

    def cal_loss(self, logits, one_hot):
        logits = torch.log(F.softmax(logits, dim=1))
        loss = -torch.sum(logits * one_hot, dim=1).mean()
        return loss

    def cal_one_hot(self, logits, target, mask, ind):

        one_hot = torch.zeros(logits.size()).cuda()
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        with torch.no_grad():
            W_yi = torch.sum(one_hot * logits, dim=1)
            theta_yi = torch.acos(W_yi/self.s).float().cuda() # B

        delta_theta = torch.zeros_like(theta_yi).cuda()
        
        for i in range(self.num_experts-1): 
            theta = theta_yi[mask[i]].mean()
            # theta = torch.median(theta_yi[mask[i]])
            # theta, _ = torch.mode(theta_yi[mask[i]])
            # theta = theta_yi[mask[i]].min()
            delta_theta[mask[i]] = theta_yi[mask[i]]-theta

        delta_theta = delta_theta.double()
        delta_theta = torch.where(delta_theta<0.0, 1.0, 1.0 + delta_theta) # B
        # delta_theta = delta_theta + 1.0
        delta_theta = delta_theta.float().unsqueeze(1)
        one_hot = one_hot * delta_theta

        return one_hot
 
    
    
    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        target_num = self.cls_num_list[target]
        
        mask = []
        
        for i in range(len(self.region_points)):
            mask.append((target_num> self.region_points[i][0]) & (target_num <= self.region_points[i][1]))
        
        for ind in range(self.num_experts):
            expert_logits = extra_info['logits'][ind]
            one_hot = self.cal_one_hot(expert_logits, target, mask, ind)
            
            if ind != self.num_experts -1:
                prior = torch.zeros_like(self.prior).float().cuda() + self.prior.max()
                prior[self.mask_cls[ind]]=self.prior[self.mask_cls[ind]]
                loss += self.cal_loss(expert_logits + torch.log(prior + 1e-9), one_hot)
            else:
                loss += self.cal_loss(expert_logits + torch.log(self.prior + 1e-9) * self.tau, one_hot)
        return loss
    
 
     