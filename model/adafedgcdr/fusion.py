import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    def __init__(self, num_sources: int):
        super().__init__()
        self.num_sources = num_sources
        self.domain_bias = nn.Parameter(torch.zeros(num_sources))
        self.count_scale = nn.Parameter(torch.tensor(1.0))
        self.agreement_scale = nn.Parameter(torch.tensor(1.0))
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, source_vec, source_mask, source_count, history_vec, history_mask):
        count = history_mask.sum(dim=1)
        local = (history_vec * history_mask[..., None]).sum(dim=1) / count.clamp_min(1)[:, None]
        agreement = torch.nn.functional.cosine_similarity(source_vec, local[:, None, :], dim=-1)
        agreement = torch.where(count[:, None] > 0, agreement, torch.zeros_like(agreement))
        logits = self.domain_bias + self.count_scale * torch.log1p(source_count) + self.agreement_scale * agreement
        source_weight = torch.softmax(logits.masked_fill(~source_mask, -1e9), dim=1) * source_mask
        source_weight = source_weight / source_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        source = (source_weight[..., None] * source_vec).sum(dim=1)
        target_weight = count / (count + torch.nn.functional.softplus(self.tau_raw))
        target_weight = torch.where(source_mask.any(dim=1), target_weight, (count > 0).float())
        user_vec = (1 - target_weight[:, None]) * source + target_weight[:, None] * local
        personalized_mask = source_mask.any(dim=1) | (count > 0)
        return user_vec, target_weight, source_weight, personalized_mask
