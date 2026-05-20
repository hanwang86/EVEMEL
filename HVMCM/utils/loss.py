import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, mention_embeds, pos_entity_embeds, neg_entity_embeds=None):
        batch_size = mention_embeds.size(0)
        device = mention_embeds.device

        mention_embeds = F.normalize(mention_embeds, dim=1)
        pos_entity_embeds = F.normalize(pos_entity_embeds, dim=1)
        if neg_entity_embeds is not None:
            neg_entity_embeds = F.normalize(neg_entity_embeds, dim=1)

        logit_scale = self.logit_scale.exp()
        sim_pos = torch.sum(mention_embeds * pos_entity_embeds, dim=1, keepdim=True)
        sim_all = torch.matmul(mention_embeds, pos_entity_embeds.t())
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_in_batch = sim_all[~mask].view(batch_size, -1)

        if neg_entity_embeds is not None:
            num_hard = neg_entity_embeds.size(0) // batch_size
            neg_entity_embeds_reshaped = neg_entity_embeds.view(batch_size, num_hard, -1)
            sim_hard = torch.einsum('bd,bkd->bk', mention_embeds, neg_entity_embeds_reshaped)
            logits = torch.cat([sim_pos, sim_hard, sim_in_batch], dim=1)
        else:
            logits = torch.cat([sim_pos, sim_in_batch], dim=1)

        logits = logits * logit_scale
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.loss_fct(logits, labels)


class IMNContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, score_matrix, hard_neg_scores=None):
        batch_size = score_matrix.size(0)
        device = score_matrix.device

        pos_score = score_matrix.diag().unsqueeze(1)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        in_batch_neg_score = score_matrix[~mask].view(batch_size, -1)

        if hard_neg_scores is not None:
            num_hard = hard_neg_scores.size(1) // batch_size
            reshaped_scores = hard_neg_scores.view(batch_size, batch_size, num_hard)
            aligned_hard_neg_score = reshaped_scores.diagonal(dim1=0, dim2=1).transpose(0, 1)
            logits = torch.cat([pos_score, aligned_hard_neg_score, in_batch_neg_score], dim=1)
        else:
            logits = torch.cat([pos_score, in_batch_neg_score], dim=1)

        logits = logits / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.loss_fct(logits, labels)
