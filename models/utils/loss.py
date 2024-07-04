"""
Customize Loss functions for the model.
"""

import torch.nn as nn
import torch

class RelativePositionLoss(nn.Module):
    """
    Relative Position Loss.

    This loss is based on the observation that the pen-up and pen-down actions are related to the relative position of the strokes.

    The loss is defined as:

    L = -log(1 - sum(p_rel * pen_down))
    
        where p_rel is the probability of the relation and pen_down is the probability of the pen-down action.

    The loss is computed for each stroke in the sequence.

    The loss can be computed in two modes: "mean" or "sum".
    """
    def __init__(self, reduction="mean"):
        super(RelativePositionLoss, self).__init__()
        self.reduction = reduction
        self.relation = ["Above", "Below", "Inside", "NoRel", "Right", "Sub", "Sup"]
        self.relation_idx = [6, 100, 88, 45, 32, 108, 25]

    def forward(self, log_prob, pen_up, target=None):
        B, T, C = log_prob.shape
        
        probs = torch.softmax(log_prob, dim=-1)

        rel_mask = torch.zeros(C, device=log_prob.device)
        rel_mask[self.relation_idx] = 1

        probs_rel = probs * rel_mask

        pen_up = pen_up.unsqueeze(-1)
        pen_down = 1 - pen_up

        probs_rel_pen_down = probs_rel * pen_down
        sum_probs_rel_pen_down = probs_rel_pen_down.sum(dim=-1)

        loss = -torch.log(1 - sum_probs_rel_pen_down + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        
        return loss

class AWPLoss(nn.Module):
    """
    WARNING: EXPERIMENTAL
    TODO: Implement f_prop method.

    
    Align With Purpose Loss.

    This loss is based on the method proposed in the paper
    
    ALIGN WITH PURPOSE: OPTIMIZE DESIRED PROPERTIES IN CTC MODELS WITH A GENERAL PLUG-AND-PLAY FRAMEWORK

    https://arxiv.org/pdf/2307.01715v3
    """

    def __init__(self, blank=0, reduction="mean"):
        super(AWPLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.hinge_loss = nn.HingeEmbeddingLoss(reduction=reduction)  # Hinge loss for alignment

    def f_prop(self, sampled_alignment, mask):
        # TODO: implement mask_pen_up
        enhanced_alignment = sampled_alignment.clone()
        return enhanced_alignment

    def forward(self, log_probs, targets, input_lengths, target_lengths, input_pen_up=None, lambda_val=0.01):
        a = self.sample_alignments(log_probs, N=1)
        a_ = self.f_prop(a, input_pen_up)
        
        original_prob = torch.gather(log_probs, 2, a.unsqueeze(-1)).squeeze(-1)
        enhanced_prob = torch.gather(log_probs, 2, a_.unsqueeze(-1)).squeeze(-1)

        # Compute the loss
        loss = torch.mean(torch.clamp(lambda_val + original_prob - enhanced_prob, min=0))
        return loss

    def sample_alignments(self, log_probs, N):
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs)  # Shape: (B, T, 109)
        probs /= probs.sum(dim=-1, keepdim=True)  # Normalize

        # Initialize tensor to store sampled alignments
        B, T, C = probs.shape
        samples = torch.zeros(B, N, T, dtype=torch.long, device=probs.device)

        # Efficient sampling per timestep across the whole batch
        for t in range(T):
            samples[:, :, t] = torch.multinomial(probs[:, t, :], N, replacement=True)

        return samples