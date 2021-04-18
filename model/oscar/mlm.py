import torch


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    scores = logits == labels 
    return scores