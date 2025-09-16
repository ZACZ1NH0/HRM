import torch

class AdamAtan2(torch.optim.AdamW):
    """
    Fallback shim for environments where the CUDA adam_atan2 kernel isn't available.
    Behaves like AdamW with the same signature.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Some repos import with a different capitalization:
# from adam_atan2 import AdamATan2  (note the capital 'T')
AdamATan2 = AdamAtan2  # alias to keep both names valid