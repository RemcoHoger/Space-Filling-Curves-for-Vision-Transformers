import math


class WarmupCosineScheduler:
    """
    Custom learning rate scheduler with linear warmup followed by cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of training steps.
        min_lr (float): Minimum learning rate at the end of scheduling.
        base_lr (float, optional): Base learning rate.
                                   If None, it's derived from optimizer.
    """

    def __init__(self, optimizer, warmup_steps,
                 total_steps, min_lr=1e-6, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Get base learning rate from optimizer if not provided
        if base_lr is None:
            self.base_lr = optimizer.param_groups[0]['lr']
        else:
            self.base_lr = base_lr

        self.current_step = 0

    def step(self):
        """Update the learning rate based on the current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                max(1, (self.total_steps - self.warmup_steps))
            lr = self.min_lr + 0.5 * \
                (self.base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * min(1.0, progress)))

        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        return lr
