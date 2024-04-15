from torch.optim.lr_scheduler import LRScheduler


class AIAYNScheduler(LRScheduler):

    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        scale = self.d_model ** (-0.5) * min(
            step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]
