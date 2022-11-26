from torch.optim.lr_scheduler import _LRScheduler
import warnings
import math

class STLR(_LRScheduler):

    """
    SLANTED TRIANGULAR LEARNING RATE
    Linearly increases the learning rate first and then linearly decays it 
    according to a schedule specified in the following paper:
    https://arxiv.org/pdf/1801.06146.pdf
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False, num_iters=0, cut_frac=0, ratio=0):

        self.num_iters = num_iters
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.cut = math.floor(num_iters * self.cut_frac)
        super(STLR, self).__init__(optimizer, last_epoch, verbose)


    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self._step_count < self.cut:
            p = self._step_count / self.cut
        else:
            p = 1 - ((self._step_count - self.cut)/(self.cut*(1/self.cut_frac - 1)))

        factor = (1 + p*(self.ratio - 1))/self.ratio

        return [max_lr * factor
                for max_lr in self.base_lrs]









