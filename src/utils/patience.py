class Patience:

    def __init__(self, patience, mode, metric):
        self.patience = patience
        self.mode = mode
        self.metric = metric
        self.best = None
        self.counter = 0

    def should_stop(self, results):
        if (
            self.best is None
            or (self.mode == "min" and self.best is results[self.metric] < self.best)
            or (self.mode == "max" and self.best is results[self.metric] > self.best)
        ):
            self.best = results[self.metric]
            self.counter = 0
            return False
        self.counter += 1
        return self.counter > self.patience
