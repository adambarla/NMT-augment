class Patience:

    def __init__(self, patience, mode, metric):
        self.patience = patience
        self.mode = mode
        self.metric = metric
        self.best = None
        self.counter = 0
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'.")
        if patience < 0:
            raise ValueError("Patience must be at least 0.")

    def should_stop(self, results):
        if self.metric not in results:
            raise ValueError(f"Metric {self.metric} not in results.")
        if (
            self.best is None
            or (self.mode == "min" and results[self.metric] < self.best)
            or (self.mode == "max" and results[self.metric] > self.best)
        ):
            self.best = results[self.metric]
            self.counter = 0
            return False
        self.counter += 1
        return self.counter > self.patience
