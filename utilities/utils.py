class EarlyStopping:
    """
    Class to stop training early if validation loss doesn't improve after a specified patience.

    Attributes:
        patience (int): The number of epochs to wait after the last improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        trace_func (callable): Function to print progress messages, defaults to `print`.
        counter (int): Counts epochs since the last improvement.
        best_score (float): Best score observed in the monitored metric.
        early_stop (bool): If True, training will be stopped at the next epoch check.
        test_accuracy_max (float): Maximum test accuracy observed.
    """

    def __init__(self, patience=0, verbose=False, delta=0, trace_func=print):
        """
        Initializes the EarlyStopping instance.

        Parameters:
            patience (int): The number of epochs with no improvement after which training will be stopped.
            verbose (bool): Enables output of messages indicating improvement.
            delta (float): Minimum change to qualify as an improvement.
            trace_func (callable): Function used to output messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_accuracy_max = 0

    def __call__(self, test_accuracy):
        """
        Evaluates the current test accuracy and updates the stopping criteria.

        Parameters:
            test_accuracy (float): The current epoch's test accuracy.
        """
        score = test_accuracy

        if self.best_score is None:
            # If no best score is recorded, set current score as best
            self.best_score = score
            self.improvement_detected(test_accuracy)
        elif score < self.best_score + self.delta:
            # No improvement detected, increment the counter
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # Stop training if patience is exceeded
                self.early_stop = True
        else:
            # Improvement detected, reset the counter
            self.best_score = score
            self.improvement_detected(test_accuracy)
            self.counter = 0

    def improvement_detected(self, test_accuracy):
        """
        Handles improvement detection, updating the maximum test accuracy and potentially logging.

        Parameters:
            test_accuracy (float): The current test accuracy.
        """
        if self.verbose:
            self.trace_func(f'Test accuracy increased to {test_accuracy:.6f}, resetting patience.')
        self.test_accuracy_max = test_accuracy
