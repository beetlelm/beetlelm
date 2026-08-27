class L2OnsetDetector:
    """
    Detects when L2 sampling probability crosses threshold.
    """

    def __init__(self, data_mix, threshold=0.05):
        self.data_mix = data_mix
        self.threshold = threshold
        self.triggered = False

    def check(self):

        l2_prob = self.data_mix.current_l2_probability()

        if not self.triggered and l2_prob >= self.threshold:
            self.triggered = True
            return True

        return False
