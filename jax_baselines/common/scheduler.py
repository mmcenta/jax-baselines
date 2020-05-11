

class LinearDecay:
    def __init__(self, initial_value, final_value, final_step):
        self.initial_value = initial_value
        self.final_value = final_value
        self.final_step = final_step
        self.coef = (final_value - initial_value) / float(final_step)

    def __call__(self, t):
        return self.initial_value + self.coef * t

    def get_parameters(self):
        return {
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'final_step': self.final_step,
        }
