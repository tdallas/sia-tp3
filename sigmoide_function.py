from math import e


class SigmoideFunction():
    def __init__(self):
        pass

    def evaluate(self, value):
        return 1 / (1 + (e ** (- value)))
