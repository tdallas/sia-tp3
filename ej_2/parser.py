import pandas as pd

class Parser():
    def __init__(self, input_path, output_path):
        self.inputs = self.parse_inputs(input_path)
        self.outputs = self.parse_outputs(output_path)

    def parse_inputs(self, inputs_path):
        return pd.read_csv(inputs_path).values

    def parse_outputs(self, outputs_path):
        return pd.read_csv(outputs_path).values

    def get_outputs(self):
        return self.outputs
    
    def get_inputs(self):
        return self.inputs
