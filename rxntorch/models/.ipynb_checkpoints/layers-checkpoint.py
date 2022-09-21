import math
import torch.nn as nn

class Linear(nn.Linear):
    def reset_parameters(self):
        stdev = min(1.0 / math.sqrt(self.weight.shape[0]), 0.1)
        nn.init.normal_(self.weight, std=stdev)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            nn.init.constant_(self.bias, 0.0)
