import torch
import torch.nn as nn


class ElementWise(nn.Module):
    def __init__(self, func):
        self.func = func
        super(ElementWise, self).__init__()

        if self.func == "CONCAT":
            pass
        else:
            raise NotImplementedError("Element wise function not implemented.")

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)

        return x
