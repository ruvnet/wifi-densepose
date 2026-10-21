import torch
from torch.nn import Module


class DemoModule(Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(16 * 28 * 28, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)


traced_script_module = torch.jit.script(DemoModule())
traced_script_module.save("model.pt")
