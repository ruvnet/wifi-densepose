# This generates some test weights for a linear layer, as well as a sample input/output pair. This
# is used to demonstrate that the implementing a linear layer with `Tensor::matmul()` produces
# outputs that are close to, but not exactly equal to, the outputs produced by `Tensor::linear()`.
#
# The tensor sizes chosen are as small as I could make them while still reproducing this issue.

import numpy as np
import torch

linear = torch.nn.Linear(257, 4)
input = torch.rand(2, 257)

np.save('ws.npy', linear.weight.data)
np.save('bs.npy', linear.bias.data)
np.save('in.npy', input.data)
np.save('out.npy', linear(input).data)
