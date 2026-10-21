## Custom Optimizer - Sparse Adam optimizer

This example implements an Adam optimizer for sparse and dense gradients. This
is useful for large embedding matrices, where the gradient is very sparse.
Instead of updating the whole embedding, just a small portion of the matrix is
updated and therefore a significant speed-up in training time is gained. 

For the dense update step it uses the `addcdiv_` (fraction and then add assign
in-place) and `addcmul_` (multiply and then add assign in-place) functions and
for the sparse part `index_select` and `index_add` to reduce the necessary
work. The sparse gradient update subroutine is only faster for sparse matrices.
There is a `force_sparse` parameter, which enforces that the sparse subroutine
is used, even for dense gradients. This is only for testing purposes, because
the problem is actual of dense nature.

As a toy example, a MNIST classification problem is solved with help of the
Adam implementation. For further information take a look at the [python
implementation](https://github.com/pytorch/pytorch/blob/master/torch/optim/sparse_adam.py)
and at [Adam: A Method for Stochastic
Optimization](https://arxiv.org/abs/1412.6980). The Adam optimizer should reach
97% in about 170 epochs.
