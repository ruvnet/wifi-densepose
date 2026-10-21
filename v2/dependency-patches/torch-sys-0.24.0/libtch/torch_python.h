#ifndef __TORCH_PYTHON_H__
#define __TORCH_PYTHON_H__

#include "torch_api.h"
#include<torch/csrc/autograd/python_variable.h>

#ifdef __cplusplus
extern "C" {
#endif

bool thp_variable_check(PyObject*);
PyObject* thp_variable_wrap(tensor);
tensor thp_variable_unpack(PyObject*);

#ifdef __cplusplus
};
#endif

#endif
