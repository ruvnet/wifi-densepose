#include "torch_python.h"

bool thp_variable_check(PyObject* pyobj) {
  PROTECT(
      return THPVariable_Check(pyobj);
  )
  return false;
}

PyObject* thp_variable_wrap(tensor t) {
  PROTECT(
      return THPVariable_Wrap(*t);
  )
  return nullptr;
}

tensor thp_variable_unpack(PyObject* pyobj) {
  PROTECT(
      return new torch::Tensor(THPVariable_Unpack(pyobj));
  )
  return nullptr;
}
