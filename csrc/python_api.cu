#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <tuple>

#include "pybind11_register.h"
#include "util.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME moe_test
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    register_apis(m);
}