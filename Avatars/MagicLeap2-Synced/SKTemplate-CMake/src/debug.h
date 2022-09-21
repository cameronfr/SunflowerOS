#include <torch/script.h>

void tensorInfo (torch::Tensor t, char *name = "tensor") {
  std::cout << "Tensor " << name << ": " << t.sizes() << " " << t.dtype() << std::endl;
};