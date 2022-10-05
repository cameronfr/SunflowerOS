#pragma once

#include <torch/script.h>
#include <android/log.h>
#include <iostream>

void tensorInfo (torch::Tensor t, char *name = "tensor") {
  std::ostringstream oss;
  oss << "Tensor " << name << ": " << t.sizes() << " " << t.dtype() << std::endl;

  __android_log_print(ANDROID_LOG_INFO, "Sunflower tensorInfo", "%s", oss.str().c_str());
};

