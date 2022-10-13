import cppyy
import os
print("Cppyy version:", cppyy.__version__)
cppyy.cppexec("""printf("C++ version %ld", __cplusplus);""")

cppyy.gbl.gInterpreter.GetIncludePath()

# Check python version
import sys
sys.version

# Load LibTorch
cppyy.add_library_path("/opt/homebrew/lib/python3.9/site-packages/torch/lib")
# There is a macro named ClassDef in ROOT that is included in cling and conflicts with ClassDef struct in torch. Have to go to torch header torch/csrc/jit/frontend/tree_views.h, rename ClassDef. Prob breaks something in libtorch
# cppyy.gbl.gInterpreter.Declare("#undef ClassDef")
cppyy.add_include_path(os.path.expanduser('~/Downloads/libtorch_macos/include'))
cppyy.gbl.gInterpreter.Declare('#include <torch/script.h>')
cppyy.load_library("libtorch_cpu")

# Test torch tensor
cppyy.cppexec("""
int a = 5;
for (int i = 0; i < 10; i++) {
  torch::Tensor tensor = torch::rand({1000, 1000});
  // print mean
  std::cout << tensor.mean().item<float>() << std::endl;
}
""")

# 1. Initialize StereoKit, get OpenXR running. Load Stereokit as shared lib, compiled with Cmake. Should also see if can redefine function in SK on the fly, w/ change handler (e.g. if change asset loading code, should reload all assets)? No idea what it's going to look like.
# 2. cpp playground where can initialize and render objects. Not sure how to handle frame loop.
