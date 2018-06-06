TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

App_CC_Files := relu_slalom.cc relumaxpool_slalom.cc

App_Include_Paths := -I. -I$(TF_INC) -I$(TF_INC)/external/nsync/public

App_C_Flags := -O3 -fPIC $(App_Include_Paths) -march=native

App_Cpp_Flags := $(App_C_Flags) -std=c++11 -shared
App_Cpp_Flags_SGX := $(App_C_Flags) -std=c++11 -shared -DUSE_SGX
App_Link_Flags := -L$(TF_LIB)
#App_Link_Flags += -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0
App_Link_Flags += -ltensorflow_framework

Target := ../lib/slalom_ops.so
Target_SGX := ../lib/slalom_ops_sgx.so

all: $(Target) $(Target_SGX)

$(Target): $(App_CC_Files)
	$(CXX) $(App_Cpp_Flags) $(App_CC_Files) -o $@ $(App_Link_Flags)
	@echo "CXX  <=  $<"
	
$(Target_SGX): $(App_CC_Files)
	$(CXX) $(App_Cpp_Flags_SGX) $(App_CC_Files) -o $@ $(App_Link_Flags)
	@echo "CXX  <=  $<"

clean:
	rm -rf $(Target) $(Target_SGX)
