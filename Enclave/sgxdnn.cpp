#define USE_EIGEN_TENSOR

#include "sgxdnn_main.hpp"

#include "tensor_types.h"

#include "Enclave.h"
#include "Enclave_t.h"

#include "Crypto.h"

using namespace tensorflow;

void ecall_load_model_float(char* model_json, float** filters)
{
	load_model_float(model_json, filters);
}

void ecall_predict_float(float* input, float* output, int batch_size)
{
	predict_float(input, output, batch_size);
}

void ecall_load_model_float_verify(char* model_json, float** filters, int preproc)
{
	load_model_float_verify(model_json, filters, preproc);
}

void ecall_predict_verify_float(float* input, float* output, float** aux_data, int batch_size)
{
	predict_verify_float(input, output, aux_data, batch_size);
}

void ecall_slalom_relu(float *in, float *out, float* blind, int numElements, char* activation) {
	slalom_relu(in, out, blind, numElements, activation);
}

void ecall_slalom_maxpoolrelu(float *in, float *out, float *blind, 
                   			  long int dim_in[4], long int dim_out[4],
                   			  int window_rows, int window_cols,
                   			  int row_stride, int col_stride,
                   			  int is_padding_same) {
	slalom_maxpoolrelu(in, out, blind, dim_in, dim_out, window_rows, window_cols, row_stride, col_stride, is_padding_same);
}

void ecall_slalom_init(int integrity, int privacy, int batch_size) {
	slalom_init(integrity, privacy, batch_size);
}

void ecall_slalom_get_r(float* out, int size) {
	slalom_get_r(out, size);
}

void ecall_slalom_set_z(float* z, float* z_enc, int size) {
	slalom_set_z(z, z_enc, size);
}

void ecall_slalom_blind_input(float* inp, float* out, int size) {
	slalom_blind_input(inp, out, size);
}

void ecall_sgxdnn_benchmarks(int num_threads) {
	sgxdnn_benchmarks(num_threads);
}
