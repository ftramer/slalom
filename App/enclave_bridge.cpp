
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

#include "sgx_urts.h"
#include "Enclave_u.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define TOKEN_FILENAME   "enclave.token"
#define ENCLAVE_FILENAME "enclave.signed.so"

using namespace std::chrono;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

thread_local std::chrono::time_point<std::chrono::high_resolution_clock> start;

void ocall_start_clock()
{
	start = std::chrono::high_resolution_clock::now();
}

void ocall_end_clock(const char * str)
{
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    printf(str, elapsed.count());
}

double ocall_get_time()
{
    auto now = std::chrono::high_resolution_clock::now();
	return time_point_cast<microseconds>(now).time_since_epoch().count();
}


extern "C"
{

    /*
     * Initialize the enclave
     */
    unsigned long int initialize_enclave(void)
    {

        std::cout << "Initializing Enclave..." << std::endl;

        sgx_enclave_id_t eid = 0;
        sgx_launch_token_t token = {0};
        sgx_status_t ret = SGX_ERROR_UNEXPECTED;
        int updated = 0;

        /* call sgx_create_enclave to initialize an enclave instance */
        /* Debug Support: set 2nd parameter to 1 */
        ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }

        std::cout << "Enclave id: " << eid << std::endl;

        return eid;
    }

    /*
     * Destroy the enclave
     */
    void destroy_enclave(unsigned long int eid)
    {
        std::cout << "Destroying Enclave with id: " << eid << std::endl;
        sgx_destroy_enclave(eid);
    }

    void load_model_float(unsigned long int eid, char* model_json, float** filters) {
    	sgx_status_t ret = ecall_load_model_float(eid, model_json, filters);
    	if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

    void predict_float(unsigned long int eid, float* input, float* output, int batch_size) {

		for (int i=0; i<batch_size; i++) {
			//sgx_status_t ret = ecall_predict_float(eid, input, output, batch_size);
			sgx_status_t ret = ecall_predict_float(eid, input, output, 1);
			printf("predict returned!\n");
			if (ret != SGX_SUCCESS) {
				print_error_message(ret);
				throw ret;
			}
		}
        printf("returning...\n");
	}

    void load_model_float_verify(unsigned long int eid, char* model_json, float** filters, bool preproc) {
		sgx_status_t ret = ecall_load_model_float_verify(eid, model_json, filters, preproc);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void predict_verify_float(unsigned long int eid, float* input, float* output, float** aux_data, int batch_size) {

		sgx_status_t ret = ecall_predict_verify_float(eid, input, output, aux_data, batch_size);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
		printf("returning...\n");
	}

	void slalom_relu(unsigned long int eid, float* input, float* output, float* blind, int num_elements, char* activation) {
		sgx_status_t ret = ecall_slalom_relu(eid, input, output, blind, num_elements, activation);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void slalom_maxpoolrelu(unsigned long int eid, float* input, float* output, float* blind, 
							long int dim_in[4], long int dim_out[4],
                            int window_rows, int window_cols, 
							int row_stride, int col_stride, 
							bool same_padding)
	{
		sgx_status_t ret = ecall_slalom_maxpoolrelu(eid, input, output, blind, 
										  			dim_in, dim_out, 
											 		window_rows, window_cols, 
											 		row_stride, col_stride,
													same_padding);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void slalom_init(unsigned long int eid, bool integrity, bool privacy, int batch_size) {
		sgx_status_t ret = ecall_slalom_init(eid, integrity, privacy, batch_size);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void slalom_get_r(unsigned long int eid, float* out, int size) {
		sgx_status_t ret = ecall_slalom_get_r(eid, out, size);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void slalom_set_z(unsigned long int eid, float* z, float* z_enc, int size) {
		sgx_status_t ret = ecall_slalom_set_z(eid, z, z_enc, size);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void slalom_blind_input(unsigned long int eid, float* inp, float* out, int size) {
		sgx_status_t ret = ecall_slalom_blind_input(eid, inp, out, size);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}

	void sgxdnn_benchmarks(unsigned long int eid, int num_threads) {
		sgx_status_t ret = ecall_sgxdnn_benchmarks(eid, num_threads);
		if (ret != SGX_SUCCESS) {
			print_error_message(ret);
			throw ret;
		}
	}
}

/* Application entry */
int main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    try {
        sgx_enclave_id_t eid = initialize_enclave();

        std::cout << "Enclave id: " << eid << std::endl;
		
		const unsigned int filter_sizes[] = {3*3*3*64, 64, 
											3*3*64*64, 64, 
											3*3*64*128, 128, 
											3*3*128*128, 128, 
											3*3*128*256, 256, 
											3*3*256*256, 256, 
											3*3*256*256, 256, 
											3*3*256*512, 512, 
											3*3*512*512, 512, 
											3*3*512*512, 512, 
											3*3*512*512, 512, 
											3*3*512*512, 512, 
											3*3*512*512, 512, 
											7 * 7 * 512 * 4096, 4096,
											4096 * 4096, 4096,
											4096 * 1000, 1000};

		float** filters = (float**) malloc(2*16*sizeof(float*));
        for (int i=0; i<2*16; i++) {
			filters[i] = (float*) malloc(filter_sizes[i] * sizeof(float));
		}

		const unsigned int output_sizes[] = {224*224*64,
                                             224*224*64, 
                                             112*112*128, 
                                             112*112*128, 
                                             56*56*256, 
                                             56*56*256, 
                                             56*56*256, 
                                             28*28*512, 
                                             28*28*512, 
                                             28*28*512, 
                                             14*14*512, 
                                             14*14*512, 
                                             14*14*512, 
											 4096,
											 4096,
											 1000};

		float** extras = (float**) malloc(16*sizeof(float*));
		for (int i=0; i<16; i++) {
			extras[i] = (float*) malloc(output_sizes[i] * sizeof(float));
		}

        float* img = (float*) malloc(224 * 224 * 3 * sizeof(float));
        float* output = (float*) malloc(1000 * sizeof(float));
		printf("filters initalized\n");		

		std::ifstream t("App/vgg16.json");
		std::stringstream buffer;
		buffer << t.rdbuf();
		std::cout << buffer.str() << std::endl;
		printf("Loading model...\n");
		load_model_float(eid, (char*)buffer.str().c_str(), filters);
		printf("Model loaded!\n");

		for(int i=0; i<4; i++) {
			auto start = std::chrono::high_resolution_clock::now();
			//predict_float(eid, img, output, 1);
			predict_verify_float(eid, img, output, extras, 1);
			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
			printf("predict required %4.2f sec\n", elapsed.count());
		}
        printf("Enter a character to destroy enclave ...\n");
        getchar();

        // Destroy the enclave
        sgx_destroy_enclave(eid);

        printf("Info: Enclave Launcher successfully returned.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return 0;
    }
    catch (int e)
    {
        printf("Info: Enclave Launch failed!.\n");
        printf("Enter a character before exit ...\n");
        getchar();
        return -1;
    }
}
