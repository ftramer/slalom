

#define USE_EIGEN_TENSOR

#ifndef USE_SGX
#define EIGEN_USE_THREADS
#else
#include "Enclave.h"
#include "sgx_trts.h"
#endif

#include "sgxdnn_main.hpp"
#include "layers/eigen_maxpool.h"
#include "randpool.hpp"
#include "utils.hpp"
#include "benchmark.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include "model.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <cstring>
#include <deque>

#include "Crypto.h"

using namespace SGXDNN;

// prime P chosen for data blinding. Chosen such that P + P/2 < 2^24
int p_int = (1 << 23) + (1 << 21) + 7;
float p = (float) p_int;
float mid = (float) (p_int / 2);

// prime used for Freivalds checks. Largest prime smaller than 2^24
int p_verif = ((1 << 24) - 3);
double inv_p_verif = 1.0 / p_verif;

// some vectorized constants
__m256 p8f = _mm256_set1_ps(p);
__m256 mid8f = _mm256_set1_ps(mid);
__m256 negmid8f = _mm256_set1_ps(-mid);
__m256 zero8f = _mm256_set1_ps((float)(0));
__m256 inv_shift8f = _mm256_set1_ps((float)(1.0/256));
__m256 six8f = _mm256_set1_ps((float) 6 * 256 * 256);

// unblind data mod p, compute activation and write to output buffer
template <typename F>
inline void unblind(F func, float* inp, float* blind, float* out, int num_elements) {
	for(size_t i = 0; i < num_elements; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &inp[i] );             // blinded input
			const __m256 blind8f = _mm256_load_ps( &blind[i] );   		// blinding factor
			const __m256 sub8f = _mm256_sub_ps(inp8f, blind8f);         // unblinded

			const __m256 if_geq = _mm256_cmp_ps(sub8f, mid8f, 0x0d);    // unblinded >= mid
			const __m256 if_lt = _mm256_cmp_ps(sub8f, negmid8f, 0x01);  // unblinded < -mid
			const __m256 then8f = _mm256_sub_ps(sub8f, p8f);            // unblinded - p
			const __m256 elif8f = _mm256_add_ps(sub8f, p8f);            // unblinded + p
			const __m256 res8f = _mm256_blendv_ps(
										_mm256_blendv_ps(
												sub8f,
												elif8f,
												if_lt),
										then8f,
										if_geq);

			_mm256_stream_ps(&out[i], func(res8f));
    }
}

extern "C" {

	Model<float> model_float;

	bool slalom_privacy;
	bool slalom_integrity;
	int batch_size;
	aes_stream_state producer_PRG;
	aes_stream_state consumer_PRG;
	std::deque<sgx_aes_gcm_128bit_iv_t*> aes_gcm_ivs;
	std::deque<sgx_aes_gcm_128bit_tag_t*> aes_gcm_macs;
	float* temp_buffer;
	float* temp_buffer2;
	Tensor<float, 1> buffer_t;
	Tensor<float, 1> buffer2_t;
	int act_idx;
	bool verbose;
	std::vector<int> activation_idxs;

	#ifdef EIGEN_USE_THREADS
	int n_threads = 1;
	Eigen::ThreadPool pool(n_threads);
	Eigen::ThreadPoolDevice device(&pool, n_threads);
	#endif

	// load a model into the enclave
	void load_model_float(char* model_json, float** filters) {
		model_float.load_model(model_json, filters, false, false);
		#ifdef EIGEN_USE_THREADS
		Eigen::setNbThreads(n_threads);
		#endif
	}

	// load a model in verify-mode
	void load_model_float_verify(char* model_json, float** filters, bool verify_preproc) {
		model_float.load_model(model_json, filters, true, verify_preproc);
		#ifdef EIGEN_USE_THREADS
        Eigen::setNbThreads(n_threads);
        #endif
	}

	// forward pass
	void predict_float(float* input, float* output, int batch_size) {

		array4d input_dims = {batch_size,
							  model_float.input_shape[0],
							  model_float.input_shape[1],
							  model_float.input_shape[2]};

		int input_size = batch_size * model_float.input_shape[0] * model_float.input_shape[1] * model_float.input_shape[2];
		assert(input_size != 0);

		// copy input into enclave
		float* inp_copy = model_float.mem_pool->alloc<float>(input_size);
		std::copy(input, input + input_size, inp_copy);

		auto map_in = TensorMap<float, 4>(inp_copy, input_dims);
		TensorMap<float, 4>* in_ptr = &map_in;

		sgx_time_t start_time;
        sgx_time_t end_time;
        double elapsed;

        start_time = get_time_force();

        // loop over all layers
		for (int i=0; i<model_float.layers.size(); i++) {
			if (TIMING) {
				printf("before layer %d (%s)\n", i, model_float.layers[i]->name_.c_str());
			}

			sgx_time_t layer_start = get_time();
			#ifdef EIGEN_USE_THREADS
			auto temp_output = model_float.layers[i]->apply(*in_ptr, (void*) &device);
			#else
			auto temp_output = model_float.layers[i]->apply(*in_ptr);
			#endif

			in_ptr = &temp_output;

			sgx_time_t layer_end = get_time();
			if (TIMING) {
				printf("layer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
			}
		}

		// copy output outside enclave
		std::copy(((float*)in_ptr->data()), ((float*)in_ptr->data()) + ((int)in_ptr->size()), output);
		model_float.mem_pool->release(in_ptr->data());

		end_time = get_time_force();
        printf("total time: %4.4f sec\n", get_elapsed_time(start_time, end_time));
	}

	// forward pass with verification
	void predict_verify_float(float* input, float* output, float** aux_data, int batch_size) {
		array4d input_dims = {batch_size,
				model_float.input_shape[0],
				model_float.input_shape[1],
				model_float.input_shape[2]
		};

		int input_size = batch_size * model_float.input_shape[0] * model_float.input_shape[1] * model_float.input_shape[2];
		assert(input_size != 0);

		float* inp_copy = model_float.mem_pool->alloc<float>(input_size);
		std::copy(input, input + input_size, inp_copy);

		auto map_in = TensorMap<float, 4>(inp_copy, input_dims);

		TensorMap<float, 4>* in_ptr = &map_in;
		sgx_time_t start_time;
        sgx_time_t end_time;
        double elapsed;

        start_time = get_time_force();

		int linear_idx = 0;
		for (int i=0; i<model_float.layers.size(); i++) {
			if (TIMING) {
				printf("before layer %d (%s)\n", i, model_float.layers[i]->name_.c_str());
			}

			#ifdef USE_SGX
			size_t aux_size = batch_size * model_float.layers[i]->output_size();
			assert(sgx_is_outside_enclave(aux_data[linear_idx], aux_size * sizeof(float)));
			#endif

			sgx_time_t layer_start = get_time();
			#ifdef EIGEN_USE_THREADS
			auto temp_output = model_float.layers[i]->fwd_verify(*in_ptr, aux_data, linear_idx, (void*) &device);
			#else
        	auto temp_output = model_float.layers[i]->fwd_verify(*in_ptr, aux_data, linear_idx);
			#endif

			in_ptr = &temp_output;

			linear_idx += model_float.layers[i]->num_linear();

			sgx_time_t layer_end = get_time();
			if (TIMING) {
				printf("layer %d required %4.4f sec\n", i, get_elapsed_time(layer_start, layer_end));
			}
		}
		
		std::copy(((float*)in_ptr->data()), ((float*)in_ptr->data()) + ((int)in_ptr->size()), output);
		model_float.mem_pool->release(in_ptr->data());

		end_time = get_time_force();
		printf("total time: %4.4f sec\n", get_elapsed_time(start_time, end_time));
	}

	// initialize SLALOM protocol
	void slalom_init(bool integrity, bool privacy, int batch) {
		slalom_privacy = privacy;
		slalom_integrity = integrity;
		batch_size = batch;
		
		// TODO pass max size as a parameter
		buffer_t = Tensor<float, 1>(224 * 224 * 64);
		buffer2_t = Tensor<float, 1>(224 * 224 * 64);
		temp_buffer = buffer_t.data();
		temp_buffer2 = buffer2_t.data();

		printf("SLALOM INIT: PRIVACY = %d, INTEGRITY = %d, batch_size = %d\n", privacy, integrity, batch_size);

		if (slalom_privacy) {
			// create two queues of MAC tags to check encrypted blinding factors stored outside the enclave
			init_PRG(&producer_PRG);
			init_PRG(&consumer_PRG);
			aes_gcm_ivs = std::deque<sgx_aes_gcm_128bit_iv_t*>();
			aes_gcm_macs = std::deque<sgx_aes_gcm_128bit_tag_t*>();
		}

		if (slalom_integrity) {
			//TODO check that the model was loaded correctly
		}

		assert(model_float.layers.size() > 0);

		std::vector<std::shared_ptr<Layer<float>>> new_layers;	
		for (int i=0; i<model_float.layers.size(); i++) {
			if (dynamic_cast<ResNetBlock<float>*>(model_float.layers[i].get()) != nullptr) {
				auto resblock = dynamic_cast<ResNetBlock<float>*>(model_float.layers[i].get());
				for (int j=0; j<resblock->get_path1().size(); j++) {
					new_layers.push_back(resblock->get_path1()[j]);
				}
                for (int j=0; j<resblock->get_path2().size(); j++) {
                    new_layers.push_back(resblock->get_path2()[j]);
                }
			} else {
				new_layers.push_back(model_float.layers[i]);
			}
		}
		model_float.layers.assign(new_layers.begin(), new_layers.end());

		printf("MODEL LAYERS (%lu layers):\n", model_float.layers.size());
		// get the indices of all the activation layers
		for (int i=0; i<model_float.layers.size(); i++) {
			printf("%s\n", model_float.layers[i].get()->name_.c_str());
			if (dynamic_cast<Activation<float>*>(model_float.layers[i].get()) != nullptr) {
				activation_idxs.push_back(i);
			}
		}

		printf("=========\n");
		printf("ACTIVATION IDXS:\n");
		for (int i=0; i<activation_idxs.size(); i++) {
			printf("%d\n", activation_idxs[i]);
		}
		printf("=========\n");

		act_idx = 0;
		verbose = false;
	}

	// blind an input
	void slalom_blind_input(float* inp, float* out, int size) {
		act_idx = 0;
		if (slalom_privacy) {
			int size_per_batch = size/batch_size;
			for(int i=0; i<batch_size; i++) {
				slalom_blind_input_internal(inp+i*size_per_batch, out+i*size_per_batch, size_per_batch, temp_buffer2);
			}
		}
	}

	// blind an input buffer and write to output
	void slalom_blind_input_internal(float* inp, float* out, int size, float* temp) {
		sgx_time_t start_time;
        sgx_time_t end_time;
		double elapsed;

        start_time = get_time();

		int num_bytes = size * sizeof(float);
		get_r(&consumer_PRG, (unsigned char*) temp, num_bytes, 9);

		end_time = get_time();
		elapsed = get_elapsed_time(start_time, end_time);
		if (TIMING) {
			printf("\trandgen of %d bytes required %f sec\n", num_bytes, elapsed);
		}

		TensorMap<float, 1> inp_map(inp, size);
		TensorMap<float, 1> out_map(out, size);
		TensorMap<float, 1> r_map((float*) temp, size);

		if (verbose) {
            Tensor<double, 0> res;
            res = r_map.template cast<double>().minimum();
            printf("min(r) = %f\n", res.data()[0]);
            res = r_map.template cast<double>().maximum();
            printf("max(r) = %f\n", res.data()[0]);
            res = r_map.template cast<double>().abs().sum();
            printf("sum(abs(r)) = %f\n", res.data()[0]);
        }

		start_time = get_time();	

		assert(size % 8 == 0);
        assert ((long int)inp % 32 == 0);
        assert ((long int)temp % 32 == 0);
        assert ((long int)out % 32 == 0);

        // loop over data and add blinding factors (mod p)
		size_t i = 0;
		for(; i < size; i += 8) {
			const __m256 inp8f = _mm256_load_ps( &inp[i] );				// unblinded input
			const __m256 blind8f = _mm256_load_ps( &temp[i] );			// blinding factor
			const __m256 add8f = _mm256_add_ps(inp8f, blind8f);			// blinded

			const __m256 if_geq = _mm256_cmp_ps(add8f, mid8f, 0x1d);	// blinded >= mid
			const __m256 if_lt = _mm256_cmp_ps(add8f, negmid8f, 0x11);	// blinded < -mid
			const __m256 then8f = _mm256_sub_ps(add8f, p8f);			// blinded - p
			const __m256 elif8f = _mm256_add_ps(add8f, p8f);			// blinded + p
			const __m256 res8f = _mm256_blendv_ps(
										_mm256_blendv_ps(
												add8f,
												elif8f,
												if_lt),
										then8f,
										if_geq);
			_mm256_stream_ps(&out[i], res8f);
		}

        if (verbose) {
			Tensor<double, 0> res;
			res = out_map.template cast<double>().minimum();
			printf("min(new blinded) = %f\n", res.data()[0]);
			res = out_map.template cast<double>().maximum();
			printf("max(new blinded) = %f\n", res.data()[0]);
			res = out_map.template cast<double>().abs().sum();
			printf("sum(abs(new blinded)) = %f\n", res.data()[0]);
		}

		end_time = get_time();
        elapsed = get_elapsed_time(start_time, end_time);
        if (TIMING) {
			printf("\tblinding of size %d required %f sec\n", num_bytes, elapsed);
        }
	}

	// get a blinding factor
	// TODO: this is obviously insecure but we currently compute the unblinding factors outside of the enclave for simplicity
	void slalom_get_r(float* out, int size) {
		if (slalom_privacy) {
			int num_bytes = size * sizeof(float) / batch_size;

			for(int i=0; i<batch_size; i++) {
				get_r(&producer_PRG, ((unsigned char*) out) + i * num_bytes, num_bytes, 9);

				if (verbose) {
					TensorMap<float, 1> r_map(out, size);
					Tensor<double, 0> res;
					res = r_map.template cast<double>().minimum();
					printf("min(r) = %f\n", res.data()[0]);
					res = r_map.template cast<double>().maximum();
					printf("max(r) = %f\n", res.data()[0]);
					res = r_map.template cast<double>().abs().sum();
					printf("sum(abs(r)) = %f\n", res.data()[0]);
				}
			}
		}
	}

	// get an unblinding factor, encrypt it, store its MAC and write the ciphertext outside the enclave
	// TODO: this is obviously insecure but we currently compute the unblinding factors outside of the enclave for simplicity
	void slalom_set_z(float* z, float* dest, int size) {
		int num_bytes = size * sizeof(float) / batch_size;

		sgx_time_t start_time;
        sgx_time_t end_time;
		double elapsed;

		for(int i=0; i<batch_size; i++) {
			//TODO should be randomized
			sgx_aes_gcm_128bit_iv_t *iv = (sgx_aes_gcm_128bit_iv_t*)new sgx_aes_gcm_128bit_iv_t;
			sgx_aes_gcm_128bit_tag_t *mac = (sgx_aes_gcm_128bit_tag_t*)new sgx_aes_gcm_128bit_tag_t;

			start_time = get_time();
			encrypt((uint8_t *) z + i * num_bytes, num_bytes, ((uint8_t *) dest) + i * num_bytes, iv, mac);

			end_time = get_time();
			elapsed = get_elapsed_time(start_time, end_time);
			if (TIMING) {
				printf("encrypt of size %d required %f sec\n", num_bytes, elapsed);
			}

			aes_gcm_ivs.push_back(iv);
			aes_gcm_macs.push_back(mac);
		}
	}

	// compute a ReLU, ReLU6 or fused AvgPool+ReLU on blinded data
	void slalom_relu(float* inp, float* out, float* blind, int num_elements, char* activation) {

		int layer_idx = activation_idxs[act_idx];
		auto curr_layer = model_float.layers[layer_idx];
		auto prev_layer = model_float.layers[layer_idx - 1];
		std::shared_ptr<Layer<float>> next_layer = nullptr;
        act_idx += 1;

        assert(dynamic_cast<Activation<float>*>(curr_layer.get()) != nullptr);

		if (slalom_integrity) {
			// we only handle convolution layers for integrity checks
			assert(dynamic_cast<Conv2D<float>*>(prev_layer.get()) != nullptr);

			if (next_layer != nullptr) {
				// skip reshape layer for MobileNet
				if (dynamic_cast<Conv2D<float>*>(next_layer.get()) == nullptr) {
					assert(dynamic_cast<Reshape<float>*>(next_layer.get()) != nullptr);
					next_layer = model_float.layers[layer_idx + 2];
				}
				assert(dynamic_cast<Conv2D<float>*>(next_layer.get()) != nullptr);
			}
		}

		num_elements /= batch_size;
		int num_bytes = num_elements * sizeof(float);
		std::string act(activation);
		int num_out_elements = num_elements;

		if (layer_idx < model_float.layers.size() -1) {
			next_layer = model_float.layers[layer_idx + 1];
			if (verbose) {
				printf("\nin activation %s: prev layer: %s, curr layer: %s, next_layer: %s\n",
						activation, prev_layer->name_.c_str(), curr_layer->name_.c_str(), next_layer->name_.c_str());
			}
		} else {
			if (verbose) {
				printf("\nin activation %s: prev layer: %s, curr layer: %s\n",
					   activation, prev_layer->name_.c_str(), curr_layer->name_.c_str());
			}
		}

		if (verbose || TIMING) {
			printf("decrypting in relu, size: %d, activation: %s\n", num_elements, activation);
		}

		for (int i = 0; i<batch_size; i++) {
			TensorMap<float, 1> inp_map(inp, num_elements);
			TensorMap<float, 1> blind_map((float*) temp_buffer, num_elements);

			sgx_time_t start_time;
			sgx_time_t end_time;
			double elapsed;

			if (slalom_privacy) {
				start_time = get_time();

				// decrypt the unblinding factors and check MACs
				sgx_aes_gcm_128bit_iv_t *iv = aes_gcm_ivs.front();
				sgx_aes_gcm_128bit_tag_t *mac = aes_gcm_macs.front();
				aes_gcm_ivs.pop_front();
				aes_gcm_macs.pop_front();
				decrypt((uint8_t*) blind, num_bytes, (uint8_t*) temp_buffer, iv, mac, (uint8_t*) inp);
				free(iv);
				free(mac);
	
				end_time = get_time();
				elapsed = get_elapsed_time(start_time, end_time);
				if (TIMING) {
					printf("\tdecrypt of %d bytes required %f sec\n", num_bytes, elapsed);
				}
		
				Tensor<double, 0> res;
				if (verbose) {
					res = blind_map.template cast<double>().minimum();
					printf("min(dec(Z)) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().maximum();
					printf("max(dec(Z)) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().abs().sum();
					printf("sum(abs(dec(Z))) = %f\n", res.data()[0]);

					res = inp_map.template cast<double>().minimum();
					printf("min(blinded) = %f\n", res.data()[0]);
					res = inp_map.template cast<double>().maximum();
					printf("max(blinded) = %f\n", res.data()[0]);
					res = inp_map.template cast<double>().abs().sum();
					printf("sum(abs(blinded)) = %f\n", res.data()[0]);
				}

				start_time = get_time();
				assert(num_elements % 8 == 0);
				assert((long int)inp % 32 == 0);

				if (act == "relu" || act == "relu6") {
					array4d out_shape = prev_layer->output_shape();
					int h = std::max((int) out_shape[1], 1);
					int w = std::max((int) out_shape[2], 1);
					int ch = std::max((int) out_shape[3], 1);

					int batch = num_elements / (h*w*ch);
					if (verbose) {
						printf("layer shape: %d, %d, %d, %d\n", batch, h, w, ch);
					}
					assert(batch == 1);

					integrityParams params;
					params.integrity = slalom_integrity;

					// prepare appropriate integrity checks
					if (slalom_integrity) {
						Conv2D<float>* conv2d_prev = dynamic_cast<Conv2D<float>*>(prev_layer.get());
						Conv2D<float>* conv2d_next = dynamic_cast<Conv2D<float>*>(next_layer.get());

						params.kernel_r_data = conv2d_next->kernel_r_data_;
						params.r_left_data = conv2d_prev->r_left_data_;
						params.r_right_data = conv2d_prev->r_right_data_;

						if (conv2d_next->r_left_data_ == NULL) {
							params.pointwise_x = true;
							params.res_x = model_float.mem_pool->alloc<double>(h*w*REPS);
							preproc_verif_pointwise_bias(params.res_x, conv2d_next->bias_r_data_, h*w);
						} else {
							params.pointwise_x = false;
							params.res_x = model_float.mem_pool->alloc<double>(REPS);
							for (int r=0; r<REPS; r++) {
                                params.temp_x[r] = _mm256_setzero_pd();
                            }
							preproc_verif_bias(params.res_x, conv2d_next->bias_r_data_);
						}

						if (conv2d_prev->r_left_data_ == NULL) {
							params.pointwise_z = true;
							params.res_z = model_float.mem_pool->alloc<double>(h*w*REPS);
							TensorMap<double, 1> res_z_map(params.res_z, h*w*REPS);
							res_z_map.setZero();
						} else {
							params.pointwise_z = false;
							params.res_z = model_float.mem_pool->alloc<double>(REPS);
							TensorMap<double, 1> res_z_map(params.res_z, REPS);
							res_z_map.setZero();
							for (int r=0; r<REPS; r++) {
                    			params.temp_z[r] = _mm256_setzero_pd();
                			}
							//params.res_z_temp = model_float.mem_pool->alloc<double>(REPS);
							//TensorMap<double, 1> res_z_temp_map(params.res_z_temp, REPS);
							//res_z_temp_map.setZero();
						}
					}

					// compute activation, integrity checks and reblinding all in one go
					fused_blind(&consumer_PRG, out, inp, temp_buffer, batch*h*w, ch, activation, params);

					// check integrity
					if (slalom_integrity) {
						Conv2D<float>* conv2d_prev = dynamic_cast<Conv2D<float>*>(prev_layer.get());
						Conv2D<float>* conv2d_next = dynamic_cast<Conv2D<float>*>(next_layer.get());
						model_float.mem_pool->release(params.res_x);
						model_float.mem_pool->release(params.res_z);

						if (conv2d_prev->r_left_data_ == NULL) {
							double sum = 0;
							for (int i=0; i<h*w*REPS; i++) {
								sum += params.res_z[i];
							}
							if (TIMING) { printf("r_out_r: %f\n", sum); }
						} else {
							if (TIMING) {
								printf("r_out_r: %f, %f\n", mod_pos(params.res_z[0], p_verif), mod_pos(params.res_z[1], p_verif));
							}
						}

						if (conv2d_next->r_left_data_ == NULL) {
							double sum = 0;
							for (int i=0; i<h*w*REPS; i++) {
								sum += params.res_x[i];
							}
							if (TIMING) {
								printf("r_inp_wr: %f\n", sum);
							}
						} else {
							if (TIMING) {
								for (int r=0; r<REPS; r++) {
                    				params.res_x[r] += sum_m256d(params.temp_x[r]);
                				}
								printf("r_inp_wr: %f, %f\n", mod_pos(params.res_x[0], p_verif), mod_pos(params.res_x[1], p_verif));
							}
						}
					}

					/*
					auto lambda = [&] (__m256 z) {
                        return _mm256_round_ps(_mm256_mul_ps(_mm256_max_ps(z, zero8f), inv_shift8f), _MM_FROUND_CUR_DIRECTION);
                    };
                    unblind(lambda, inp, temp_buffer, temp_buffer, num_elements);

					Tensor<double, 0> res;
					TensorMap<float, 1> temp_map((float*) temp_buffer, num_elements);
                    res = temp_map.template cast<double>().minimum();
                    printf("min(tmp) = %f\n", res.data()[0]);
                    res = temp_map.template cast<double>().maximum();
                    printf("max(tmp) = %f\n", res.data()[0]);
                    res = temp_map.template cast<double>().abs().sum();
                    printf("sum(abs(tmp)) = %f\n", res.data()[0]);

					slalom_blind_input_internal(temp_buffer, out, num_out_elements, temp_buffer2);
					*/
				} else if (act == "avgpoolrelu" || act == "avgpoolrelu6") {

					// special case for fused AvgPool+ReLU used in MobileNet
					if (act == "avgpoolrelu") {
                        auto act_func = [] (__m256 res8f) {
                            return _mm256_max_ps(res8f, zero8f);
                        };
                        unblind(act_func, inp, temp_buffer, temp_buffer, num_elements);

                    } else if (act == "avgpoolrelu6") {
						auto act_func = [] (__m256 res8f) {
							return _mm256_min_ps(_mm256_max_ps(res8f, zero8f), six8f);
						};
						unblind(act_func, inp, temp_buffer, temp_buffer, num_elements);

					} else {
						assert(0);
					}

					// TODO add integrity check here as well

					array4d input_shape = curr_layer->input_shape_;
					input_shape[0] = 1;
					array4d output_shape = {{1, 1, 1, input_shape[3]}};
					assert(input_shape[1] * input_shape[2] * input_shape[3] == num_elements);

					TensorMap<float, 4> inp_map4d(temp_buffer, input_shape);
					TensorMap<float, 4> out_map4d(temp_buffer2, output_shape);

					float shift = 1.0/256;
					Eigen::array<int, 2> mean_dims({1, 2});
					out_map4d = ((inp_map4d.mean(mean_dims).reshape(out_map4d.dimensions())) * shift).round();

					num_out_elements = input_shape[3];
					slalom_blind_input_internal(temp_buffer2, out, num_out_elements, temp_buffer);

				} else {
					// for the last layer, just unblind and send outside the enclave
					assert(act == "softmax" || act == "linear");

					auto act_func = [] (__m256 res8f) {
						return res8f;
					};
					unblind(act_func, inp, temp_buffer, out, num_elements);
				}

			} else {
				// no privacy, just compute activation

				TensorMap<float, 1> out_map((float*) out, num_elements);
				float shift = 1.0/256;

				if (act == "relu") {
					out_map = (inp_map * shift).round().cwiseMax(static_cast<float>(0));
				} else if (act == "relu6") {
					out_map = (inp_map.cwiseMax(static_cast<float>(0)).cwiseMin(static_cast<float>(6 * 256 * 256)) * shift).round();
				} else if (act == "avgpoolrelu" || act == "avgpoolrelu6") {
					array4d input_shape = curr_layer->input_shape_;
					input_shape[0] = 1;
					array4d output_shape = {{1, 1, 1, input_shape[3]}};
					assert(input_shape[1] * input_shape[2] * input_shape[3] == num_elements);

					TensorMap<float, 4> inp_map4d(inp, input_shape);
					TensorMap<float, 4> out_map4d((float*) out, output_shape);

					Eigen::array<int, 2> mean_dims({1, 2 /* dimensions to reduce */});

					if (act == "avgpoolrelu") {
						auto temp = inp_map4d.cwiseMax(static_cast<float>(0));
						out_map4d = ((temp.eval().mean(mean_dims).reshape(out_map4d.dimensions())) * shift).round();
					} else if (act == "avgpoolrelu6") {
						auto temp = inp_map4d.cwiseMax(static_cast<float>(0)).cwiseMin(static_cast<float>(6 * 256 * 256));
						out_map4d = ((temp.eval().mean(mean_dims).reshape(out_map4d.dimensions())) * shift).round();
					} else {
						assert(0);
					}
					num_out_elements = input_shape[3];
				} else {
					out_map = inp_map;
				}
			}

			inp += num_elements;
			out += num_out_elements;
			blind += num_elements;
		}
	}

	// compute a MaxPool + ReLU on blinded data
	void slalom_maxpoolrelu(float* inp, float* out, float* blind, long int dim_in[4], long int dim_out[4],
                            int window_rows_, int window_cols_, int row_stride_, int col_stride_, bool is_padding_same)
	{
		int num_elements = dim_in[1] * dim_in[2] * dim_in[3];
		assert(dim_in[0] == batch_size);
		int num_bytes = num_elements * sizeof(float);

		int layer_idx = activation_idxs[act_idx];
		auto curr_layer = model_float.layers[layer_idx];
		auto prev_layer = model_float.layers[layer_idx - 1];
		auto next_layer = model_float.layers[layer_idx + 1];
		auto next2_layer = model_float.layers[layer_idx + 2];

		if (verbose) {
			printf("\nin maxpoolrelu: prev layer: %s, curr layer: %s, next layer: %s\n",
				   prev_layer->name_.c_str(), curr_layer->name_.c_str(), next_layer->name_.c_str());
		}
		act_idx += 1;

		assert(dynamic_cast<Activation<float>*>(curr_layer.get()) != nullptr);
		assert(dynamic_cast<MaxPool2D<float>*>(next_layer.get()) != nullptr);

		if (slalom_integrity) {
			// we only handle convolutional layers
			assert(dynamic_cast<Conv2D<float>*>(prev_layer.get()) != nullptr);
			if (dynamic_cast<Conv2D<float>*>(next2_layer.get()) == nullptr) {
				assert(dynamic_cast<Reshape<float>*>(next2_layer.get()) != nullptr);
				next2_layer = model_float.layers[layer_idx + 3];
			}
			assert(dynamic_cast<Conv2D<float>*>(next2_layer.get()) != nullptr);
		}

		int h = dim_in[1];
		int w = dim_in[2];
		int ch = dim_in[3];

		sgx_time_t start_time;
        sgx_time_t end_time;
		double elapsed;

		if (verbose || TIMING) {
            printf("decrypting in maxpoolrelu, size: %d\n", num_elements);
        }

		for (int b = 0; b<batch_size; b++) {

			TensorMap<float, 1> inp_map(inp, num_elements);
			TensorMap<float, 1> blind_map((float*) temp_buffer, num_elements);
			Tensor<double, 0> res;

			if (slalom_privacy) {

				// decrypt the unblinding factors and check MACs
				sgx_aes_gcm_128bit_iv_t *iv = aes_gcm_ivs.front();
				sgx_aes_gcm_128bit_tag_t *mac = aes_gcm_macs.front();
				aes_gcm_ivs.pop_front();
				aes_gcm_macs.pop_front();
				start_time = get_time();

				decrypt((uint8_t*) blind, num_bytes, (uint8_t*) temp_buffer, iv, mac, (uint8_t*) inp);
				free(iv);
				free(mac);

				end_time = get_time();
				elapsed = get_elapsed_time(start_time, end_time);
				if (TIMING) {
					printf("\t decrypt of %d bytes required %f sec\n", num_bytes, elapsed);
				}

				if (verbose) {	
					res = blind_map.template cast<double>().minimum();
					printf("min(dec(Z)) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().maximum();
					printf("max(dec(Z)) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().abs().sum();
					printf("sum(abs(dec(Z))) = %f\n", res.data()[0]);

					res = inp_map.template cast<double>().minimum();
					printf("min(blinded) = %f\n", res.data()[0]);
					res = inp_map.template cast<double>().maximum();
					printf("max(blinded) = %f\n", res.data()[0]);
					res = inp_map.template cast<double>().abs().sum();
					printf("sum(abs(blinded)) = %f\n", res.data()[0]);
				}

				start_time = get_time();
				assert(num_elements % 8 == 0);
				assert ((long int)inp % 32 == 0);

				// unblind the data
				unblind(id_avx, inp, temp_buffer, temp_buffer, num_elements);

				if (verbose) { 
					res = blind_map.template cast<double>().minimum();
					printf("min(unblinded) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().maximum();
					printf("max(unblinded) = %f\n", res.data()[0]);
					res = blind_map.template cast<double>().abs().sum();
					printf("sum(abs(unblinded)) = %f\n", res.data()[0]);
				}

				end_time = get_time();
				elapsed = get_elapsed_time(start_time, end_time);
				if (TIMING) {
					printf("\tintrinsic stuff required %f sec\n", elapsed);
				}

				if (slalom_integrity) {
					// do the integrity checks on the output of the last convolution
					Conv2D<float>* conv2d_prev = dynamic_cast<Conv2D<float>*>(prev_layer.get());
					if (conv2d_prev->r_left_data_ == NULL) {
						conv2d_prev->res_z = model_float.mem_pool->alloc<double>(h*w*REPS);
						TensorMap<double, 1> res_z_map(conv2d_prev->res_z, h*w*REPS);
						res_z_map.setZero();
						conv2d_prev->preproc_verif_pointwise_Z(relu_avx, blind_map.data());
						double sum = 0;
						for (int i=0; i<h*w*REPS; i++) {
							sum += conv2d_prev->res_z[i];
						}
						if (TIMING) {
							printf("r_out_r: %f\n", sum);
						}
						model_float.mem_pool->release(conv2d_prev->res_z);
					} else {
						conv2d_prev->res_z = model_float.mem_pool->alloc<double>(REPS);
						TensorMap<double, 1> res_z_map(conv2d_prev->res_z, REPS);
						res_z_map.setZero();
						conv2d_prev->res_z_temp = model_float.mem_pool->alloc<double>(REPS);
						TensorMap<double, 1> res_z_temp_map(conv2d_prev->res_z_temp, REPS);
						res_z_temp_map.setZero();
						conv2d_prev->preproc_verif_Z(relu_avx, blind_map.data());
						if (TIMING) {
							printf("r_out_r: %f, %f\n", mod_pos(conv2d_prev->res_z[0], p_verif), mod_pos(conv2d_prev->res_z[1], p_verif));
						}
						model_float.mem_pool->release(conv2d_prev->res_z);
						model_float.mem_pool->release(conv2d_prev->res_z_temp);
					}
				}
			}
			else
			{
				new (&blind_map) TensorMap<float, 1>(inp, num_elements);
			}

			int h_out;
			int w_out;
			int pad_rows_;
			int pad_cols_;

			Eigen::PaddingType padding_ = Eigen::PaddingType::PADDING_VALID;
			if (is_padding_same) {
				padding_ = Eigen::PaddingType::PADDING_SAME;
			}

			GetWindowedOutputSize(h, window_rows_, row_stride_,
								  padding_, &h_out, &pad_rows_);
			GetWindowedOutputSize(w, window_cols_, col_stride_,
								  padding_, &w_out, &pad_cols_);
		
			int num_out_elements = h_out * w_out * ch;

			fast_maxpool(blind_map.data(), temp_buffer2, 1, h, w, ch, h_out, w_out,
						 window_rows_, window_cols_, pad_rows_, pad_cols_, row_stride_, col_stride_, false);

			TensorMap<float, 1> out_map((float*) temp_buffer2, num_out_elements);

			if (verbose) {	
				res = out_map.template cast<double>().minimum();
				printf("min(maxpoolrelu output) = %f\n", res.data()[0]);
				res = out_map.template cast<double>().maximum();
				printf("max(maxpoolrelu output) = %f\n", res.data()[0]);
				res = out_map.template cast<double>().abs().sum();
				printf("sum(abs(maxpoolrelu output)) = %f\n", res.data()[0]);
			}		

			if (slalom_privacy) {
				// compute ReLU
				out_map = (out_map * static_cast<float>(1.0/256)).round().cwiseMax(static_cast<float>(0));

				if (slalom_integrity) {
					// do the integrity checks on the input of the next convolution
					Conv2D<float>* conv2d_next = dynamic_cast<Conv2D<float>*>(next2_layer.get());
					if (conv2d_next->r_left_data_ == NULL) {
						conv2d_next->res_x = model_float.mem_pool->alloc<double>(conv2d_next->h*conv2d_next->w*REPS);

						conv2d_next->preproc_verif_pointwise_X(out_map.data());
						double sum = 0;
						for (int i=0; i<conv2d_next->h*conv2d_next->w*REPS; i++) {
							sum += conv2d_next->res_x[i];
						}
						if (TIMING) {
							printf("r_inp_wr: %f\n", sum);
						}
						model_float.mem_pool->release(conv2d_next->res_x);
					} else {
						conv2d_next->res_x = model_float.mem_pool->alloc<double>(REPS);
						conv2d_next->preproc_verif_X(out_map.data());
						if (TIMING) {
							printf("r_inp_wr: %f, %f\n", mod_pos(conv2d_next->res_x[0], p_verif), mod_pos(conv2d_next->res_x[1], p_verif));
						}
						model_float.mem_pool->release(conv2d_next->res_x);
					}
				}

				// reblind output
				slalom_blind_input_internal(temp_buffer2, out, num_out_elements, temp_buffer);
			} else {
				TensorMap<float, 1> out_map2(out, num_out_elements);
				out_map2 = (out_map * static_cast<float>(1.0/256)).round().cwiseMax(static_cast<float>(0));
			}

		inp += num_elements;
		blind += num_elements;
		out += num_out_elements;
		}
	}

	void sgxdnn_benchmarks(int num_threads) {
		benchmark(num_threads);
	}
}
