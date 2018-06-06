#ifndef RAND_POOL_H
#define RAND_POOL_H

#include "assert.h"
#include "aes-stream.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

#define STATE_LEN ((AES_STREAM_ROUNDS) + 1) * 16 + 16
unsigned char init_key[STATE_LEN] = {0x00};	// TODO generate at random
unsigned char init_seed[AES_STREAM_SEEDBYTES] = {0x00}; //TODO generate at random

void init_PRG(aes_stream_state* state) {
	std::copy(init_key, init_key + STATE_LEN , state->opaque);
	aes_stream_init(state, init_seed);
}

void get_PRG(aes_stream_state* state, unsigned char* out, size_t length_in_bytes) {
	aes_stream(state, out, length_in_bytes, true);
}

void get_r(aes_stream_state* state, unsigned char* out, size_t length_in_bytes, int shift) {
	get_PRG(state, out, length_in_bytes);
}

void fused_blind(aes_stream_state* state, float* out, float* blinded_input, float* blind, size_t image_size, size_t ch,
				 char* activation, integrityParams& params) {
	aes_stream_fused(state, out, blinded_input, blind, image_size, ch, activation, params);
} 

#endif
