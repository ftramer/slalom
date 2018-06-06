#ifndef aes_stream_H
#define aes_stream_H

#include <stdlib.h>
#include "sgxdnn_main.hpp"

#ifndef CRYPTO_ALIGN
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#  define CRYPTO_ALIGN(x) __declspec(align(x))
# else
#  define CRYPTO_ALIGN(x) __attribute__((aligned(x)))
# endif
#endif

#ifndef AES_STREAM_ROUNDS
# define AES_STREAM_ROUNDS 10
#endif

typedef struct CRYPTO_ALIGN(16) aes_stream_state {
    unsigned char opaque[((AES_STREAM_ROUNDS) + 1) * 16 + 16];
} aes_stream_state;

#define AES_STREAM_SEEDBYTES 32

void aes_stream_init(aes_stream_state *st,
                     const unsigned char seed[AES_STREAM_SEEDBYTES]);

void aes_stream(aes_stream_state *st, unsigned char *buf, size_t buf_len, bool shift);

void aes_stream_fused(aes_stream_state *st, float *out, float* input, float* blind, size_t image_size, size_t ch,
					  char* activation, integrityParams& params);

#endif
