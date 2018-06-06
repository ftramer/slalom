
#ifndef _ENCLAVE_H_
#define _ENCLAVE_H_

#include <stdlib.h>
#include <assert.h>

#if defined(__cplusplus)
extern "C" {
#endif

void printf(const char *fmt, ...);
void start_clock();
void end_clock(const char *str);

inline void print_hex(unsigned char *mem, int len) {
  for (int i = 0; i < len; i++) {
    printf("%#02x, ", *(mem+i));
  }
}

#if defined(__cplusplus)
}
#endif

#endif /* !_ENCLAVE_H_ */
