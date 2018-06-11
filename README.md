# SLALOM
**Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware.**

Slalom is a framework for accelerating Deep Neural Network evaluations in trusted hardware, by selectively outsourcing computations to an untrusted (but faster) colocated device while preserving the integrity and privacy of the computation.
In its current implementation, Slalom runs the evaluation of a neural network inside an Intel SGX enclave, and delegates the computation of all linear layers to an untrusted GPU on the same mahcine.

This project is based on the following paper:

*Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware* </br>
**Florian Tramèr and Dan Boneh** </br>
[arXiv:1806.03287](http://arxiv.org/abs/1806.03287)

## Disclaimer

**DO NOT USE THIS SOFTWARE TO SECURE ANY 
REAL-WORLD DATA OR COMPUTATION!**

This software is a proof-of-concept meant for 
performance testing of the Slalom framework ONLY.
It is full of security vulnerabilities that 
facilitate testing, debugging and performance 
measurements. In any real-world deployment, 
these vulnerabilities can be easily exploited 
to leak all user inputs. 

Some parts that have a negligble impact on performance but that are required for a real-world deployment are not currently implemented (e.g., setting up a secure communication channel with a remote client and producing verifiable attestations).

## Background
Trusted hardware (e.g., [Intel SGX](https://software.intel.com/en-us/sgx), [AMD TrusZone](https://www.amd.com/en/technologies/security), or the open-source [Sanctum](https://eprint.iacr.org/2015/564.pdf) architecture) can construct isolated execution environments ("enclaves") for running security or privacy sensitive applications. Using trusted hardware, it is possible to execute a full neural network evaluation in an enclave, but this comes at a relatively steep cost in performance. Existing trusted hardware platforms currently only support low-end computation devices (e.g, not your brand-new shiny GPU or multicore server CPU), and incur additional costs for isolating computations and handling large memory regions.

Slalom uses a novel approach that consists in *delegating* computations from a (slow) trusted environment to a co-located untrusted---yet much faster---device. Slalom builds upon a [well-known](https://en.wikipedia.org/wiki/Freivalds%27_algorithm) efficient method for verifying outsourced matrix multiplications, which we adapt to enable (privacy-preserving) verification of the main linear operators used in modern neural networks (i.e., convolutions, separable convolutions, and dense layers). Nonlinear computations (e.g., activations, pooling, etc.) are computed locally by the trusted enclave, but represent only a tiny fraction of the total execution time, which is dominated by linear operations. 

## Installation

After cloning the Slalom repository, you can build as follows:

1. Follow the [instructions](https://github.com/intel/linux-sgx) to install the Intel SGX driver and SDK for Linux.
2. [Build TensorFlow (version 1.8.0) from source with GPU support](https://www.tensorflow.org/install/install_sources) (this requires a working CUDA installation)
3. Install the remaining python dependencies:
```
cd slalom
pip install -r requirements.txt
```
4. Build the custom TensorFlow Ops:
```
cd slalom/App
make
make -f Makefile_cu
```
5. Build the SGXDNN library (for use without SGX):
```
cd slalom/SGXDNN
make
```
6. Build the SGX application:
```
cd slalom
make
```

## Running Slalom

### Evaluation with integrity
To evaluate a forward pass of a network, run:
```
python -m python.slalom.scripts.eval [vgg_16 | mobilenet | mobilenet_sep] sgxdnn --batch_size=8 --max_num_batches=4 {--verify}  {--verify_batched} {--verify_preproc} {--use_sgx}
```
You can choose between 3 models, VGG16, MobileNet and a version of MobileNet with no intermediate activations in separable convolutions (this model is untrained). If the `verify` flag is set, computations are performed on GPU and verified on CPU. The extra ` verify_batched` and `verify_preproc` flags enable faster batched verification or verification with preprocessed secrets respecitvely. If the `use_sgx` flag is set, the CPU computations are performed inside a secure SGX enclave.

### Evaluation with privacy and integrity
To evaluate a private forward pass of a network, run:
```
python -m python.slalom.scripts.eval_slalom [vgg_16 | mobilenet | mobilenet_sep] --batch_size=8 --max_num_batches=4 {--blinding}  {--integrity} {--use_sgx}
```
Here, the computation alternates between GPU and CPU after each linear layer. If the `blinding` flag is set, input privacy is guaranteed by precomputing random blinding and unblinding factors for linear layers and storing them (encrypted) in untrusted memory. Adding the `integrity` flag additionaly enables integrity checks on the blinded computations performed by the untrusted GPU (only works for the ` vgg_16` and `mobilenet_sep` models for now).
