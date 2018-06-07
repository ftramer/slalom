# SLALOM
Fast, Verifiable and Private Machine Learning on Trusted Hardware

# Installation

After cloning the SLALOM repository, you can build as follows:

1. Follow the [instructions](https://github.com/intel/linux-sgx) to install the Intel SGX driver and SDK for Linux.
2. [Build TensorFlow from source with GPU support](https://www.tensorflow.org/install/install_sources) (this requires a working CUDA installation)
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

# Running SLALOM

## Evaluation with integrity
To evaluate a forward pass of a network, run:
```
python -m python.slalom.scripts.eval [vgg_16 | mobilenet | mobilenet_sep] sgxdnn --batch_size=1 --max_num_batches=4 {--verify}  {--verify_batched} {--verify_preproc} {--use_sgx}
```
You can choose between 3 models, VGG16, MobileNet and a version of MobileNet with no intermediate activations in separable convolutions (this model is untrained). If the `verify` flag is set, computations are performed on GPU and verified on CPU. The extra ` verify_batched` and `verify_preproc` flags enable faster batched verification or verification with preprocessed secrets respecitvely. If the `use_sgx` flag is set, the CPU computations are performed inside a secure SGX enclave.

## Evaluation with privacy and integrity
To evaluate a private forward pass of a network, run:
```
python -m python.slalom.scripts.eval_slalom [vgg_16 | mobilenet | mobilenet_sep] --batch_size=8 --max_num_batches=4 {--blinding}  {--integrity} {--use_sgx}
```
Here, the computation alternates between GPU and CPU after each linear layer. If the `blinding` flag is set, input privacy is guaranteed by precomputing random blinding and unblinding factors for linear layers and storing them (encrypted) in untrusted memory. Adding the `integrity` flag additionaly enables integrity checks on the blinded computations performed by the untrusted GPU (only works for the ` vgg_16` and `mobilenet_sep` models for now).
