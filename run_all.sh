mkdir -p results

################
## Benchmarks ##
################

python -u -m python.slalom.scripts.benchmarks &> results/benchmarks.txt
python -u -m python.slalom.scripts.benchmarks --use_sgx &> results/benchmarks_sgx.txt
python -u -m python.slalom.scripts.benchmarks --threads=4 &> results/benchmarks_threaded_4.txt
python -u -m python.slalom.scripts.benchmarks --threads=8 &> results/benchmarks_threaded_8.txt

###########
## VGG16 ##
###########

# forward pass
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 &> results/vgg_full_cpu.txt
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx &> results/vgg_full_sgx.txt

# verify
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify &> results/vgg_full_verif_cpu.txt
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --use_sgx &> results/vgg_full_verif_sgx.txt

# verify batched
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --verify_batched &> results/vgg_full_verif_batched_cpu.txt

# verify preproc
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc &> results/vgg_full_verif_preproc_cpu.txt
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx &> results/vgg_full_verif_preproc_sgx.txt

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding &> results/vgg_full_slalom_privacy_cpu.txt
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --use_sgx &> results/vgg_full_slalom_privacy_sgx.txt

# slalom privacy+integrity
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --integrity &> results/vgg_full_slalom_privacy_integrity_cpu.txt
python -u -m python.slalom.scripts.eval_slalom vgg_16 --batch_size=16 --max_num_batches=4 --blinding --integrity --use_sgx &> results/vgg_full_slalom_privacy_integrity_sgx.txt

################
## VGG No Top ##
################

# forward pass
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx --no_top &> results/vgg_notop_sgx.txt

# verif
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --use_sgx --no_top --verify &> results/vgg_notop_verif_sgx.txt

# verif preproc
python -u -m python.slalom.scripts.eval vgg_16 sgxdnn --batch_size=16 --max_num_batches=4 --use_sgx --no_top --verify --preproc &> results/vgg_notop_verif_preproc_sgx.txt

###############
## MobileNet ##
###############

# forward pass
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=1 --max_num_batches=4 &> results/mobilenet_cpu.txt
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=1 --max_num_batches=4 --use_sgx &> results/mobilenet_sgx.txt

# verify
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify &> results/mobilenet_verif_cpu.txt
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --use_sgx &> results/mobilenet_verif_sgx.txt

# verify batched
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --verify_batched &> results/mobilenet_verif_batched_cpu.txt

# verify preproc
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc &> results/mobilenet_verif_preproc_cpu.txt
python -u -m python.slalom.scripts.eval mobilenet sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx &> results/mobilenet_verif_preproc_sgx.txt

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom mobilenet --batch_size=16 --max_num_batches=4 --blinding &> results/mobilenet_slalom_privacy_cpu.txt
python -u -m python.slalom.scripts.eval_slalom mobilenet --batch_size=16 --max_num_batches=4 --blinding --use_sgx &> results/mobilenet_slalom_privacy_sgx.txt

###################
## MobileNet-Sep ##
###################

# verify preproc
python -u -m python.slalom.scripts.eval mobilenet_sep sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc &> results/mobilenet_sep_verif_preproc_cpu.txt
python -u -m python.slalom.scripts.eval mobilenet_sep sgxdnn --batch_size=16 --max_num_batches=4 --verify --preproc --use_sgx &> results/mobilenet_sep_verif_preproc_sgx.txt

# slalom privacy
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding &> results/mobilenet_sep_slalom_privacy_cpu.txt
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --use_sgx &> results/mobilenet_sep_slalom_privacy_sgx.txt

# slalom privacy+integrity
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --integrity &> results/mobilenet_sep_slalom_privacy_integrity_cpu.txt
python -u -m python.slalom.scripts.eval_slalom mobilenet_sep --batch_size=16 --max_num_batches=4 --blinding --integrity --use_sgx &> results/mobilenet_sep_slalom_privacy_integrity_sgx.txt

