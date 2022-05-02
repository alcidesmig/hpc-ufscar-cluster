cuda:
	@nvcc wave.cu tools.cpp -o wave
run-cuda: cuda
	@./wave 4000 4000 10000

omp:
	@g++ wave_seq.c -o wave_omp -fpermissive -fopenmp
run-omp: omp
	@./wave_seq 4000 4000 10000

seq:
	@g++ wave_seq.c -o wave_seq -fpermissive
run-seq: seq
	@./wave_seq 4000 4000 10000
