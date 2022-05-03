cuda:
	@nvcc wave.cu tools.cpp -o wave
run-cuda: cuda
	@./wave 1000 1000 50000
	@./wave 2000 2000 50000
	@./wave 3000 3000 50000
	@./wave 5000 5000 50000
	@./wave 7000 7000 50000
	@./wave 10000 10000 50000

cuda-opt:
	@nvcc wave_opt.cu tools.cpp -o wave_opt
run-cuda-opt: cuda-opt
	@./wave_opt 1000 1000 50000
	@./wave_opt 2000 2000 50000
	@./wave_opt 3000 3000 50000
	@./wave_opt 5000 5000 50000
	@./wave_opt 7000 7000 50000
	@./wave_opt 10000 10000 50000

omp:
	@g++ wave_seq.c -o wave_omp -fpermissive -fopenmp
run-omp: omp
	@./wave_seq 4000 4000 10000

seq:
	@g++ wave_seq.c -o wave_seq -fpermissive
run-seq: seq
	@./wave_seq 4000 4000 10000
