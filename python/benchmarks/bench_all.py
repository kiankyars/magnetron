# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from benchmarks.bench import *

bench_square_bin_ops(dim_lim=4096)
bench_square_matmul(dim_lim=4096)
bench_permuted_bin_ops(dim_lim=4096)
bench_permuted_matmul(dim_lim=4096)
