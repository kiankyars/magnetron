# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import time

class BenchInfo:
    def __init__(self, flops: list[float]):
        self.flops = flops

    def avg_flops(self) -> float:
        return sum(self.flops) / len(self.flops)

    def avg_tflops(self) -> float:
        return self.avg_flops() * 1e-12

    def min_flops(self) -> float:
        return min(self.flops)

    def min_tflops(self) -> float:
        return min(self.flops) * 1e-12

    def max_flops(self) -> float:
        return max(self.flops)

    def max_tflops(self) -> float:
        return max(self.flops) * 1e-12

    def __str__(self) -> str:
        return f'Average: {self.avg_tflops()} TFLOP/s\nMin: {self.min_tflops()} TFLOP/s\nMax: {self.max_tflops()} TFLOP/s'

def bench(dim: int, iters: int, a, b, func: callable) -> BenchInfo:
    flop: int = 2*dim**3
    flops: list[float] = []
    for _ in range(iters):
        st: float = time.monotonic()
        _r = func(a, b)
        et: float = time.monotonic()
        s: float = et - st
        flops.append(flop/s)

    return BenchInfo(flops)
