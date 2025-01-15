# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time
from abc import ABC
import matplotlib.pyplot as plt
import magnetron as mag

class BenchParticipant(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.timings = []

    def allocate_args(self, dim: int) -> tuple:
        pass

def bench_iter_avg(dim: int, iters: int, a, b, func: callable) -> float:
    flop: int = 2*dim**3
    flops: list[float] = []
    for _ in range(iters):
        st: float = time.monotonic()
        _r = func(a, b)
        et: float = time.monotonic()
        s: float = et - st
        flops.append(flop/s)
    return sum(flops) / len(flops)

class PerformanceInfo:
    def __init__(self, name: str, shapes: list[int], participants: list[BenchParticipant]):
        self.name = name
        self.shapes = shapes
        self.participants = participants

    def plot(self):
        plt.figure(figsize=(10, 6))
        for participant in self.participants:
            plt.plot(self.shapes, participant.timings, label=participant.name)
        plt.xlabel('Matrix Size (NxN)')
        plt.ylabel('Average FLOP/s')
        plt.title(f'Matrix {self.name} Benchmark - {mag.Context.active().cpu_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

def benchmark(name: str, participants: list[BenchParticipant], func: callable, dim_lim: int=2048, step: int=8, iters: int=10000) -> PerformanceInfo:
    shapes: list[int] = []
    for participant in participants:
        participant.timings.clear()
    for dim in range(step, dim_lim+step, step):
        print(f'Benchmarking {dim}x{dim}')
        shapes.append(dim)
        for participant in participants:
            x, y = participant.allocate_args(dim)
            participant.timings.append(bench_iter_avg(dim, iters, x, y, func))
    return PerformanceInfo(name, shapes, participants)
