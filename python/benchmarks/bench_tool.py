# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from abc import ABC
import matplotlib.pyplot as plt
import magnetron as mag
import timeit

class BenchParticipant(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.timings = []

    def allocate_args(self, dim: int) -> tuple:
        pass

def bench_fn(iters: int, x, y, func: callable) -> float:
    return timeit.timeit(lambda: func(x, y), number=iters) / iters

class PerformanceInfo:
    def __init__(self, name: str, shapes: list[int], participants: list[BenchParticipant]):
        self.name = name
        self.shapes = shapes
        self.participants = participants

    def plot(self, flops_per_op: int=2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        markers = ['o', '+', 'x', '*', '.', 'X', '^']
        for (i, participant) in enumerate(self.participants):
            ax1.plot(self.shapes, participant.timings, label=participant.name, marker=markers[i%len(markers)])
        ax1.set_xlabel('Matrix Size (NxN)')
        ax1.set_ylabel('Average Time (s)')
        ax1.set_title(f'Matrix {self.name} Benchmark\n(Lower is Better)')
        ax1.legend()
        ax1.grid(True)
        for (i, participant) in enumerate(self.participants):
            gflops = [(shape * shape * flops_per_op) / (time * 1e9)
                      for shape, time in zip(self.shapes, participant.timings)]
            ax2.plot(self.shapes, gflops, label=participant.name, marker=markers[i%len(markers)])
        ax2.set_xlabel('Matrix Size (NxN)')
        ax2.set_ylabel('Performance (GFLOPS)')
        ax2.set_title(f'Matrix {self.name} Throughput\n(Higher is Better)')
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(f'{mag.Context.active().cpu_name}', y=1.05)
        plt.tight_layout()
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
            participant.timings.append(bench_fn(iters, x, y, func))
    return PerformanceInfo(name, shapes, participants)
