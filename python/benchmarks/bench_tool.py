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

    def plot(self):
        plt.figure(figsize=(10, 6))
        markers = ['o', '+', 'x', '*', '.', 'X', '^']
        for (i, participant) in enumerate(self.participants):
            plt.plot(self.shapes, participant.timings, label=participant.name, marker=markers[i%len(markers)])
        plt.xlabel('Matrix Size (NxN)')
        plt.ylabel('Average Time (s)')
        plt.title(f'Matrix {self.name} Benchmark - (Lower is Better) - {mag.Context.active().cpu_name}')
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
            participant.timings.append(bench_fn(iters, x, y, func))
    return PerformanceInfo(name, shapes, participants)
