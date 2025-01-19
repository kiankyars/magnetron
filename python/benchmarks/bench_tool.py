# (c) 2024 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import magnetron as mag
import timeit

class BenchParticipant(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.timings = []

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> tuple:
        pass

def bench_fn(iters: int, x, y, func: callable) -> float:
    return timeit.timeit(lambda: func(x, y), number=iters) / iters

class PerformanceInfo:
    def __init__(self, name: str, shapes_a: list[tuple[int, int]], shapes_b: list[tuple[int, int]], participants: list[BenchParticipant]):
        self.name = name
        self.shapes_a = shapes_a
        self.shapes_b = shapes_b
        self.participants = participants

    def plot(self, flops_per_op: int=2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        x_labels = [f'A{sa} B{sb}' for sa, sb in zip(self.shapes_a, self.shapes_b)]
        x = np.arange(len(x_labels))
        width = 0.8 / len(self.participants)
        
        for i, participant in enumerate(self.participants):
            offset = (i - len(self.participants)/2 + 0.5) * width
            ax1.bar(x + offset, participant.timings, width, label=participant.name, color=colors[i%len(colors)])
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        ax1.set_ylabel('Average Time (s)')
        ax1.set_title(f'Matrix {self.name} Benchmark\n(Lower is Better)')
        ax1.legend()
        ax1.grid(True, axis='y')

        for i, participant in enumerate(self.participants):
            total_elements = [(sa[0] * sa[1] + sb[0] * sb[1]) for sa, sb in zip(self.shapes_a, self.shapes_b)]
            gflops = [(elements * flops_per_op) / (time * 1e9)
                     for elements, time in zip(total_elements, participant.timings)]
            offset = (i - len(self.participants)/2 + 0.5) * width
            ax2.bar(x + offset, gflops, width, label=participant.name, color=colors[i%len(colors)])
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_ylabel('Performance (GFLOPS)')
        ax2.set_title(f'Matrix {self.name} Throughput\n(Higher is Better)')
        ax2.legend()
        ax2.grid(True, axis='y')

        plt.suptitle(f'{mag.Context.active().cpu_name}', y=1.05)
        plt.tight_layout()
        plt.show()

def generate_matmul_shapes(max_dim: int, step: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    shapes = []
    for m in range(step, max_dim + step, step):
        for k in range(step, max_dim + step, step):
            for n in range(step, max_dim + step, step):
                shapes.append(((m, k), (k, n)))
    return shapes

def generate_elementwise_shapes(max_dim: int, step: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    shapes = []
    for m in range(step, max_dim + step, step):
        for n in range(step, max_dim + step, step):
            shapes.append(((m, n), (m, n)))
    return shapes

def benchmark(name: str, participants: list[BenchParticipant], func: callable, 
             shapes: list[tuple[tuple[int, int], tuple[int, int]]], iters: int=10000) -> PerformanceInfo:
    shapes_a = []
    shapes_b = []
    
    for participant in participants:
        participant.timings.clear()
    
    for shape_a, shape_b in shapes:
        print(f'Benchmarking A:{shape_a}, B:{shape_b}')
        shapes_a.append(shape_a)
        shapes_b.append(shape_b)
        
        for participant in participants:
            x, y = participant.allocate_args(shape_a, shape_b)
            participant.timings.append(bench_fn(iters, x, y, func))
    
    return PerformanceInfo(name, shapes_a, shapes_b, participants)
