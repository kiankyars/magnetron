import magnetron as mag

# Perceptron function (McCulloch–Pitts neuron)
def perceptron(x: mag.Tensor, w: mag.Tensor, b: mag.Tensor) -> mag.Tensor:
    return (w @ x + b).heaviside_step()

# Negating perceptron
def logical_not(input: float) -> float:
    x = mag.Tensor.const([input])
    w = mag.Tensor.const([-1])
    b = mag.Tensor.const([0.5])
    r = perceptron(x, w, b)
    return r.scalar()

truth_table = [
    0.0,
    1.0,
    0.0,
    1.0,
]

for bit in truth_table:
    print(f'¬{bit} = {logical_not(bit)}')
