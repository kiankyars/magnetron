from math import ceil

from dotenv import load_dotenv
from flask import Flask, render_template, request
import magnetron as mag
from magnetron.models import *

load_dotenv()

EPOCHS: int = 10000
LEARNING_RATE: float = 0.8

# Inputs
inputs = [
    mag.Tensor.const([0.0, 0.0]),
    mag.Tensor.const([0.0, 1.0]),
    mag.Tensor.const([1.0, 0.0]),
    mag.Tensor.const([1.0, 1.0])
]

# Targets
targets = [
    mag.Tensor.const([0.0]),
    mag.Tensor.const([1.0]),
    mag.Tensor.const([1.0]),
    mag.Tensor.const([0.0])
]

mlp = SequentialModel([
    DenseLayer(2, 4),
    DenseLayer(4, 1)
])

# Train model XOR
print('Training XOR model...')
losses = mlp.train(inputs, targets, EPOCHS, LEARNING_RATE)

print('Launching Flask server...')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/v1/bot_response')
def get_response():
    try:
        message = request.args.get('message')
        splits = message.split(' ')
        a: float = float(splits[0])
        b: float = float(splits[1])
        inp = mag.Tensor.const([a, b])
        result: float = mlp.forward(inp).scalar()
        result_rounded: float = mlp.forward(inp, activation=mag.Operator.HARD_SIGMOID).scalar()
        return f'{a} ^ {b} = {result} ≈ {result_rounded} => {int(result_rounded) == 1}'
    except:
        return 'Please enter a valid input. Enter two numbers (between 0 and 1) seperated by spaces. For example: 1 1 or 1 0 or 0 0.'


@app.route('/api/v1/system_info')
def get_system_info():
    ctx = mag.Context.active
    return f'{ctx.os_name} | {ctx.cpu_name} ({ctx.cpu_virtual_cores}) | {ctx.physical_memory_total / (1 << 30)} GiB RAM | {(ctx.total_allocated_pool_memory / (1 << 20)):.2f} MiB POOL'


if __name__ == '__main__':
    app.run(host='0.0.0.0', use_reloader=False)  #  host='0.0.0.0'
