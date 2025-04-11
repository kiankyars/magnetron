## Interactive MNIST Example

This example allows you to draw a digit on a canvas and then uses a pre-trained model to predict the digit. The model is loaded from a file, and the prediction is displayed on the canvas.

## Requirements
Make sure to have the optional dependencies for `examples` installed. You can do this by running:
```bash
pip install .[examples]
```

## Running
Run the django server with the following command:
```bash
python manage.py runserver --nothreading --noreload
```

The `--nothreading` and `--noreload` flags are required to force django into single threaded mode as magnetron is not thread-safe.