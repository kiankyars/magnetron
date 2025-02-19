# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
import math

from magnetron.core import Tensor, ffi


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor) -> None:
        x.requires_grad = True
        self.x = x


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> list[Parameter]:
        """Returns all unique and nested parameters of the module."""
        params: list[Parameter] = []
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, ModuleList):
                for mod in v:
                    params += mod.parameters()
        return list(set(params))

    def eval(self) -> None:
        """Sets the module in evaluation mode."""
        for p in self.parameters():
            p.x.requires_grad = False

    def train(self) -> None:
        """Sets the module in training mode."""
        for p in self.parameters():
            p.x.requires_grad = True

    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        """Forward pass must be implemented by subclass."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs: dict) -> Tensor:
        return self.forward(*args, **kwargs)


class ModuleList(Module, list):
    """A list of modules that can be used as a single module."""

    def __init__(self, mods: list[Module] | None) -> None:
        super().__init__()
        if mods is not None:
            self += mods

    def append(self, mod: Module) -> None:
        super().append(mod)

    def extend(self, __iterable: list[Module]) -> None:
        super().extend(__iterable)

    def __iadd__(self, other: list[Module]) -> 'ModuleList':
        self.extend(other)
        return self

    def __setitem__(self, k: int, v: Module) -> None:
        super().__setitem__(k, v)

    def __getitem__(self, k: int) -> Module:
        return super().__getitem__(k)

    def parameters(self) -> list[Parameter]:
        """Returns all unique and nested parameters of the module."""
        params: list[Parameter] = []
        for mod in self:
            params += mod.parameters()
        return list(set(params))


class Linear(Module):
    """A fully connected linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = Tensor.normal((out_features, in_features), mean=0, stddev=1)
        weight = weight / math.sqrt(in_features + out_features)
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(Tensor.zeros((out_features,), name='bias'))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.x.T.clone()
        if self.bias is not None:
            x = x + self.bias.x
        return x
