# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from abc import ABC, abstractmethod

from magnetron import Tensor


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor) -> None:
        x.requires_grad = True
        self.x = x


class Module(ABC):
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

    @abstractmethod
    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        """Forward pass must be implemented by subclass."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs: dict) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor: Tensor):
        tensor = tensor.clone().detach() if isinstance(tensor, Tensor) else tensor
        setattr(self, name, tensor)


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
        self.weight = Parameter(Tensor.full((out_features, in_features), fill_value=0.5)) # TODO: proper init
        if bias:
            self.bias = Parameter(Tensor.zeros((out_features,), name='bias'))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.x.T.clone()
        if self.bias is not None:
            x = x + self.bias.x
        return x

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            Tensor.normal((num_embeddings, embedding_dim)) / embedding_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor.zeros(shape=(dim,)))

    def _norm(self, x: Tensor) -> Tensor:
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight

class Loss(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss."""

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        d = y_hat - y
        return (d * d).mean()
