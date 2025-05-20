# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Callable, MutableMapping

from magnetron import Tensor


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor, name: str | None = None) -> None:
        x.requires_grad = True
        if name is not None:
            x.name = name
        self.x = x


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> list[Parameter]:
        """Return all unique and nested parameters of the module."""
        params: list[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, ModuleList):
                for m in v:
                    params.extend(m.parameters())
        # dedupe while preserving order
        unique: list[Parameter] = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    def children(self) -> Iterator['Module']:
        """Yield immediate child modules."""
        for v in self.__dict__.values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, ModuleList):
                for m in v:
                    yield m

    def modules(self) -> Iterator['Module']:
        """Yield self and all submodules in pre-order."""
        yield self
        for child in self.children():
            yield from child.modules()

    def apply(self, fn: Callable[['Module'], None]) -> 'Module':
        """
        Apply `fn` to self and all submodules.
        Example:
            model.apply(lambda m: init_fn(m))
        """
        for m in self.modules():
            fn(m)
        return self

    def eval(self) -> None:
        """Set module to evaluation mode (disable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = False

    def train(self) -> None:
        """Set module to training mode (enable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = True

    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        """Forward pass; must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs: dict) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a persistent buffer (non-parameter tensor)."""
        buf = tensor.clone().detach() if isinstance(tensor, Tensor) else tensor
        setattr(self, name, buf)


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


class ModuleDict(Module, MutableMapping[str, Module]):
    """A dict of named submodules that behaves like a single Module."""

    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        self._modules: dict[str, Module] = {}
        if modules is not None:
            for name, mod in modules.items():
                self[name] = mod

    def __setitem__(self, name: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise ValueError(f'ModuleDict can only hold Module, got {type(module)}')
        # store in our internal dict
        self._modules[name] = module
        # also bind it as an attribute so Module.children()/modules() will see it
        setattr(self, name, module)

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __delitem__(self, name: str) -> None:
        del self._modules[name]
        delattr(self, name)

    def __iter__(self) -> None:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def keys(self) -> 'dict_keys':
        return self._modules.keys()

    def items(self) -> 'dict_items':
        return self._modules.items()

    def values(self) -> 'dict_values':
        return self._modules.values()

    def parameters(self) -> list[Parameter]:
        # flatten out all parameters from each module
        params: list[Parameter] = []
        for mod in self._modules.values():
            params.extend(mod.parameters())
        # dedupe
        unique: list[Parameter] = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique


class Linear(Module):
    """A fully connected linear layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = Tensor.normal(shape=(out_features, in_features), mean=0.0, std=1.0)
        weight = weight / math.sqrt(in_features + out_features)
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(Tensor.zeros((out_features,), name='bias'))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.x.T
        if self.bias is not None:
            x = x + self.bias.x
        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.normal((num_embeddings, embedding_dim)) / embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor.zeros(shape=(dim,)))

    def _norm(self, x: Tensor) -> Tensor:
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x)
        return output * self.weight


class LayerNorm(Module):
    """Layer Normalization over the last dimension with optional affine transform."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(Tensor.full((normalized_shape,), fill_value=1.0), name='weight')
            self.bias = Parameter(Tensor.zeros((normalized_shape,)), name='bias')
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        if self.elementwise_affine:
            x_norm = x_norm * self.weight.x + self.bias.x
        return x_norm


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

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss."""

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y_hat = y_hat.softmax()
        return -(y * y_hat.log()).sum(dim=-1).mean()
