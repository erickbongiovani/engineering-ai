from dataclasses import dataclass, field

@dataclass(frozen=True)
class Example:
    features: tuple[float, ...]
    label: int

@dataclass
class TrainConfig:
    lr: float = 0.01
    epochs: int = 5
    hidden: list[int] = field(default_factory=lambda: [64, 32])

    def describe(self):
        return f'lr={self.lr}, epochs={self.epochs}, hidden={self.hidden}'

ex = Example((1.5, 2.0), 1)
print(ex.features)