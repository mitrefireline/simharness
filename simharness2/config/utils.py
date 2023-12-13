from dataclasses import dataclass, field

@dataclass
class GetClassConf:
    """Base class for objects to be instantiated with `hydra.utils.instantiate`."""

    _target_: str = field(default="hydra.utils.get_class", init=False)
