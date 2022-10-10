import logging
from abc import ABC
from typing import Any, Optional, TypeVar

__all__ = ("Registrable",)

_registry: dict[str, dict[str, type["Registrable"]]] = {}

log = logging.getLogger(__name__)

T = TypeVar("T", bound="Registrable", covariant=True)


class Registrable(ABC):  # noqa: B024

    prefix: str = ""
    suffix: str = ""
    transparent: bool = False

    def __init__(self, **kwargs: Any) -> None:
        # Registrable is an abstract class because its subclasses will be,
        # but each of them will have a different interface, so we don't have
        # abstract methods to be defined;
        # however, Registrable is not meant to be instantiated, so we raise
        # an error if someone tries to do so
        if type(self) is Registrable:
            raise TypeError("Can't instantiate abstract class Registrable")
        super().__init__(**kwargs)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # Make sure the subclass inherits only once from Registrable along its
        # ancestors chain
        if not _check_only_one_registrable_in_hierarchy(cls):
            raise TypeError(
                f"Class {cls.__name__} inherits from more than one Registrable"
            )
        parent_name = _get_parent_registrable(cls)
        if parent_name in _registry:
            parent_registry = _registry[parent_name]
            # Make sure there is no other algorithm with the same case-insensitive name
            lowercase_classes = [
                k.lower().replace("_", "") for k in parent_registry.keys()
            ]
            if cls.__name__.lower() not in lowercase_classes:
                _registry[parent_name][cls.__name__] = cls
            else:
                raise TypeError(
                    f"Class {cls.__name__} already exists in {parent_name} registry"
                )
        else:
            _registry[parent_name] = {cls.__name__: cls}

    @classmethod
    def find(cls: type[T], name: str, add_affixes: bool = True, **kwargs: Any) -> T:
        if cls.__name__ not in _registry:
            raise ValueError(f"Could not find a registry for {cls.__name__}")
        class_registry = _registry[cls.__name__]
        log.debug(
            "Looking for %s in registry %s with keys %s",
            name,
            cls.__name__,
            class_registry.keys(),
        )
        lowercase_classes = [k.lower().replace("_", "") for k in class_registry.keys()]
        if add_affixes:
            wanted_class = f"{cls.prefix.lower()}{name.lower()}{cls.suffix.lower()}"
        else:
            wanted_class = name.lower()
        wanted_class = wanted_class.replace("_", "")
        if wanted_class not in lowercase_classes:
            raise ValueError(
                f"Could not find {name} in {cls.__name__} registry "
                f"(tried {wanted_class})"
            )
        # Get the correct class name from the lowercase list
        class_index = lowercase_classes.index(wanted_class)
        class_name = list(class_registry.keys())[class_index]
        class_type: type[T] = class_registry[class_name]  # type: ignore
        # This cast is safe because the class registry has all classes of requested type
        return class_type(**kwargs)


def _count_registrable_paths(cls: type[T]) -> int:
    if Registrable in cls.__bases__:
        return 1 + sum(
            _count_registrable_paths(base)
            for base in cls.__bases__
            if base is not Registrable
        )
    else:
        return sum(_count_registrable_paths(base) for base in cls.__bases__)


def _check_only_one_registrable_in_hierarchy(cls: type[T]) -> bool:
    return _count_registrable_paths(cls) == 1


def _get_parent_registrable(cls: type[T]) -> str:
    candidate_parent: Optional[type] = None
    for base in cls.__bases__:
        if base is Registrable or _count_registrable_paths(base) == 1:
            candidate_parent = base
    if candidate_parent is not None:
        if getattr(candidate_parent, "transparent", False):
            return _get_parent_registrable(candidate_parent)
        else:
            return candidate_parent.__name__
    else:
        raise TypeError(  # pragma: no cover
            f"Could not find parent Registrable for {cls.__name__}"
        )
