from typing import Generator

import pytest

from revelio.registry.registry import Registrable, _registry


@pytest.fixture(autouse=True)
def cleanup_registry() -> Generator:
    _registry.clear()
    yield


def test_registrable_cannot_be_constructed() -> None:
    with pytest.raises(TypeError):
        Registrable()


def test_registrable_registers() -> None:
    class Abstract(Registrable):
        pass

    class Concrete(Abstract):
        pass

    assert type(Registrable.find(Abstract, "concrete")) is Concrete
    with pytest.raises(ValueError):
        Registrable.find(Abstract, "nonexistent")


def test_registrable_registers_with_suffix() -> None:
    class AbstractSuffix(Registrable):
        suffix = "Suffix"

    class ConcreteSuffix(AbstractSuffix):
        pass

    assert type(Registrable.find(AbstractSuffix, "concrete")) is ConcreteSuffix
    with pytest.raises(ValueError):
        Registrable.find(AbstractSuffix, "concretesuffix")


def test_registrable_registers_with_prefix() -> None:
    class PrefixAbstract(Registrable):
        prefix = "Prefix"

    class PrefixConcrete(PrefixAbstract):
        pass

    assert type(Registrable.find(PrefixAbstract, "concrete")) is PrefixConcrete
    with pytest.raises(ValueError):
        Registrable.find(PrefixAbstract, "prefixconcrete")


def test_registrable_registers_with_prefix_and_suffix() -> None:
    class PreAbstractSuf(Registrable):
        prefix = "Pre"
        suffix = "Suf"

    class PreConcreteSuf(PreAbstractSuf):
        pass

    assert type(Registrable.find(PreAbstractSuf, "concrete")) is PreConcreteSuf
    with pytest.raises(ValueError):
        Registrable.find(PreAbstractSuf, "preconcretesuf")


def test_multiple_hierarchy_fails() -> None:
    class A(Registrable):
        pass

    class B(A):
        pass

    class C(A):
        pass

    with pytest.raises(TypeError):

        class D(B, C):
            pass


def test_conflicting_names_fails() -> None:
    class Foo(Registrable):
        pass

    class FooChild(Foo):
        pass

    with pytest.raises(TypeError):

        class FOOCHILD(Foo):
            pass


def test_find_nonexistent_registry_fails() -> None:
    class Foo:
        pass

    with pytest.raises(ValueError):
        Registrable.find(Foo, "bar")
