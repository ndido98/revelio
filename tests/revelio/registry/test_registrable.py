from typing import Any, Generator

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

    assert type(Abstract.find("concrete")) is Concrete
    with pytest.raises(ValueError):
        Abstract.find("nonexistent")


def test_registrable_registers_with_suffix() -> None:
    class AbstractSuffix(Registrable):
        suffix = "Suffix"

    class ConcreteSuffix(AbstractSuffix):
        pass

    assert type(AbstractSuffix.find("concrete")) is ConcreteSuffix
    with pytest.raises(ValueError):
        AbstractSuffix.find("concretesuffix")


def test_registrable_registers_with_prefix() -> None:
    class PrefixAbstract(Registrable):
        prefix = "Prefix"

    class PrefixConcrete(PrefixAbstract):
        pass

    assert type(PrefixAbstract.find("concrete")) is PrefixConcrete
    with pytest.raises(ValueError):
        PrefixAbstract.find("prefixconcrete")


def test_registrable_registers_with_prefix_and_suffix() -> None:
    class PreAbstractSuf(Registrable):
        prefix = "Pre"
        suffix = "Suf"

    class PreConcreteSuf(PreAbstractSuf):
        pass

    assert type(PreAbstractSuf.find("concrete")) is PreConcreteSuf
    with pytest.raises(ValueError):
        PreAbstractSuf.find("preconcretesuf")


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


def test_registrable_with_args_kwargs() -> None:
    class Foo(Registrable):
        def __init__(self, foo: str, bar: str):
            self.foo = foo
            self.bar = bar
            super().__init__()

    class FooChild(Foo):
        def __init__(self, foo: str, bar: str, baz: str, *args: Any, **kwargs: Any):
            self.baz = baz
            super().__init__(foo, bar, *args, **kwargs)

    x: FooChild = Foo.find("foochild", foo="123", bar="456", baz="789")  # type: ignore
    assert x.foo == "123"
    assert x.bar == "456"
    assert x.baz == "789"


def test_transparent() -> None:
    class Foo(Registrable):
        pass

    class Bar(Foo):
        transparent: bool = True

    class Baz(Bar):
        pass

    assert type(Foo.find("baz")) is Baz
    with pytest.raises(ValueError):
        Bar.find("baz")


def test_snake_case() -> None:
    class PreAbstractSuf(Registrable):
        prefix = "Pre"
        suffix = "Suf"

    class PreConcreteLongNameSuf(PreAbstractSuf):
        pass

    assert type(PreAbstractSuf.find("concretelongname")) is PreConcreteLongNameSuf
    assert type(PreAbstractSuf.find("concrete_long_name")) is PreConcreteLongNameSuf
