# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>


def test_import_magnetron() -> None:
    import magnetron

    assert magnetron.__version__ is not None


def test_simple_exec() -> None:
    import magnetron as mag

    a = mag.tensor([1, 4, 1])
    assert a.max()[0] == 4
