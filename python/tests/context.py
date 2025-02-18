# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

GlobalConfig.verbose = True


def test_context_creation() -> None:
    # Test that a context can be created and defaults are correct.
    ctx = Context.primary()
    ctx = Context.primary()
    ctx = Context.primary()
    assert ctx.execution_mode.name in ('EAGER', 'DEFERRED')
    assert isinstance(ctx.os_name, str)
    assert isinstance(ctx.cpu_name, str)
    assert ctx.cpu_virtual_cores >= 1
    assert ctx.cpu_physical_cores >= 1
