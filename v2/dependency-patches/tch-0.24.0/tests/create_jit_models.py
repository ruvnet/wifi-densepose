import torch
from typing import Dict, Tuple

class Foo(torch.jit.ScriptModule):
    def __init__(self, v):
        super(Foo, self).__init__()
        self.register_buffer('value', v)

    @torch.jit.script_method
    def forward(self, x, y):
        return 2 * x + y + self.value

foo = Foo(torch.Tensor([42.0]))
foo.save('foo.pt')

class Foo1(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo1, self).__init__()

    @torch.jit.script_method
    def forward(self, x, y):
        return 2 * x + y

foo = Foo1()
script_foo = torch.jit.script(foo)
script_foo.save('foo1.pt')

class Foo2(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo2, self).__init__()

    @torch.jit.script_method
    def forward(self, x, y):
        return (2 * x + y, x - y)

foo = Foo2()
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
traced_foo.save('foo2.pt')

class Foo3(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo3, self).__init__()

    @torch.jit.script_method
    def forward(self, x):
        result = x[0]
        for i in range(x.size(0)):
            if i: result = result * x[i]
        return result

foo = Foo3()
foo.save('foo3.pt')

from typing import Tuple, List

class Foo4(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo4, self).__init__()

    @torch.jit.script_method
    def forward(self, x: Tuple[float, float, int]):
        return x[0] + x[1] * x[2]

foo = Foo4()
foo.save('foo4.pt')

class Foo5(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo5, self).__init__()

    @torch.jit.script_method
    def forward(self, xs: List[str]):
      return [x[:-1] for x in xs]

foo = Foo5()
foo.save('foo5.pt')

@torch.jit.script
class TorchScriptClass:
    def __init__(self, x: torch.Tensor):
        self.x = x

    def y(self) -> torch.Tensor:
        return self.x * 2

@torch.jit.script
def foo_6(x: torch.Tensor):
    return TorchScriptClass(x)


foo_6.save("foo6.pt")

# https://github.com/LaurentMazare/tch-rs/issues/475
@torch.jit.script
class InputObject:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar

class TorchScriptExample(torch.jit.ScriptModule):
    @torch.jit.script_method
    def add_them(self, data: InputObject) -> torch.Tensor:
        return data.foo + data.bar

    @torch.jit.script_method
    def make_input_object(self, foo, bar):
        return InputObject(foo, bar)

foo_7 = TorchScriptExample()
foo_7.save("foo7.pt")

# https://github.com/LaurentMazare/tch-rs/issues/597
class DictExample(torch.jit.ScriptModule):
    @torch.jit.script_method
    def generate(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return batch["foo"], batch["bar"]

foo_8 = DictExample()
foo_8.save("foo8.pt")
