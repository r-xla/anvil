from __future__ import annotations

import pytree
import numpy as np

from contextlib import contextmanager
from typing import Any, Callable

import numbers  # for type info

microjax_types = (numbers.Number, np.number, np.ndarray)


class OPS:
    ADD = "add"
    MUL = "mul"
    NEG = "neg"
    RECIP = "recip"
    EXP = "exp"
    SIN = "sin"


def add(*args):
    return bind_single(OPS.ADD, *args)


def mul(*args):
    return bind_single(OPS.MUL, *args)


def neg(x):
    return bind_single(OPS.NEG, x)


def recip(x):
    return bind_single(OPS.RECIP, x)


def exp(x):
    return bind_single(OPS.EXP, x)


def sin(x):
    return bind_single(OPS.SIN, x)


def cos(x):
    return sin(x + np.pi / 2)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def silu(x):
    return x * sigmoid(x)


# <basic interpreter>
class Interpreter:
    def __init__(self, level: int = 0, *args, **kwargs):
        self.level = level

    def process_primitive(self, prim, boxes, params):
        "in this function, either you process primitives or you unbox and send to lower level interpreter"
        raise NotImplementedError


# =========================================================
# this is global stack that have level and interpreter type
STACK: list[Interpreter] = []


def push_interpreter(interpreter: Interpreter):
    STACK.append(interpreter)
    return STACK


def pop_interpreter() -> Interpreter:
    return STACK.pop()


@contextmanager
def interpreter_context(interpreter_type: Interpreter):
    stack_item = interpreter_type(level=len(STACK))
    push_interpreter(stack_item)
    try:
        yield stack_item
    finally:
        pop_interpreter()


# =========================================================
def ensure_box(x):
    if isinstance(x, Box):
        return x.aval
    if isinstance(x, microjax_types):
        return ContrateArray(np.asarray(x))
    assert False, f"Unsupported type: {type(x)}"


class Box:
    _interpreter: Interpreter

    @property
    def aval(self):
        raise NotImplementedError

    def full_lower_to_ir(self):
        return self

    def __add__(self, other):
        return self.aval.add(self, other)

    def __radd__(self, other):
        return self.aval.add(other, self)

    def __mul__(self, other):
        return self.aval.mul(self, other)

    def __rmul__(self, other):
        # print(self.aval)
        return self.aval.mul(other, self)

    def __neg__(self):
        return self.aval.neg(self)

    def __sub__(self, other):
        return self.aval.add(self, neg(other))

    def __rsub__(self, other):
        return self.aval.add(other, neg(self))

    def __truediv__(self, other):
        return self.aval.mul(self, recip(other))

    def __rtruediv__(self, other):
        return self.aval.mul(other, recip(self))

    def __iadd__(self, other):
        return self.aval.add(self, other)

    def __imul__(self, other):
        return self.aval.mul(self, other)

    def __isub__(self, other):
        return self.aval.add(self, neg(other))

    def __itruediv__(self, other):
        return self.aval.mul(self, recip(other))


# array wrapper
class ContrateArray:
    def __init__(self, primal):
        self._interpreter = STACK[0]
        self.primal = primal
        self.shape = primal.shape
        self.dtype = primal.dtype

    def ensure_box(self, x):
        if isinstance(x, Box):
            return x
        if isinstance(x, microjax_types):
            return ContrateArray(np.asarray(x))
        assert False, f"Unsupported type: {type(x)}"

    def full_lower_to_ir(self):
        return self

    add = staticmethod(add)
    mul = staticmethod(mul)
    neg = staticmethod(neg)
    recip = staticmethod(recip)
    sin = staticmethod(sin)
    cos = staticmethod(cos)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return add(self, neg(other))

    def __rsub__(self, other):
        return add(other, neg(self))

    def __truediv__(self, other):
        return mul(self, recip(other))

    def __rtruediv__(self, other):
        return mul(other, recip(self))

    def __iadd__(self, other):
        return add(self, other)

    def __imul__(self, other):
        return mul(self, other)

    def __isub__(self, other):
        return add(self, neg(other))

    def __itruediv__(self, other):
        return mul(self, recip(other))


class EvalRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
            OPS.NEG: self.neg,
            OPS.RECIP: self.recip,
            OPS.EXP: self.exp,
            OPS.SIN: self.sin,
        }

    def __getitem__(self, op):
        return self.rules[op]

    def add(self, primals, *args):
        x, y = primals
        return [x + y]

    def mul(self, primals, *args):
        x, y = primals
        return [x * y]

    def neg(self, primals, *args):
        (x,) = primals
        return [-x]

    def recip(self, primals, *args):
        (x,) = primals
        return [1 / x]

    def exp(self, primals, *args):
        (x,) = primals
        return [np.exp(x)]

    def sin(self, primals, *args):
        (x,) = primals
        return [np.sin(x)]


class EvalInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = EvalRules()

    def pure(self, val):
        return val

    def process_primitive(self, prim, boxes, params):
        return self.rules[prim](boxes, *params)


def find_top_interpreter(args):
    interpreters = []
    for item in args:
        if isinstance(item, Box):
            interpreters.append(item._interpreter)

    if interpreters:
        return max(interpreters, key=lambda x: x.level)

    # if no interpreters are found, return the default EvalInterpreter
    return STACK[0]


def full_raise(interpreter: Interpreter | Any, out) -> Box | JVPBox:
    """
    if interpreter need values boxed
    if out is not boxed, box it (using interpreter.pure)
    ie. raise out to the box level
    """
    if not isinstance(out, Box):
        return interpreter.pure(out)
    return out


def full_lower_to_ir(val):
    if isinstance(val, Box):
        return val.full_lower_to_ir()
    return val


def bind(prim, *args, **params):
    interpreter = find_top_interpreter(args)
    # this will raise the boxes to the top level
    # eg converts primitive values to Boxes if interpreter is not the top level
    boxes = [full_raise(interpreter, arg) for arg in args]
    outs = interpreter.process_primitive(prim, boxes, params)
    return [out for out in outs]


def bind_single(prim, *args, **params):
    (out,) = bind(prim, *args, **params)
    return out


### Push EvalInterpreter at bottom of the stack
push_interpreter(EvalInterpreter())


# </basic interpreter>
###

# =========================================================
# Jacobian Vector Product (JVP)
# forward mode Automatic Differentiation
# =========================================================


class JVPBox(Box):
    def __init__(self, interpreter: Interpreter, primal, tangent) -> None:
        super().__init__()
        self._interpreter = interpreter
        self.primal = primal
        self.tangent = tangent

    def __repr__(self):
        return f"JVPBox (primal={self.primal}, tangent={self.tangent})"

    @property
    def aval(self):
        return ensure_box(self.primal)


class JVPRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
            OPS.NEG: self.neg,
            OPS.RECIP: self.recip,
            OPS.EXP: self.exp,
            OPS.SIN: self.sin,
        }

    # dont forget to return tuple(primals),tuple(tangents)
    def __getitem__(self, op):
        return self.rules[op]

    @staticmethod
    def add(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return (x + y,), (x_dot + y_dot,)

    @staticmethod
    def mul(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return (x * y,), (x_dot * y + x * y_dot,)

    @staticmethod
    def neg(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return (-x,), (-x_dot,)

    @staticmethod
    def recip(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        y = 1 / x
        return (y,), (-y * y * x_dot,)

    @staticmethod
    def exp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        y = exp(x)
        return (y,), (y * x_dot,)

    @staticmethod
    def sin(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return (sin(x),), (cos(x) * x_dot,)


class JVPInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = JVPRules()

    def pure(self, val):
        return JVPBox(self, val, 0.0)

    def process_primitive(self, prim, boxes, params):
        primals = [box.primal for box in boxes]
        tangents = [box.tangent for box in boxes]
        op = self.rules[prim]
        primals_out, tangents_out = op(primals, tangents, **params)

        result = []
        for p, t in zip(primals_out, tangents_out):
            result.append(JVPBox(self, p, t))
        return result


def jvp_simple(func, primals, tangents):
    with interpreter_context(JVPInterpreter) as iptr:
        box_in = [JVPBox(iptr, x, t) for x, t in zip(primals, tangents)]
        out = func(*box_in)
        box_out = full_raise(iptr, out)
        primal_out, tangent_out = box_out.primal, box_out.tangent
    return primal_out, tangent_out


def deriv(function):
    def jvp_forward(input_value):
        _, gradient = jvp_simple(function, (input_value,), (1.0,))
        return gradient

    return jvp_forward


# =========================================================
if __name__ == "__main__":
    print("## Forward Mode Automatic Differentiation (JVP) ##")

    def func(x):
        return 3 * x * x * x + 2 * x * x + 2 * x

    x = 3.14

    f = func
    print(f"f(x) = {f(x)}")

    f = deriv(func)
    print(f"f'(x) = {f(x)}")

    f = deriv(deriv(func))
    print(f"f''(x) = {f(x)}")
    # exit()
    f = deriv(deriv(deriv(func)))
    print(f"f'''(x) = {f(x)}")

    print("-" * 100)


# =========================================================
# Vector Jacobian Product (VJP)
# reverse mode Automatic Differentiation
# =========================================================


class Node:
    def __init__(self, vjp: Callable, parents: list[Node]) -> None:
        self.vjp = vjp
        self.parents = parents

    @property
    def is_leaf(self):
        return len(self.parents) == 0


def get_leaf_nodes() -> Node:
    return Node(None, [])


class VJPRules:
    def __init__(self):
        self.rules = {
            OPS.ADD: self.add,
            OPS.MUL: self.mul,
            OPS.NEG: self.neg,
            OPS.RECIP: self.recip,
            OPS.EXP: self.exp,
            OPS.SIN: self.sin,
        }
        """
        Jax define one of vjp or jvp rules
        it derives one from the other
        but this is much more simple to understand
        """

    def __getitem__(self, op):
        return self.rules[op]

    def add(self, primals):
        x, y = primals

        def vjp_add(grad):
            return grad, grad

        return (x + y,), vjp_add

    def mul(self, primals):
        x, y = primals

        def vjp_mul(grad):
            return grad * y, grad * x

        return (x * y,), vjp_mul

    def tanh(self, primals):
        (x,) = primals
        y = tanh(x)

        def vjp_tanh(grad):
            return ((1 - y * y) * grad,)

        return (y,), vjp_tanh

    def neg(self, primals):
        (x,) = primals

        def vjp_neg(grad):
            return (-grad,)

        return (-x,), vjp_neg

    def recip(self, primals):
        (x,) = primals
        y = 1 / x

        def vjp_recip(grad):
            return (-y * y * grad,)

        return (y,), vjp_recip

    def exp(self, primals):
        (x,) = primals
        y = exp(x)

        def vjp_exp(grad):
            return (y * grad,)

        return (y,), vjp_exp

    def sin(self, primals):
        (x,) = primals
        y = sin(x)

        def vjp_sin(grad):
            return (cos(x) * grad,)

        return (y,), vjp_sin


class VJPInterpreter(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = VJPRules()

    def pure(self, val):
        return VJPBox(self, val, get_leaf_nodes())

    def process_primitive(self, prim, boxes, params):
        primals_in = [box.primal for box in boxes]
        nodes_in = [box.node for box in boxes]
        op = self.rules[prim]
        primals_out, vjp_out = op(primals_in, **params)
        nodes_out = [Node(vjp_out, nodes_in)]
        result = []
        for p, n in zip(primals_out, nodes_out):
            result.append(VJPBox(self, p, n))
        return result


class VJPBox(Box):
    def __init__(self, interpreter: VJPInterpreter, primal, node: Node) -> None:
        super().__init__()
        self._interpreter = interpreter
        self.primal = primal
        self.node = node

    def __repr__(self):
        return f"VJPBox (primal={self.primal}, node={self.node})"

    def full_lower_to_ir(self):
        return self

    @property
    def aval():
        return ensure_box(self.primal)


def vjp_simple(func, *args):
    with interpreter_context(VJPInterpreter) as iptr:
        box_in = [VJPBox(iptr, x, get_leaf_nodes()) for x in args]
        out = func(*box_in)
        box_out = full_raise(iptr, out)
        in_nodes = [box.node for box in box_in]
        out_node = box_out.node
        primal_out = box_out.primal

        def func_vjp(grad):
            return backward_pass(in_nodes, out_node, grad)

    return primal_out, func_vjp



def backward_pass(in_nodes, out_node, gradient):
    node_map = {id(out_node): gradient}

    topo_sorted = toposort(out_node)
    for node in topo_sorted:
        node_grad = node_map.pop(id(node))

        input_grads = node.vjp(node_grad)

        for input_grad, parent in zip(input_grads, node.parents):
            parent_id = id(parent)
            node_map[parent_id] = add_grads(node_map.get(parent_id), input_grad)

    return [node_map.get(id(node)) for node in in_nodes]


def add_grads(grad1, grad2):
    if grad1 is None:
        return grad2
    return grad1 + grad2


def toposort(end_node):
    def _toposort(seen, node):
        result = []
        if id(node) not in seen:
            seen.add(id(node))
            for p in node.parents:
                result.extend(_toposort(seen, p))
            result.append(node)
        return result

    return reversed([n for n in _toposort(set(), end_node) if n.parents])


def grad(func):
    def grad_func(*args):
        _, backward = vjp_simple(func, *args)
        return backward(1)[0]

    return grad_func


def func(x):
    # return x*x
    return 3 * x * x * x + 2 * x * x + 2 * x


if __name__ == "__main__":
    x = 3.14

    print("## Reverse Mode Automatic Differentiation (VJP) ##\n")
    f = func
    print(f"f(x) = {f(x)}")

    f = grad(func)
    print(f"f'(x) = {f(x)}")

    f = grad(grad(func))
    print(f"f''(x) = {f(x)}")

    f = grad(grad(grad(func)))
    print(f"f'''(x) = {f(x)}")

    print("-" * 100, "\n")

    print("## Composition of Forward and Backward #\n")
    print(f"Forward on Backward {grad(deriv(func))(x)}")
    print(f"Backward on Forward {deriv(grad(func))(x)}")


#### TODO: Pytree


### Refinement of JVP
def jvp_flat(func, primals, tangents):
    with interpreter_context(JVPInterpreter) as iptr:
        tracers_in = [JVPBox(iptr, x, t) for x, t in zip(primals, tangents)]

        outs = func(*tracers_in)

        tracers_out = [full_raise(iptr, out) for out in outs]

        primals_out, tangents_out = [], []
        for t in tracers_out:
            primals_out.append(t.primal)
            tangents_out.append(t.tangent)

    return primals_out, tangents_out


def jvp(func, primals, tangents):
    # Flatten the primals and tangents into flat lists
    primals_flat, in_tree = pytree.tree_flatten(primals)
    tangents_flat, in_tree2 = pytree.tree_flatten(tangents)
    assert in_tree == in_tree2, "Input trees for primals and tangents must match"

    # Flatten the function f according to the input tree structure
    func_flat, out_tree = pytree.flatten_fun(func, in_tree)

    # forward pass
    primals_out_flat, tangents_out_flat = jvp_flat(
        func_flat, primals_flat, tangents_flat
    )

    assert len(out_tree) == 1, "out tree dict must have only one item"
    out_tree: pytree.PyNode = out_tree["tree"]

    primals_out = pytree.tree_unflatten(primals_out_flat, out_tree)
    tangents_out = pytree.tree_unflatten(tangents_out_flat, out_tree)

    return primals_out, tangents_out


def deriv(func, argnums=0):
    if isinstance(argnums, int):
        argnums = [argnums]

    def jvp_forward(*input_value):
        # pass tangent 1 for argnums and 0 for others
        tangents = tuple(
            pytree.nested_ones_like(x) if idx in argnums else pytree.nested_zero_like(x)
            for idx, x in enumerate(input_value)
        )

        _, gradient = jvp(func, input_value, tangents)

        return gradient

    return jvp_forward


def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def func(x, y):
    k = tanh(x) * 2.0 + y * y
    z = -y + k
    return {"hi": z, "there": [x, y]}


if __name__ == "__main__":
    print("------------------")
    print("## pytree.py ##")
    x = 3.14
    y = 2.71
    print(deriv(func, argnums=0)(x, y))


#####
### Refinement of VJP


def add_grads(grad1, grad2):
    if grad1 is None:
        return grad2
    return grad1 + grad2


def toposort(end_nodes):
    def _toposort(seen, node):
        result = []
        if node not in seen:
            seen.add(node)
            for p in node.parents:
                result.extend(_toposort(seen, p))
            result.append(node)
        return result

    outs = []
    seen = set()
    topo_sorted = []
    for end_node in end_nodes:
        topo_sorted.extend(_toposort(seen, end_node))

    for node in topo_sorted:
        if node.parents:
            outs.append(node)
    result = reversed(outs)
    return list(result)


def backward_pass(in_nodes, out_nodes, gradient):
    node_map = {out_node: g for g, out_node in zip(gradient, out_nodes)}

    topo_sorted = toposort(out_nodes)
    for node in topo_sorted:
        node_grad = node_map.pop(node)

        input_grads = node.vjp(node_grad)

        for input_grad, parent in zip(input_grads, node.parents):
            node_map[parent] = add_grads(node_map.get(parent), input_grad)

    return [node_map.get(node) for node in in_nodes]


def add_grads(grad1, grad2):
    if grad1 is None:
        return grad2
    return grad1 + grad2


def vjp_flat(func, args):
    with interpreter_context(VJPInterpreter) as iptr:
        box_in = [VJPBox(iptr, x, get_leaf_nodes()) for x in args]
        outs = func(*box_in)
        box_out = [full_raise(iptr, o) for o in outs]
        in_nodes = [box.node for box in box_in]
        out_nodes = [box.node for box in box_out]
        out_primals = [box.primal for box in box_out]

        def func_vjp(grad):
            return backward_pass(in_nodes, out_nodes, grad)

    return out_primals, func_vjp


def vjp(func, primals):
    # Flatten the primals and tangents into flat lists
    primals_flat, in_tree = pytree.tree_flatten(primals)

    # Flatten the function f according to the input tree structure
    func_flat, out_tree = pytree.flatten_fun(func, in_tree)

    # forward pass
    primals_out_flat, vjp_func = vjp_flat(
        func_flat,
        primals_flat,
    )

    assert len(out_tree) == 1, "out tree dict must have only one item"
    out_tree: pytree.PyNode = out_tree["tree"]

    primals_out = pytree.tree_unflatten(primals_out_flat, out_tree)

    return primals_out, vjp_func


def grad(func, argnums=0):
    if isinstance(argnums, int):
        argnums = [argnums]

    def vjp_func(*input_value):
        result, vjp_func = vjp(func, input_value)

        ones = pytree.nested_ones_like(result)
        flat, _ = pytree.tree_flatten(ones)
        grads = vjp_func(flat)
        _, in_tree = pytree.tree_flatten(input_value)
        grads = pytree.tree_unflatten(grads, in_tree)
        grads = tuple(g for idx, g in enumerate(grads) if idx in argnums)
        return grads[0] if len(argnums) == 1 else grads

    return vjp_func


def value_and_grad(func, argnums=0):
    if isinstance(argnums, int):
        argnums = [argnums]

    def vjp_forward(*input_value):
        result, vjp_func = vjp(func, input_value)

        # <hack>jax dont do this nasted ones funnny busniess
        # it just requires output to be scalar
        # but I you can pass one to all output nodes
        # which is effectively like result = sum(result) I dont have redution op
        # basically result.sum().backward() in pytorch
        ones = pytree.nested_ones_like(result)
        flat, _ = pytree.tree_flatten(ones)
        # </hack>

        # backward pass
        grads = vjp_func(flat)

        output, in_tree = pytree.tree_flatten(input_value)
        grads = pytree.tree_unflatten(grads, in_tree)

        grads = tuple(g for idx, g in enumerate(grads) if idx in argnums)

        return result, grads[0] if len(argnums) == 1 else grads

    return vjp_forward


if __name__ == "__main__":
    print("------------------")
    PI = 3.14159265358979323846
    x = 3.14

    x = 3.14
    y = 2.71

    def func(x, y):
        k = tanh(x) * 2.0 + y * y
        z = -y + k
        return z

    print("MicroJAX: ", grad(func)(x, y))

    try:
        import jax
        import jax.numpy as jnp

        def func(x, y):
            k = jnp.tanh(x) * 2.0 + y * y
            z = -y + k
            return z

        print("JAX: ", jax.grad(func)(x, y))
    except:
        print("Jax not installed for comparison")
