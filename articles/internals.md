# Internals

## Transforming Code

While a real anvil is made for reshaping metal, this package is a tool
for reshaping code. We refer to such a rewriting of code as a
**transformation**, of which there are three types:

1.  `R` \\\rightarrow\\ `AnvilGraph`: Generic `R` functions are too
    complicated to handle, so the first step in {anvil} is always to
    convert them into a computational `anvil::Graph` object via
    **tracing**. Such a `AnvilGraph` is similar to `JAXExpr` objects in
    `JAX`. It operates only on `AnvilTensor` objects and applies
    `anvil::Primitive` operations to them.
2.  `AnvilGraph` \\\rightarrow\\ `AnvilGraph`: It is possible to
    transform `AnvilGraph`s into other `AnvilGraph`s. Their purpose is
    to change the functionality of the code. At the time of writing,
    there is essentially only one such transformation, namely
    backward-mode automatic differentiation via
    [`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md).
3.  `AnvilGraph` \\\rightarrow\\ `Executable`: In order to perform the
    actual computation, the `AnvilGraph` needs to be converted into an
    executable. Currently, we only support the XLA backend (via
    `stablehlo` and `pjrt`), but we are working on an experimental
    [quickr](https://github.com/t-kalinowski/quickr) backend.

### Tracing R Functions into Graphs

All functionality in the {anvil} package is centered around the
`anvil::Graph` class. While it is in principle possible to create
`AnvilGraph`s by hand, these are usually created by tracing R functions.
In general, when we want to convert some code into another form (in our
case, R Code into a `AnvilGraph`), there are two approaches:

1.  Static analysis, which would require operating on the abstract
    syntax tree (AST) of the code.
2.  Dynamic analysis (aka “tracing”), which executes the code and
    records selected operations.

The former approach is followed by the {quickr} package, while we go
with tracing. We start with a simple, yet illustrative example that
either adds or multiplies two inputs `x` and `y` depending on the value
of `op`.

``` r
library(anvil)
f <- function(x, y, op) {
  if (op == "add") {
    nv_add(x, y)
  } else if (op == "mul") {
    nv_mul(x, y)
  } else {
    stop("Unsupported operation")
  }
}
```

To do this, we use
[`anvil::trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md),
which takes in an `R` function and a list of `AbstractTensor` inputs
that specify the types of the inputs.

``` r
aten <- nv_aten("f32", c())
aten
```

    ## AbstractTensor(dtype=f32, shape=)

``` r
graph <- trace_fn(f, list(x = aten, y = aten, op = "mul"))
graph
```

    ## <AnvilGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Body:
    ##     %1: f32[] = mul(%x1, %x2)
    ##   Outputs:
    ##     %1: f32[]

The output of
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md) is
now a `AnvilGraph` object that represents the computation. The fields of
the `AnvilGraph` are:

- `inputs`, which are `GraphNode`s that represent the inputs to the
  function.
- `outputs`, which are `GraphNode`s that represent the outputs of the
  function.
- `calls`, which are `PrimitiveCall`s that take in `GraphNode`s (and
  parameters) and produce output `GraphNode`s.
- `in_tree`, `out_tree`, which we will cover later (do we??)

What happens during
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md) is
that a new `GraphDescriptor` is created and the inputs `x` and `y` are
converted into
[`anvil::GraphBox`](https://r-xla.github.io/anvil/reference/GraphBox.md)
objects. Then, the function `f` is simply evaluated with the `GraphBox`
objects as inputs. During this evaluation, we need to distinguish
between two cases:

1.  A “standard” `R` function is called: Here, nothing special happens
    and the function is simply evaluated.
2.  An `anvil` function is called: Here, the operation that underlies
    the function is recorded in the `GraphDescriptor`.

The evaluation of the `if` statement is an example for the first
category. Because we set `op = "mul"`, only the second branch is
executed. Then, we are calling `nv_mul`, which attaches a
`PrimitiveCall` that represents the multiplication of the two tensors to
the `$calls` of the `GraphDescriptor`. Note that the `nv_mul` is itself
not primitive, but performs some type promotion and broadcasting if
needed, before calling into the primitive
[`nvl_mul()`](https://r-xla.github.io/anvil/reference/nvl_mul.md).

A `PrimitiveCall` object consists of the following fields:

- `primitive`: The primitive function that was called.
- `inputs`: The inputs to the primitive function.
- `params`: The parameters (non-tensors) to the primitive function.
- `outputs`: The outputs of the primitive function.

When the evaluation of `f` is complete, the `$outputs` field of the
`GraphDescriptor` is set and the `AnvilGraph` is subsequently created
from the `GraphDescriptor`. The only difference between the `AnvilGraph`
and the `GraphDescriptor` is that the latter has some utility fields
that are useful during graph creation, but for the purposes of this
tutorial, you can think of them as being the same.

### Transforming Graphs into other Graphs

Once the `R` function is staged out into a simpler format, it is ready
to be transformed. The {anvil} package does not in any way dictate how
such a `AnvilGraph` to `AnvilGraph` transformation can be implemented.
For most interesting transformations, however, we need to store some
information for each {anvil} primitive function. In the case of the
gradient, we need to store the derivative rules. For this,
`anvil::Primitive` objects have a `$rules` field that can be populated.
The derivative rules are stored as functions under the `"backward"`
name. We can access a primitive by it’s name via the
[`prim()`](https://r-xla.github.io/anvil/reference/prim.md) function:

``` r
prim("mul")$rules[["backward"]]
```

    ## function (inputs, outputs, grads, .required) 
    ## {
    ##     lhs <- inputs[[1L]]
    ##     rhs <- inputs[[2L]]
    ##     grad <- grads[[1L]]
    ##     list(if (.required[[1L]]) nvl_mul(grad, rhs), if (.required[[2L]]) nvl_mul(grad, 
    ##         lhs))
    ## }
    ## <bytecode: 0x55711f65db80>
    ## <environment: namespace:anvil>

The
[`anvil::transform_gradient`](https://r-xla.github.io/anvil/reference/transform_gradient.md)
function uses these rules to compute the gradient of a function. For
this specific transformation, we are walking the graph backwards and
apply the derivative rules, which will append the “backward pass” to the
graph. Besides the forward graph, the transformation takes in the `wrt`
argument, which specifies with respect to which arguments to compute the
gradient.

``` r
bwd_graph <- transform_gradient(graph, wrt = c("x", "y"))
bwd_graph
```

    ## <AnvilGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = mul(%x1, %x2)
    ##     %2: f32[] = mul(%c1, %x2)
    ##     %3: f32[] = mul(%c1, %x1)
    ##   Outputs:
    ##     %2: f32[]
    ##     %3: f32[]

### Lowering a Graph

In order to execute a `AnvilGraph`, we need to convert it into a – wait
for it – executable. Here, we show how to compile using the XLA backend.
First, we will translate the `AnvilGraph` into the StableHLO
representation via the {stablehlo} package. Then, we will compile this
program using the XLA compiler that is accessible via the {pjrt}
package.

Like for the gradient transformation, the rules of how to do this
transformation are stored in the `$rules` fields of the primitives.

``` r
prim("mul")$rules[["stablehlo"]]
```

    ## function (lhs, rhs) 
    ## {
    ##     list(stablehlo::hlo_multiply(lhs, rhs))
    ## }
    ## <bytecode: 0x55711f660d38>
    ## <environment: namespace:anvil>

The
[`anvil::stablehlo`](https://r-xla.github.io/anvil/reference/stablehlo.md)
function will create a
[`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)
object and will sequentially translate the `PrimitiveCall`s into
StableHLO operations.

``` r
func <- stablehlo(graph)[[1L]]
func
```

    ## func.func @main (%0: tensor<f32>, %1: tensor<f32>) -> tensor<f32> {
    ## %2 = "stablehlo.multiply" (%0, %1): (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    ## "func.return"(%2): (tensor<f32>) -> ()
    ## }

Now, we can compile the function via `pjrt_compile()`.

``` r
hlo_str <- stablehlo::repr(func)
program <- pjrt::pjrt_program(src = hlo_str, format = "mlir")
exec <- pjrt::pjrt_compile(program)
```

To run the function, we need to extract the underlying buffers from the
tensors before passing them to the executable, which will output a
`PJRTBuffer` that we can easily convert to an `AnvilTensor`.

``` r
x <- nv_scalar(3, "f32")
y <- nv_scalar(4, "f32")
out <- pjrt::pjrt_execute(exec, x$tensor, y$tensor)
out
```

    ## PJRTBuffer 
    ##  12
    ## [ CPUf32{} ]

``` r
nv_tensor(out)
```

    ## AnvilTensor
    ##  12
    ## [ CPUf32{} ]

## The User Interface

In the previous section, we have shown how the transformations are
implemented under the hood. The actual user interface is a little more
convenient and follows the `JAX` interface.

### `jit()`

The [`jit()`](https://r-xla.github.io/anvil/reference/jit.md) function
allows to convert a regular `R` function into a Just-In-Time compiled
function that can be executed on `AnvilTensor`s. We apply it to our
simple example function, where we mark the non-tensor parameter `op` as
“static”. This means that the value of this parameter needs to be known
at compile time.

``` r
f_jit <-  jit(f, static = "op")
f_jit(x, y, "add")
```

    ## AnvilTensor
    ##  7
    ## [ CPUf32{} ]

One might think that
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md) first calls
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md),
then runs
[`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md),
followed by `pjrt_compile()`. This is, however, not what is happening,
as this requires the input types to be known. Instead, `f_jit` is a
“lazy” function that will only perform these steps once the inputs are
provided. However, if those steps were applied every time the `f_jit`
function is called, this would be very inefficient, because tracing and
compiling takes some time. Therefore, the function `f_jit` also contains
a cache (implemented as an
[`xlamisc::LRUCache`](https://rdrr.io/pkg/xlamisc/man/LRUCache.html)),
which will check whether there is already a compiled executable for the
given inputs. For this, the types of all `AnvilTensor`s need to match
exactly (data type and shape) and all static arguments need to be
identical. For example, if we run the function with `AnvilTensor`s of
the same type, but different values, the function won’t be recompiled,
which we can see by checking the size of the cache, which is already 1,
because we have called it on `x` and `y` above.

``` r
cache_size <- function(f) environment(f)$cache$size
cache_size(f_jit)
```

    ## [1] 1

After calling it with tensors of the same types and identical static
argument values, the size of the cache remains 1:

``` r
f_jit(nv_scalar(-99, "f32"), nv_scalar(2, "f32"), "add")
```

    ## AnvilTensor
    ##  -97
    ## [ CPUf32{} ]

``` r
cache_size(f_jit)
```

    ## [1] 1

When we execute the function with tensors of different `dtype` or
`shape`, the function will be recompiled:

``` r
f_jit(nv_scalar(1, "i32"), nv_scalar(2, "i32"), "add")
```

    ## AnvilTensor
    ##  3
    ## [ CPUi32{} ]

``` r
cache_size(f_jit)
```

    ## [1] 2

Also, if we provide different values for static arguments, the function
will be recompiled:

``` r
f_jit(nv_scalar(1, "f32"), nv_scalar(2, "f32"), "mul")
```

    ## AnvilTensor
    ##  2
    ## [ CPUf32{} ]

``` r
cache_size(f_jit)
```

    ## [1] 3

### `gradient()`

Just like [`jit()`](https://r-xla.github.io/anvil/reference/jit.md),
[`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md) also
returns a function that will lazily create the graph and transform it,
once the inputs are provided.

``` r
g <- gradient(f, wrt = c("x", "y"))
```

Calling `g()` on `AnvilTensor`s will not actually compute the gradient,
but instead just output the output types, c.f. the [debugging
vignette](https://r-xla.github.io/anvil/articles/debugging.md) for more.

``` r
g(x, y, "add")
```

    ## $x
    ## DebugBox(ConcreteTensor)
    ##  1
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## DebugBox(ConcreteTensor)
    ##  1
    ## [ CPUf32{} ]

If we want to actually compute the gradient, we need to wrap it in
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md).

``` r
g_jit <- jit(g, static = "op")
g_jit(x, y, "add")
```

    ## $x
    ## AnvilTensor
    ##  1
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvilTensor
    ##  1
    ## [ CPUf32{} ]

Moreover, we can also use `g` in another function:

``` r
h <- function(x, y) {
  z <- nv_add(x, y)
  g(z, x, "mul")
}
h_jit <- jit(h)
h_jit(x, y)
```

    ## $x
    ## AnvilTensor
    ##  3
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvilTensor
    ##  7
    ## [ CPUf32{} ]

So, what is happening here? Once the inputs `x` and `y` are provided to
`h_jit`, a new `GraphDescriptor` is created and the inputs `x` and `y`
are converted into `GraphBox` objects. Then, the addition of `x` and `y`
is recorded in the `GraphDescriptor`. The call into `g()` is a bit more
involved. First, a new `GraphDescriptor` is created and the forward
computation of `g` is recorded. Subsequently, the backward pass will be
added to the descriptor, after which it will be converted into a
`AnvilGraph`. This `AnvilGraph` will then be inlined into the parent
`GraphDescriptor` (representing the whole function `h`), which is then
converted into the main `AnvilGraph`. We can look at this graph below,
where `trace_fn` internally converts the `AnvilTensor`s `x` and `y` into
their abstract representation.

``` r
h_graph <- trace_fn(h, list(x = x, y = y))
h_graph
```

    ## <AnvilGraph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = add(%x1, %x2)
    ##     %2: f32[] = mul(%1, %x1)
    ##     %3: f32[] = mul(%c1, %x1)
    ##     %4: f32[] = mul(%c1, %1)
    ##   Outputs:
    ##     %3: f32[]
    ##     %4: f32[]

Afterwards, this graph is lowered to stableHLO and subsequently
compiled.

## More Internals

### Debug Mode

For how to use debug mode, see the [debugging
vignette](https://r-xla.github.io/anvil/articles/debugging.md).

Debug-mode is different from jit-mode, because we don’t have a context
that can initialize a main `GraphDescriptor`. For this reason, every
primitive initializes its own `GraphDescriptor` that is thrown away
after the primitive returns `DebugBox` objects. These `DebugBox` objects
are only for user-interaction and have a nice printer. Whenever a
primitive is evaluated, this `DebugBox` is converted to a `GraphBox`
object that is used for the actual evaluation via `maybe_box_tensorish`.
This ensures that we don’t have to duplicate any evaluation logic as the
graph-building functions only have to work with `GraphBox` objects.

What gets lost in debug mode is identity of values, because the
`GraphDescriptor` is thrown away. This means that we cannot say anything
about identity of values, only about their types.

Unfortunately, our current mode for detecting debug mode is whether a
`GraphDescriptor` is active. For this reason, we don’t allow calling
[`local_descriptor()`](https://r-xla.github.io/anvil/reference/local_descriptor.md)
in the global environment. Maybe we can improve this in the future, but
for now it seems to work.

### Constant Handling

Constants are handled specially in {anvil}. Consider the program below:

``` r
y <- nv_tensor(rnorm(1000000L))
graph <- trace_fn(function(x) {
  x + y + 1
}, list(x = nv_scalar(1L)))
graph
```

    ## <AnvilGraph>
    ##   Inputs:
    ##     %x1: i32?[]
    ##   Constants:
    ##     %c1: f32[1000000]
    ##   Body:
    ##     %1: f32[] = convert [dtype = f32, ambiguous = FALSE] (%x1)
    ##     %2: f32[1000000] = broadcast_in_dim [shape = 1000000, broadcast_dimensions = <any>] (%1)
    ##     %3: f32[1000000] = add(%2, %c1)
    ##     %4: f32?[1000000] = broadcast_in_dim [shape = 1000000, broadcast_dimensions = <any>] (1:f32?)
    ##     %5: f32[1000000] = add(%3, %4)
    ##   Outputs:
    ##     %5: f32[1000000]

Here, `y` is a closed-over constant and it is included in the
`$constants` field of the graph, just like the literal `1`.

``` r
graph$constants
```

    ## [[1]]
    ## GraphValue(ConcreteTensor(f32, (1000000)))

When compiling such a program to stableHLO, constants are treated
differently depending on their shape (we follow JAX’s approach here).
That is, constants with 1 element are **inlined** into the program,
whereas other constants are added as inputs to the stableHLO program.
This is because inlining large constants into the executable is
inefficient. However, if we didn’t inline small scalars, the compiler
would be unable to do constant folding.

``` r
out <- stablehlo(graph)
out[[1L]]
```

    ## func.func @main (%0: tensor<1000000xf32>, %1: tensor<i32>) -> tensor<1000000xf32> {
    ## %2 = "stablehlo.convert" (%1): (tensor<i32>) -> (tensor<f32>)
    ## %3 = "stablehlo.broadcast_in_dim" (%2) {
    ## broadcast_dimensions = array<i64>
    ## }: (tensor<f32>) -> (tensor<1000000xf32>)
    ## %4 = "stablehlo.add" (%3, %0): (tensor<1000000xf32>, tensor<1000000xf32>) -> (tensor<1000000xf32>)
    ## %5 = "stablehlo.constant" () {
    ## value = dense<1.00000000e+00> : tensor<f32>
    ## }: () -> (tensor<f32>)
    ## %6 = "stablehlo.broadcast_in_dim" (%5) {
    ## broadcast_dimensions = array<i64>
    ## }: (tensor<f32>) -> (tensor<1000000xf32>)
    ## %7 = "stablehlo.add" (%4, %6): (tensor<1000000xf32>, tensor<1000000xf32>) -> (tensor<1000000xf32>)
    ## "func.return"(%7): (tensor<1000000xf32>) -> ()
    ## }

``` r
out[[2L]]
```

    ## [[1]]
    ## GraphValue(ConcreteTensor(f32, (1000000)))

Also, before compiling, we remove unused constants. Captured constants
can become unused when we apply code transformations like below, where
the gradient of the function w.r.t. `x` does not depend on the captured
`y`:

``` r
f <- function(x) {
x + y
}
transform_gradient(trace_fn(f, list(x = nv_scalar(1))))
```

In principle, the compiler is able to do this itself, but because we
pass constants as inputs to the program, we need to handle it ourselves.

Further note that:

1.  R Literals are immediately embedded as literals into the program.
2.  Currently, constants with the same value (that refer to different
    `AnvilTensor`s) are not deduplicated, which we might change in the
    future.

### Nested Inputs and Outputs

TODO

## Design Decisions

### Literal handling

It is appealing to support the conversion of R literals (`1L`, `1.0`
`TRUE`, etc.) to `AnvilTensor`s when calling into `jit`-ted functions,
i.e. allow the following:

``` r
jit(nv_add, static = character())(1, 2)
```

In `jit`(), we could in principle do this, because we know which
arguments are static and which are expected to be `AnvilTensor`s.
However, in
[`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md), we
don’t know which arguments are static and which are not. For example, we
can’t make the following work:

``` r
jit(\() {
  gradient(nv_add)(1, 2)
})
```

This would be somewhat inconsistent and hard to to reason about.
Furthermore, auto-converting literals passed as non-static arguments to
`jit`-ted functions would also entail various suble differences between
debug-mode and jit-mode, as the former has no top-level hook for this
conversion. Finally, requiring the user to think about the input data
types should also be advantageous; we want to prioritize clarity over
minor convenience. Note that for primitive calls like `nv_tensor(1)^2`
we do auto-convert literals, because we know which arguments are
expected to be `AnvilTensor`s and otherwise code just becomes much
harder to read.
