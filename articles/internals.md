# Internals

## Transformations under the Hood

Just like a real anvil, the {anvil} package is a tool that allows you to
reshape objects. While the former is used to shape metal, our {anvil}
package can be used to transform code into other code.

In general, there are three types of transformations:

1.  `R` -\> `Graph`: Generic `R` functions are way too complicated to
    handle, so the first step in {anvil} is always to convert them into
    a computational [`anvil::Graph`](../reference/Graph.md) object via
    **tracing**. Such a `Graph` is similar to `JAXExpr` objects in
    `JAX`.
2.  `Graph` -\> `Graph`: It is possible to transform `Graph`s into other
    `Graph`s. Currently, there is only one such transformation, which is
    the [`gradient()`](../reference/gradient.md) transformation.
3.  `Graph` -\> `Executable`: In order to perform the actual
    computation, the `Graph` needs to be converted into an executable.
    Currently, we only support the XLA backend (via `stablehlo` and
    `pjrt`), but it is in principle possible to support other backends
    as well.

### Tracing R functions into Graphs

All functionality in the {anvil} package is centered around the `Graph`
class, and such `Graph`s are created by tracing R functions. In general,
when we want to convert some code into another form, there are two
approaches:

1.  Static analysis, which would require operating on the abstract
    syntax tree (AST) of the code.
2.  Dynamic analysis (aka “tracing”), which executes the code and
    records (selected) operations.

The former approach is followed by the {quickr} package, which allows
you to transpile R code into Fortran. In {anvil}, we are following the
dynamic approach, which we will now illustrate. Our goal will be to
trace the following function, which either adds or subtracts two
tensors. The first two arguments are expected to be `AnvilTensor`s,
while the third is either `"add"` or `"sub"`.

``` r
library(anvil)
f <- function(x, y, op) {
  if (op == "add") {
    nv_add(x, y)
  } else if (op == "sub") {
    nv_sub(x, y)
  } else {
    stop("Unsupported operation")
  }
}
```

To do this, we use [`anvil::trace_fn()`](../reference/trace_fn.md),
which takes in an `R` function and a list of example arguments that
specify the types of the inputs.

``` r
x <- nv_scalar(1, "f32")
y <- nv_scalar(5, "f32")
graph <- trace_fn(f, list(x = x, y = y, op = "add"))
graph
```

    ## <Graph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Body:
    ##     %1: f32[] = add(%x1, %x2)
    ##   Outputs:
    ##     %1: f32[]

The output of [`trace_fn()`](../reference/trace_fn.md) is now a `Graph`
object that represents the computation. The fields of the `Graph` are:

- `inputs`, which are `GraphValue`s that represent the inputs to the
  function.
- `outputs`, which are `GraphValue`s that represent the outputs of the
  function.
- `calls`, which are `PrimitiveCall`s that take in `GraphValue`s (and
  parameters) and produce output `GraphValue`s.
- `in_tree`, `out_tree`, which we will cover later (do we??)

What happens internally in [`trace_fn()`](../reference/trace_fn.md) is
that a new `GraphDescriptor` is created and the inputs `x` and `y` are
converted into `anvil::GraphBox` objects. Then, the function `f` is
simply evaluated with the `GraphBox` objects as inputs. During this
evaluation, we need to distinguish between two cases:

1.  A “standard” `R` function is called: Here, nothing special happens
    and the function is simply evaluated.
2.  An `anvil` function is called: Here, the operation that underlies
    the function is recorded in the `GraphDescriptor`.

The evaluation of the `if` statement is an example for the first
category. Because we set `op = "add"`, only the first branch is
executed. Then, we are calling `nv_add`, which attaches a
`PrimitiveCall` that represents the addition of the two tensors to the
`@calls` of the `GraphDescriptor`.

A `PrimitiveCall` object consists of the following fields:

- `primitive`: The primitive function that was called.
- `inputs`: The inputs to the primitive function.
- `params`: The parameters (non-tensors) to the primitive function.
- `outputs`: The outputs of the primitive function.

When the evaluation of `f` is complete, the `@outputs` field of the
`GraphDescriptor` is set and the `Graph` is subsequently created from
the `GraphDescriptor`. The only difference between the `Graph` and the
`GraphDescriptor` is that the latter has some utility fields that are
useful during graph creation, but for the purposes of this tutorial, you
can think of them as being the same.

### Transforming Graphs into other Graphs

Once we have staged out our `R` function into a simpler format, we can
do all sorts of things with it, e.g., compute its gradient. The {anvil}
package does not in any way dictate how such a `Graph` to `Graph`
transformation can be implemented. For most interesting transformations,
however, we need to store some information for each {anvil} primitive
function. In the case of the gradient, we need to store the derivative
rules. For this, [`anvil::Primitive`](../reference/Primitive.md) objects
have a `@rules` field that can be populated. The derivative rules are
stored as functions under the `"backward"` name.

``` r
anvil:::p_add@rules[["backward"]]
```

    ## function (inputs, outputs, grads, .required) 
    ## {
    ##     grad <- grads[[1L]]
    ##     list(if (.required[[1L]]) grad, if (.required[[2L]]) grad)
    ## }
    ## <bytecode: 0x55a4cb9920b0>
    ## <environment: namespace:anvil>

The `anvil:::transform_gradient` function uses these rules to compute
the gradient of a function. For this specific transformation, we are
walking the graph backwards and apply the derivative rules, which will
add the “backward pass” to the graph. Besides the forward graph, the
transformation takes in the `wrt` argument, which specifies with respect
to which arguments to compute the gradient.

``` r
bwd_graph <- anvil:::transform_gradient(graph, wrt = c("x", "y"))
bwd_graph
```

    ## <Graph>
    ##   Inputs:
    ##     %x1: f32[]
    ##     %x2: f32[]
    ##   Constants:
    ##     %c1: f32[]
    ##   Body:
    ##     %1: f32[] = add(%x1, %x2)
    ##   Outputs:
    ##     %c1: f32[]
    ##     %c1: f32[]

### Lowering a Graph

In order to execute a `Graph`, we need to convert it into a – wait for
it – executable. Currently, there is only one way to do this, namely to
convert the `Graph` into a
[`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)
object and to then compile it via
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).

Like for the gradient transformation, the rules of how to do this
transformation are stored in the `@rules` fields of the primitives.

``` r
anvil:::p_add@rules[["stablehlo"]]
```

    ## function (lhs, rhs) 
    ## {
    ##     list(stablehlo::hlo_add(lhs, rhs))
    ## }
    ## <bytecode: 0x55a4cb991898>
    ## <environment: namespace:anvil>

They are applied in the [`anvil::stablehlo`](../reference/stablehlo.md)
function.

``` r
func <- stablehlo(graph)[[1L]]
func
```

    ## func.func @main (%0: tensor<f32>, %1: tensor<f32>) -> tensor<f32> {
    ## %2 = "stablehlo.add" (%0, %1): (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    ## "func.return"(%2): (tensor<f32>) -> ()
    ## }

Now, we can compile the function via `pjrt_compile()`.

``` r
hlo_str <- stablehlo::repr(func)
program <- pjrt::pjrt_program(src = hlo_str, format = "mlir")
exec <- pjrt::pjrt_compile(program)
```

To run the function, we simply pass the tensors to the executable.

``` r
out <- pjrt::pjrt_execute(exec, x, y)
out
```

    ## PJRTBuffer 
    ##  6.0000
    ## [ CPUf32{} ]

## The user interface

### jit

In the previous section, we have shown how the transformations are
implemented under the hood. The actual user interface is a little more
user-friendly and follows the `JAX` interface. We will start by
explaining [`jit()`](../reference/jit.md), where we mark the `op`
argument as `static`, i.e., it is not an `AnvilTensor`.

``` r
f_jit <-  jit(f, static = "op")
f_jit(x, y, "add")
```

    ## AnvilTensor 
    ##  6.0000
    ## [ CPUf32{} ]

It might be intuitive to think that [`jit()`](../reference/jit.md) first
calls [`trace_fn()`](../reference/trace_fn.md), then runs
[`stablehlo()`](../reference/stablehlo.md), followed by
`pjrt_compile()`. This is, however, not what is happening, because for
all of this, we need example inputs to the function. The function
`f_jit` should instead be seen as a recipe to do all of this Just In
Time (JIT).

However, if we were to apply these steps every time the `f_jit` function
is called, this would be very inefficient, because tracing and compiling
takes some time. Therefore, the function `f_jit` also contains a cache
(implemented as an
[`xlamisc::LRUCache`](https://rdrr.io/pkg/xlamisc/man/LRUCache.html)),
which will check whether there is already an executable for the given
inputs. For this, the types of all `AnvilTensor`s need to match exactly
(data type and shape) and all static arguments need to be identical. For
example, if we run the function with `AnvilTensor`s of the same type,
but different values, the function won’t be recompiled, which we can see
by checking the size of the cache:

``` r
cache_size <- function(f) environment(f)$cache$size
cache_size(f_jit)
```

    ## [1] 1

``` r
f_jit(nv_scalar(-99, "f32"), nv_scalar(2, "f32"), "add")
```

    ## AnvilTensor 
    ##  -97.0000
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
f_jit(nv_scalar(1, "f32"), nv_scalar(2, "f32"), "sub")
```

    ## AnvilTensor 
    ##  -1.0000
    ## [ CPUf32{} ]

``` r
cache_size(f_jit)
```

    ## [1] 3

### gradient

Just like [`jit()`](../reference/jit.md),
[`gradient()`](../reference/gradient.md) also returns a function that
will lazily create the graph and transform it, once the inputs are
provided.

``` r
g <- gradient(f, wrt = c("x", "y"))
```

However, the output `g()` cannot be called on `AnvilTensor`s, because it
is just a graph building function.

In order to execute it, we need to wrap it in a
[`jit()`](../reference/jit.md) call:

``` r
g_jit <- jit(g, static = "op")
g_jit(x, y, "add")
```

    ## $x
    ## AnvilTensor 
    ##  1.0000
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvilTensor 
    ##  1.0000
    ## [ CPUf32{} ]

Moreover, we can also use `g()` in another function:

``` r
h <- function(x, y) {
  z <- nv_add(x, y)
  g(z, z, "add")
}
h_jit <- jit(h)
h_jit(x, y)
```

    ## $x
    ## AnvilTensor 
    ##  2.0000
    ## [ CPUf32{} ] 
    ## 
    ## $y
    ## AnvilTensor 
    ##  2.0000
    ## [ CPUf32{} ]

So, what is happening here? Once the inputs `x` and `y` are provided to
`h_jit`, a new `GraphDescriptor` is created and the inputs `x` and `y`
are converted into `GraphBox` objects. Then, we record the addition of
`x` and `y` in the `GraphDescriptor`. When we are then calling into `g`,
a second `GraphDescriptor` is created, where we will first build up the
`Graph` representing the `g` function. After we are done doing this, we
transform this graph into its gradient graph. Once this is done, the
sub-graph is inlined into the parent graph (that so far only holds the
addition of `x` and `y`). We can look at this graph below:

``` r
h_graph <- trace_fn(h, list(x = x, y = y))
```

After that, this graph is lowered to stablehlo and subsequently
compiled.

## More Internals

### Constant Handling

### Nested Inputs and Outputs
