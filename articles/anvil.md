# Get Started

In this vignette, you will learn everything you need to know to get
started implementing numerical algorithms using {anvil}. If you have
experience with JAX in Python, you should feel right at home.

## The `AnvilTensor`

We will start by introducing the main data structure, which is the
`AnvilTensor`. It is essentially like an `R` array, with some
differences:

1.  It supports more data types, such as different precisions, as well
    as unsigned integers.
2.  The tensor can live on different *platforms*, such as CPU or GPU.
3.  0-dimensional tensors can be used to represent scalars.

We can create such an object from R data types using the `nv_tensor` and
friends functions. Below, we create a 0-dimensional tensor (i.e., a
scalar) of type `int16` on the CPU.

``` r
library(anvil)
set.seed(42)
nv_tensor(1L, dtype = "i16", device = "cpu", shape = integer())
```

    ## AnvilTensor 
    ##  1
    ## [ CPUi16{} ]

Note that for creation of scalars, you can also use `nv_scalar` as a
shorthand to skip specifying the shape.

``` r
nv_scalar(1L, dtype = "i16", device = "cpu")
```

    ## AnvilTensor 
    ##  1
    ## [ CPUi16{} ]

We can also create higher-dimensional tensors, for example a 2x3 CPU
tensor of type `f32`. Below, we omit specifying the platform and
datatype, as it will default to `"cpu"` and `"f32"`. Note that the
default datatype depends on the input datatype.

``` r
x <- array(1:6, dim = c(2, 3))
nv_tensor(x)
```

    ## AnvilTensor 
    ##  1 3 5
    ##  2 4 6
    ## [ CPUi32{2x3} ]

``` r
y <- nv_tensor(x, dtype = "f32")
y
```

    ## AnvilTensor 
    ##  1.0000 3.0000 5.0000
    ##  2.0000 4.0000 6.0000
    ## [ CPUf32{2x3} ]

In order to convert an `AnvilTensor` back to a regular R array, you can
use the
[`as_array()`](https://r-xla.github.io/tengen/reference/as_array.html)
function.

``` r
as_array(y)
```

    ##      [,1] [,2] [,3]
    ## [1,]    1    3    5
    ## [2,]    2    4    6

At first, working with `AnvilTensor`s may feel a bit cumbersome, because
you cannot directly apply functions to them like you would with regular
R arrays.

``` r
x + x
```

    ##      [,1] [,2] [,3]
    ## [1,]    2    6   10
    ## [2,]    4    8   12

``` r
y + y
```

    ## Error in y + y: non-numeric argument to binary operator

## JIT Compilation

In order to work with `AnvilTensor`s, you need to convert the function
you want to apply to a jit-compiled version via
[`anvil::jit()`](../reference/jit.md).

``` r
plus_jit <- jit(`+`)
plus_jit(y, y)
```

    ## AnvilTensor 
    ##   2.0000  6.0000 10.0000
    ##   4.0000  8.0000 12.0000
    ## [ CPUf32{2x3} ]

The result of the operation is again an `AnvilTensor`.

We can, of course, jit-compile more complex functions as well.

Below, we define a function that takes in a data matrix `X`, a weight
vector `beta` and a scalar bias `b`, and computes the linear model
output \\y = X \times \beta + \alpha\\.

``` r
linear_model_r <- function(X, beta, alpha) {
  X %*% beta + alpha
}

linear_model <- jit(linear_model_r)

X <- nv_tensor(rnorm(6), dtype = "f32", shape = c(2, 3))
beta <- nv_tensor(rnorm(3), dtype = "f32", shape = c(3, 1))
alpha <- nv_scalar(rnorm(1), dtype = "f32")

linear_model(X, beta, alpha)
```

    ## AnvilTensor 
    ##   2.7911
    ##  -1.1904
    ## [ CPUf32{2x1} ]

One current restriction of {anvil} is that the function has to be
re-compiled for every unique combination of inputs shapes, data-types,
and platforms. To show this, we create a slightly modified version of
`linear_model`.

``` r
linear_model2 <- jit(function(X, beta, alpha) {
  cat("compiling ...\n")
  X %*% beta + alpha
})
```

Next, we create a little helper function that creates example input data
with different numbers of observations:

``` r
simul_data <- function(n, p) {
  list(
    X = nv_tensor(rnorm(n * p), dtype = "f32", shape = c(n, p)),
    beta = nv_tensor(rnorm(p), dtype = "f32", shape = c(p, 1)),
    alpha = nv_scalar(rnorm(1), dtype = "f32")
  )
}
```

Below, we call the function twice on data with the same shapes.

``` r
do.call(linear_model2, simul_data(2, 3))
```

    ## compiling ...

    ## AnvilTensor 
    ##   4.9640
    ##  -0.1413
    ## [ CPUf32{2x1} ]

``` r
do.call(linear_model2, simul_data(2, 3))
```

    ## AnvilTensor 
    ##   0.6140
    ##  -2.5214
    ## [ CPUf32{2x1} ]

We can notice that we only see the `"compiling ..."` message the first
and not the second time. This is, because the first time, the function
is compiled into an `XLA` executable and cached for later reuse. When we
call into `linear_model2` the second time, we don’t execute the R
function at all, but directly run the cached `XLA` executable. This
executable does not contain “standard” R code, like the
[`cat()`](https://rdrr.io/r/base/cat.html) call, but only operations
applied to `AnvilTensor`s.

If we now call the function on data with different shapes, we see that
the function is re-compiled.

``` r
y_hat <- do.call(linear_model2, simul_data(4, 3))
```

    ## compiling ...

Because the compilation step itself can take some time, {anvil}
therefore gives the best results when the same function is called many
times with the same input shapes, data types, and platforms, or the
computation itself is sufficiently large to amortize the compilation
overhead. One common application scenario where this assumption holds
are iterative optimization algorithms.

### Static Arguments

One feature of {anvil} is that not all arguments of `jit`-compiled
functions need to be `AnvilTensor`s. For example, we might want a linear
model with or without an intercept term. To do so, we add the logical
argument `with_bias` to our function. We need to mark this argument as
`static`, so {anvil} knows to treat this as a regular R value instead of
an `AnvilTensor`.

``` r
linear_model3 <- jit(function(X, beta, alpha = NULL, with_bias) {
  if (with_bias) {
    cat("Compiling without bias ...\n")
    X %*% beta + alpha
  } else {
    cat("Compiling with bias ...\n")
    X %*% beta
  }
}, static = "with_bias")
```

We can call this function now with or without a bias term:

``` r
linear_model3(X, beta, with_bias =  FALSE)
```

    ## Compiling with bias ...

    ## AnvilTensor 
    ##   2.8538
    ##  -1.1277
    ## [ CPUf32{2x1} ]

``` r
linear_model3(X, beta, alpha, with_bias =  TRUE)
```

    ## Compiling without bias ...

    ## AnvilTensor 
    ##   2.7911
    ##  -1.1904
    ## [ CPUf32{2x1} ]

Static arguments work differently than `AnvilTensors` as the function
will not be re-compiled for each new observed value of the static
argument.

### Nested Inputs and Outputs

Note also, that the inputs, as well as the outputs, can also contain
nested data structures that contain `AnvilTensor`s, although we
currently only support (named) lists.

``` r
linear_model4 <- jit(function(inputs) {
  list(y_hat = inputs[[1]] %*% inputs[[2]] + inputs[[3]])
})
linear_model4(list(X, beta, alpha))
```

    ## $y_hat
    ## AnvilTensor 
    ##   2.7911
    ##  -1.1904
    ## [ CPUf32{2x1} ]

So far, we have only implemented the prediction step for a linear model.
One of the core applications of anvil is to implement learning
algorithms, for which we often need gradients, as well as control flow.
We will start with gradients.

## Automatic Differentiation

In anvil, you can easily obtain the gradient function of a scalar-valued
function using [`gradient()`](../reference/gradient.md): Currently,
vector-valued functions cannot be differentiated. Below, we implement
implement the loss function for our linear model.

``` r
mse <- function(y_hat, y) {
  mean((y_hat - y)^nv_scalar(2.0))
}
```

We now need some target variables `y`, so we simulate some data from a
linear model:

``` r
beta <- rnorm(1)
X <- matrix(rnorm(100), ncol = 1)
alpha <- rnorm(1)
y <- X %*% beta + alpha + rnorm(100, sd = 0.5)
plot(X, y)
```

![](anvil_files/figure-html/unnamed-chunk-16-1.png)

``` r
X <- nv_tensor(X)
y <- nv_tensor(y)
```

Next, we randomly initialize the model parameters:

``` r
beta_hat <- nv_tensor(rnorm(1), shape = c(1, 1), dtype = "f32")
alpha_hat <- nv_scalar(rnorm(1), dtype = "f32")
```

We can now define a function that does the prediction and calculates the
loss. Note that we are calling into the original R function that does
the prediction and not its jit-compiled version.

``` r
model_loss <- function(X, beta, alpha, y) {
  y_hat <- linear_model_r(X, beta, alpha)
  mse(y_hat, y)
}
```

Using the [`gradient()`](../reference/gradient.md) transformation, we
can automatically obtain the gradient function of `model_loss` with
respect to some of its arguments, which we specify.

``` r
model_loss_grad <- gradient(
  model_loss,
  wrt = c("beta", "alpha")
)
```

Finally, we define the update step for the weights using gradient
descent.

``` r
update_weights <- jit(function(X, beta, alpha, y, lr) {
  lr <- nv_scalar(0.1)
  grads <- model_loss_grad(X, beta, alpha, y)
  beta_new <- beta - lr * grads$beta
  alpha_new <- alpha - lr * grads$alpha
  list(beta = beta_new, alpha = alpha_new)
})
```

This already allows us to train our linear model using gradient descent:

``` r
weights <- list(beta = beta_hat, alpha = alpha_hat)
for (i in 1:100) {
  weights <- update_weights(X, weights$beta, weights$alpha, y)
}
```

![](anvil_files/figure-html/unnamed-chunk-22-1.png)

While this might seem like a reasonable solution, it continuously
switches between the R interpreter and the XLA runtime. Moreover, we
allocate new tensors in each iteration for the weights. While the latter
might not be a big problem for small models, it can lead to significant
overhead when working with bigger tensors. Next, we will briefly address
the concept of immutability in anvil and which options you have have to
work around it.

## Immutability

Conceptually, whenever we are defining programs in anvil, we are
strictly following **value semantics**. This means, in-place
modifications like updating an array element are conceptually
impossible.

When we are dealing with updating an existing tensor, this might either
be: 1. Updating an `AnvilTensor` that “lives within” a jit-compiled
function. 2. Updating an `AnvilTensor` living in R through a
jit-compiled function.

For the first category, there is function TODO demonstrated below.

``` r
# TODO
```

The thing to note is that while conceptually, this is *not* an in-place
update, but creates a new tensor, the XLA compiler is able to optimize
this, ensuring that no unnecessary copies are actually made.

For the second category, we can mark arguments of a jit-compiled
function as “donatable”. This means, we are telling the XLA runtime that
after we pass those tensors marked as donatable into the function, we
will no longer use them in R. The XLA compiler will therefore be able to
reuse the memory.

``` r
# TODO
```

## Control Flow

Earlier, we have already used R control flow to train our linear model.

In principle, there are three ways to handle control-flow in anvil:

1.  Embed jit-compiled functions inside R control-flow constructs, which
    we have seen earlier.

2.  Embed R control flow inside a jit-compiled function:

    - We have seen this earlier for conditionals, where – depending on
      the value of the static argument – only one of the branches was
      part of the XLA executable.
    - Loops get unrolled.

3.  Use special control-flow primitives provided by anvil, such as
    `nv_while()` and `nv_if()` (to be implemented).

What’s the best solution depends on the specific scenario.

One thing to be aware of is that we usually don’t want R loops within
the jit-compiled function. This is, because the loop will be unrolled
during compilation, which can lead to very large compilation times and
big executables.

TODO
