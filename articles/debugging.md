# Debugging

The main benefit of {anvil} is that it allows you to compile R code into
an executable, which considerably increases execution speed. The main
drawback is that this makes it harder to debug code, because you can’t
rely on the R debugger to step through the function during execution
(only during tracing/compilation).

For this reason, {anvil} provides a **debug mode** that allows you to
perform abstract evaluation, which we will explain first. Afterwards, we
cover different ways to print values during execution.

## Debug Mode

When {anvil} functions are executed without being wrapped in
[`jit()`](../reference/jit.md), they will run in debug mode and output a
`DebugBox` object, which essentially represents the type of the output
tensor.

``` r
library(anvil)
y <- nv_scalar(1) + nv_tensor(1:4, shape = c(2, 2))
y
```

    ## f32{2,2}

``` r
mean(y)
```

    ## f32{}

To use debug mode, you can pass `AnvilTensor` and literals (`1L`,
`1.0`).

``` r
1 + nv_scalar(1)
```

    ## f32{}

If you only want to specify the abstract types, you can also directly
pass `DebugBox` objects:

``` r
debug_box("f32", c(2, 3)) %*% debug_box("f32", c(3, 1))
```

    ## f32{2,1}

You can even evaluate transformations like gradients in debug mode:

``` r
gradient(mean)(debug_box("f32", c(2, 2)))
```

    ## $x
    ## f32{2,2}

Because this all happens in the R interpreter, you can also add
breakpoints to the code and step through it to identify bugs. However,
even if a program is valid and can be compiled, it might not work as
expected, e.g. because of logical bugs or invalid hyperparameters. For
this, it’s important to monitor (intermediate) values, which we will
cover in the next section.

## Printing Values

There are different ways to print values in {anvil} that might be
confusing at first. We will start with the naive way of simply inserting
[`print()`](https://rdrr.io/r/base/print.html) statements into the code
of a `jit`-compiled function.

``` r
f_jit <- jit(\(x) {
  y <- x^2 + x^2
  print(y)
  mean(y)
})
```

If we run this function, we see that not the actual value is printed,
but some `GraphBox` object. This `GraphBox` object is passed around
**during tracing** so that we can convert the function into a `Graph`
that is subsequently compiled.

``` r
f_jit(nv_tensor(1:4, shape = c(2, 2)))
```

    ## GraphBox(GraphValue(AbstractTensor(dtype=f32?, shape=2x2)))

    ## AnvilTensor 
    ##  15.0000
    ## [ CPUf32{} ]

Furthermore, if we call the function with identical input types, it
won’t be printed because the executable is retrieved from the cache.

``` r
f_jit(nv_tensor(0:3, shape = c(2, 2)))
```

    ## AnvilTensor 
    ##  7.0000
    ## [ CPUf32{} ]

If you want to get the actual content of the values during execution,
there are two options:

1.  Ensure that the value to print is returned by the jit-compiled
    function so it can be printed **after** execution. This comes
    naturally when the `jit`-compiled function is called within an R
    loop.
2.  Using the special [`nv_print()`](../reference/nv_print.md) function
    to print **during** execution. This is useful when the value to
    print is not naturally returned by the function.

For illustrative purposes, we will count to 10 and print all
intermediate results.

In the first approach, we only `jit`-compile the update function and
iteratively call it in an R loop.

``` r
add_one <- jit(\(x) x + 1L)
```

``` r
x <- nv_scalar(0L)
for (i in 1:10) {
  x <- add_one(x)
  print(x)
}
```

    ## AnvilTensor 
    ##  1
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  2
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  3
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  4
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  5
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  6
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  7
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  8
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  9
    ## [ CPUi32{} ] 
    ## AnvilTensor 
    ##  10
    ## [ CPUi32{} ]

For the second approach, we use `nv_while` to implement the loop.

``` r
jit(\() {
  init <- nv_fill(1L, shape = c())
  nv_while(
    list(x = init),
    \(x) x <= 10,
    \(x) {
      nv_print(x)
      list(x = x + 1L)
    }
  )
}, device = "cpu")()
```

    ## AnvilTensor
    ##  1
    ## [ S32{} ]
    ## AnvilTensor
    ##  2
    ## [ S32{} ]
    ## AnvilTensor
    ##  3
    ## [ S32{} ]
    ## AnvilTensor
    ##  4
    ## [ S32{} ]
    ## AnvilTensor
    ##  5
    ## [ S32{} ]
    ## AnvilTensor
    ##  6
    ## [ S32{} ]
    ## AnvilTensor
    ##  7
    ## [ S32{} ]
    ## AnvilTensor
    ##  8
    ## [ S32{} ]
    ## AnvilTensor
    ##  9
    ## [ S32{} ]
    ## AnvilTensor
    ##  10
    ## [ S32{} ]

    ## $x
    ## AnvilTensor 
    ##  11
    ## [ CPUi32{} ]

We will provide more formatting options in the future!
