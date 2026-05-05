# FAQ

## Why is my function slow?

There can be many reasons why an {anvl} program is not as fast as one
might expect. See the
[Efficiency](https://r-xla.github.io/anvl/dev/articles/efficiency.md)
vignette which explains various pitfals and levers for optimizing
program runtime.

## Why does timing my function show suspiciously fast results?

Many operations {anvl} happen asynchronously under the hood (c.f.
[Asynchronous
execution](https://r-xla.github.io/anvl/dev/articles/efficiency.html#asynchronous-execution)).
In order to properly time an {anvl} program, you therefore need to
ensure that:

1.  All creation of buffers used in the benchmark function have
    finisehd.
2.  The results of the computation are awaited.

``` r

library(anvl)

mul_n <- function(x, n) {
  for (i in 1:n) {
    x <- x * x
  }
  return(x)
}

# returns immediately
x <- nv_array(rnorm(1e8))

# Ensure buffer creation is finished
await(x)

# Bad (only measures kernel launch):
system.time(mul_n(x, 20))
```

    ##    user  system elapsed 
    ##   0.284   0.133   0.191

``` r

# Good (also measures actual computation):
system.time(await(mul_n(x, 20)))
```

    ##    user  system elapsed 
    ##   0.857   0.937   0.891

## How do I control the number of threads used by XLA?

XLA (the compiler backend used by `anvl`) may use multiple CPU threads
for parallelism. On shared systems such as HPC clusters, it is often
necessary to restrict which cores a process can use.

The recommended approach is to control thread affinity **from outside
the process**, using OS-level tools. On Linux, `taskset` is the most
common option:

``` bash
# Pin the R process to cores 0-3
taskset -c 0-3 Rscript my_script.R
```

See [jax#15866](https://github.com/jax-ml/jax/issues/15866) for more
information.
