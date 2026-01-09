# Random Number Generation

In this vignette, you will learn how to generate random numbers in
{anvil}. If you’re familiar with R’s built-in random number generation,
you’ll notice that {anvil} handles things a bit differently.

## The RNG State

In base R, random number generation uses a global state (`.Random.seed`)
that is automatically updated after each call:

``` r
set.seed(42)
.Random.seed[2:4]
#> [1]        624  507561766 1260545903
rnorm(3)
#> [1]  1.3709584 -0.5646982  0.3631284
.Random.seed[2:4]
#> [1]           6 -1577024373  1699409082
rnorm(3)
#> [1]  0.6328626  0.4042683 -0.1061245
.Random.seed[2:4]
#> [1]          12 -1577024373  1699409082
```

In {anvil}, we take a different approach: the random number generator
state must be explicitly passed around. This is because {anvil} follows
a functional programming paradigm where functions are pure and don’t
have side effects.

**Note:** This explicit state-passing behavior might change in the
future to provide a more R-like experience, but for now you need to
manage the state yourself.

## Creating an Initial State

To generate random numbers, you first need to create an initial RNG
state, which is simply a `ui64[2]`. For convenience, you can convert an
R seed into a state using
[`nv_rng_state()`](../reference/nv_rng_state.md):

``` r
library(anvil)
state <- nv_rng_state(seed = 42L)
state
#> AnvilTensor 
#>  42
#>   0
#> [ CPUui64{2} ]
```

## Generating Random Numbers

The main functions for generating random numbers are
[`nv_runif()`](../reference/nv_runif.md),
[`nv_rdunif()`](../reference/nv_rdunif.md),
[`nv_rbinom()`](../reference/nv_rbinom.md), and
[`nv_rnorm()`](../reference/nv_rnorm.md). Both functions return a list
with two elements:

1.  The **new** RNG state (to be used for subsequent random number
    generation)
2.  The generated random numbers

Let’s generate some uniform random numbers:

``` r
f <- jit(function(state) {
  nv_runif(state, dtype = "f32", shape = c(2, 3))
})

result <- f(state)
result[[1]]  # new state
#> AnvilTensor 
#>  42
#>   3
#> [ CPUui64{2} ]
result[[2]]  # random numbers
#> AnvilTensor 
#>  0.8690 0.1506 0.5203
#>  0.3103 0.9928 0.1065
#> [ CPUf32{2x3} ]
```

For normally distributed random numbers:

``` r
g <- jit(function(state) {
  nv_rnorm(state, dtype = "f32", shape = c(2, 3), mu = 0, sigma = 1)
})

result <- g(state)
result[[2]]
#> AnvilTensor 
#>  -0.0675  0.9489  1.9457
#>  -0.5255  1.2002  0.0008
#> [ CPUf32{2x3} ]
```

## What Happens When You Reuse the State?

Here’s the key insight: if you use the same state twice, you get the
same random numbers.

``` r
h <- jit(function(state) {
  result1 <- nv_runif(state, dtype = "f32", shape = 3L)
  result2 <- nv_runif(state, dtype = "f32", shape = 3L)
  list(first = result1[[2]], second = result2[[2]])
})

output <- h(state)
as_array(output$first)
#> [1] 0.8690484 0.3102535 0.1506324
as_array(output$second)
#> [1] 0.8690484 0.3102535 0.1506324
```

As you can see, both calls produced identical random numbers because we
used the same state for both.

## Properly Chaining Random Number Generation

To get different random numbers in subsequent calls, you need to pass
the **new** state returned by the previous call:

``` r
proper_rng <- jit(function(state) {
  result1 <- nv_runif(state, dtype = "f32", shape = c(3))
  new_state <- result1[[1]]
  result2 <- nv_runif(new_state, dtype = "f32", shape = c(3))
  list(first = result1[[2]], second = result2[[2]])
})

output <- proper_rng(state)
as_array(output$first)
#> [1] 0.8690484 0.3102535 0.1506324
as_array(output$second)
#> [1] 0.5203207 0.1064724 0.2499373
```

Now we get different random numbers because we properly propagated the
state.
