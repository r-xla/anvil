# Subsetting

In this vignette, you will learn how to subset tensors in {anvil} and
how to update subsets. Because tensor shapes in {anvil} programs are
static, only certain subsetting operations are supported and they come
with some surprises.

We start by listing possible subsets and whether they support dynamic
values (tensors that are specified during runtime) or only static values
(e.g., R literals).

| Subset           | Dynamic | Static |
|------------------|---------|--------|
| Single Index     | Yes     | Yes    |
| Multiple Indices | Yes     | Yes    |
| Range            | No      | Yes    |
| Mask             | No      | No     |

Ranges cannot have dynamic values, because then the size of the subset
would be unknown (what’s the size of `a:b` where `a` and `b` are
unknown?). Boolean masks are not supported, because the output shape
depends on the data, which is not known at compile time. If you want to
modify tensors based on a mask, see
[`nv_ifelse()`](https://r-xla.github.io/anvil/reference/nv_ifelse.md).
Negative indexing (e.g., `x[-1]` to exclude elements) is currently also
not supported. For static values, this will throw an error, for dynamic
values, it will be clamped to the valid range. If you are missing a
feature, please open an issue on GitHub.

We will start with subsetting and then move on to subset-assignment.

## Subsetting

### Subsetting 1D tensors

Let’s start with some simple examples of selecting individual elements
from a 1-dimension tensor. The index can be either static or dynamic and
we can drop or keep the dimension:

``` r
library(anvil)
x <- nv_tensor(1:10)
x
```

    ## AnvilTensor
    ##   1
    ##   2
    ##   3
    ##   4
    ##   5
    ##   6
    ##   7
    ##   8
    ##   9
    ##  10
    ## [ CPUi32{10} ]

- Static & Drop:

  ``` r
  jit_eval({
    x[2]
  })
  ```

      ## AnvilTensor
      ##  2
      ## [ CPUi32{} ]

- Static & Keep:

  ``` r
  jit_eval({
    x[list(2)]
  })
  ```

      ## AnvilTensor
      ##  2
      ## [ CPUi32{1} ]

- Dynamic & Drop:

  ``` r
  jit_eval({
    x[nv_scalar(2L)]
  })
  ```

      ## AnvilTensor
      ##  2
      ## [ CPUi32{} ]

- Dynamic & Keep:

  Below, we almost perform the same operation as above, only do we use a
  tensor of shape `(1)` instead of a scalar with shape `()`. The
  difference is that subsetting with the former will preserve the
  dimension, while the latter will drop it, as we have seen above. This
  ensures that the dimensionality of the result is the same for any 1D
  subset specification, and not suddenly “simplify” the result to 0D.

  ``` r
  jit_eval({
    x[nv_tensor(2L)]
  })
  ```

      ## AnvilTensor
      ##  2
      ## [ CPUi32{1} ]

Next, we subset multiple elements, where we only have to distinguish
between static and dynamic indices.

- Static

  ``` r
  jit_eval({
    x[list(2, 4, 6)]
  })
  ```

      ## AnvilTensor
      ##  2
      ##  4
      ##  6
      ## [ CPUi32{3} ]

- Dynamic

  ``` r
  jit_eval({
    x[nv_tensor(c(2L, 4L, 6L))]
  })
  ```

      ## AnvilTensor
      ##  2
      ##  4
      ##  6
      ## [ CPUi32{3} ]

We are using [`list()`](https://rdrr.io/r/base/list.html) instead of
1-dimension vectors, because otherwise the case where we use a length-1
vector would be ambiguous (do we drop or keep the dimension?). This
allows us to do without a `drop` parameter.

We can also use a range that can be specified either canonically via
`a:b` or using
[`nv_seq()`](https://r-xla.github.io/anvil/reference/nv_seq.md).

``` r
jit_eval({
  x[2:5]
})
```

    ## AnvilTensor
    ##  2
    ##  3
    ##  4
    ##  5
    ## [ CPUi32{4} ]

``` r
jit_eval({
  x[nv_seq(2, 5)]
})
```

    ## AnvilTensor
    ##  2
    ##  3
    ##  4
    ##  5
    ## [ CPUi32{4} ]

Note that the `a:b` syntax works via Non-Standard-Evaluation (NSE), so
we can distinguish it from the actual vector `2:5`. Internally, it is
translated to `nv_seq(a, b)`.

It is also possible to select the whole range by omitting the
specification altogether.

``` r
jit_eval({
  x[]
})
```

    ## AnvilTensor
    ##   1
    ##   2
    ##   3
    ##   4
    ##   5
    ##   6
    ##   7
    ##   8
    ##   9
    ##  10
    ## [ CPUi32{10} ]

### Subsetting higher-dimensional tensors

We start by creating a 2-dimensional tensor.

``` r
x <- nv_tensor(matrix(1:12, nrow = 3, byrow = TRUE))
x
```

    ## AnvilTensor
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

Combining subsets just works like one would expect.

``` r
jit_eval({
  x[1, ]
})
```

    ## AnvilTensor
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPUi32{4} ]

``` r
jit_eval({
  x[1, 2]
})
```

    ## AnvilTensor
    ##  2
    ## [ CPUi32{} ]

``` r
jit_eval({
  x[list(1), 2:3]
})
```

    ## AnvilTensor
    ##  2 3
    ## [ CPUi32{1,2} ]

``` r
jit_eval({
  x[list(1, 3), 2:3]
})
```

    ## AnvilTensor
    ##   2  3
    ##  10 11
    ## [ CPUi32{2,2} ]

``` r
jit_eval({
  x[1:2, 2:3]
})
```

    ## AnvilTensor
    ##  2 3
    ##  6 7
    ## [ CPUi32{2,2} ]

``` r
jit_eval({
  x[1, 2:3]
})
```

    ## AnvilTensor
    ##  2
    ##  3
    ## [ CPUi32{2} ]

``` r
jit_eval({
  x[list(2, 2), ]
})
```

    ## AnvilTensor
    ##  5 6 7 8
    ##  5 6 7 8
    ## [ CPUi32{2,4} ]

``` r
jit_eval({
  x[list(2, 2)]
})
```

    ## AnvilTensor
    ##  5 6 7 8
    ##  5 6 7 8
    ## [ CPUi32{2,4} ]

### Out-of-bounds Handling

If one specifies out-of-bounds indices, we can only throw an error if
the indices are static (we know them at compile time), as the XLA
backend that {anvil} compiles to, does not throw errors when using
out-of-bounds indices, but instead clamps them to the valid range:

``` r
jit_eval({
  x[nv_tensor(-1L), nv_tensor(100L)]
})
```

    ## AnvilTensor
    ##  4
    ## [ CPUi32{1,1} ]

``` r
jit_eval({
  x[nv_tensor(1L), nv_tensor(4L)]
})
```

    ## AnvilTensor
    ##  4
    ## [ CPUi32{1,1} ]

Therefore, you need to be careful when using dynamic indexing in order
to avoid bugs.

## Updating Subsets

Updating subsets supports the same syntax as subsetting. The value to
write must either have the shape of the subset, or be a scalar.

``` r
x
```

    ## AnvilTensor
    ##   1  2  3  4
    ##   5  6  7  8
    ##   9 10 11 12
    ## [ CPUi32{3,4} ]

``` r
jit_eval({
  x[, 3] <- nv_tensor(-(1:3))
  x
})
```

    ## AnvilTensor
    ##   1  2 -1  4
    ##   5  6 -2  8
    ##   9 10 -3 12
    ## [ CPUi32{3,4} ]

``` r
jit_eval({
  x[, 3] <- -99L
  x
})
```

    ## AnvilTensor
    ##    1   2 -99   4
    ##    5   6 -99   8
    ##    9  10 -99  12
    ## [ CPUi32{3,4} ]

Also, it must have a data type that is convertible to the data type of
the tensor.

``` r
jit_eval({
  x[, 3] <- nv_tensor(c(1.5, 2.5, 3.5))
  x
})
```

    ## Error in `nv_subset_assign()`:
    ## ! Value type f32 is not promotable to left-hand side type i32

### Out-of-bounds Handling

Similar to subsetting, out-of-bounds indices can only be checked for
static values. For dynamic indices, out-of-bounds writes are simply
ignored:

``` r
x <- nv_tensor(1:5)
jit_eval({
  x[nv_tensor(c(1L, 100L, 3L))] <- nv_tensor(c(-1L, -2L, -3L))
  x
})
```

    ## AnvilTensor
    ##  -1
    ##   2
    ##  -3
    ##   4
    ##   5
    ## [ CPUi32{5} ]

Here, the write to index 100 is silently ignored, while indices 1 and 3
are updated.

### Duplicate Indices

When writing to the same element multiple times, there is no gaurantee
which value will be written. Specifically, this might differ between
backends (CPU vs. GPU).

``` r
x <- nv_tensor(1:5)
jit_eval({
  x[list(1, 1, 1)] <- nv_tensor(c(10L, 20L, 30L))
  x
})
```

    ## AnvilTensor
    ##  30
    ##   2
    ##   3
    ##   4
    ##   5
    ## [ CPUi32{5} ]
