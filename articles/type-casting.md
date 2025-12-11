# Type Casting

## Type Promotion Rules

When combining tensors of different types (e.g., adding an `f32` to an
`i32`), {anvil} needs to determine a common type. For example, below we
are adding an `f32` to an `f64`, where the former is promoted to the
latter’s type, because it’s more expressive.

``` r
library(anvil)
jit(nv_add)(
  nv_scalar(1.0, dtype = "f32"),
  nv_scalar(1.0, dtype = "f64")
)
```

    ## AnvilTensor 
    ##  2.0000
    ## [ CPUf64{} ]

The type-promotion rules are inspired by JAX, and they are designed for
execution on accelerators like GPUs, where one often wants speed instead
of precision.

The rules are defined by the
[`common_dtype()`](../reference/common_dtype.md) function.

``` r
common_dtype(dt_f64, dt_f32)
```

    ## <stablehlo::FloatType>
    ##  @ value: int 64

``` r
common_dtype(dt_i64, dt_f32)
```

    ## <stablehlo::FloatType>
    ##  @ value: int 32

A table with the promotion rules is below.

|      | i1   | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
|:-----|:-----|:----|:----|:----|:----|:-----|:-----|:-----|:-----|:----|:----|
| i1   | i1   | i8  | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| i8   | i8   | i8  | i16 | i32 | i64 | i16  | i32  | i64  | i64  | f32 | f64 |
| i16  | i16  | i16 | i16 | i32 | i64 | i16  | i32  | i64  | i64  | f32 | f64 |
| i32  | i32  | i32 | i32 | i32 | i64 | i32  | i32  | i64  | i64  | f32 | f64 |
| i64  | i64  | i64 | i64 | i64 | i64 | i64  | i64  | i64  | i64  | f32 | f64 |
| ui8  | ui8  | i16 | i16 | i32 | i64 | ui8  | ui16 | ui32 | ui64 | f32 | f64 |
| ui16 | ui16 | i32 | i32 | i32 | i64 | ui16 | ui16 | ui32 | ui64 | f32 | f64 |
| ui32 | ui32 | i64 | i64 | i64 | i64 | ui32 | ui32 | ui32 | ui64 | f32 | f64 |
| ui64 | ui64 | i64 | i64 | i64 | i64 | ui64 | ui64 | ui64 | ui64 | f32 | f64 |
| f32  | f32  | f32 | f32 | f32 | f32 | f32  | f32  | f32  | f32  | f32 | f64 |
| f64  | f64  | f64 | f64 | f64 | f64 | f64  | f64  | f64  | f64  | f64 | f64 |

Type promotion rules (row × column)

## Literals as Ambiguous Types

Usually, the types in an {anvil} program can be deterministically
inferred from the input types. The only case where this is not possible
is when you use R literals. The default types for literals are as
follows:

- [`double()`](https://rdrr.io/r/base/double.html) -\> `dt_f32`
- [`integer()`](https://rdrr.io/r/base/integer.html) -\> `dt_i32`
- [`logical()`](https://rdrr.io/r/base/logical.html) -\> `dt_i1` (bool)

``` r
jit(\() list(1L, 1.0, TRUE))()
```

    ## [[1]]
    ## AnvilTensor 
    ##  1
    ## [ CPUi32{} ] 
    ## 
    ## [[2]]
    ## AnvilTensor 
    ##  1.0000
    ## [ CPUf32{} ] 
    ## 
    ## [[3]]
    ## AnvilTensor 
    ##  1
    ## [ CPUpred{} ]

However, because this is just a guess, they behave differently than
known types during promotion. Therefore, the `common_dtype` function has
two arguments indicating which of the data types are ambiguous. Below,
the first type is a known `f64` and the second is an ambiguous `f32`.
The result is the known `f64`, although we would promote to an `f64` if
both were known. If both types are ambiguous, the result is generally
the same as if both were known.

``` r
common_dtype(dt_f32, dt_f64, FALSE, TRUE)
```

    ## <stablehlo::FloatType>
    ##  @ value: int 32

``` r
common_dtype(dt_f32, dt_f64, TRUE, TRUE)
```

    ## <stablehlo::FloatType>
    ##  @ value: int 64

``` r
common_dtype(dt_f32, dt_f64, FALSE, FALSE)
```

    ## <stablehlo::FloatType>
    ##  @ value: int 64

The promotion rules only change when one type is ambiguous and the other
is not. There, we usually promote the ambiguous type to the known type,
unless: We see, that we here only cast to the ambiguous type in two
cases:

1.  The ambiguous type is a float and the known type is not.
2.  The known type is a bool but the ambiguous type is not.

In both case, we promote the known type to the default type of the
ambiguous type. The table below shows the promotion rules, where the
rows are ambiguous and the columns are known.

|      | i1   | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
|:-----|:-----|:----|:----|:----|:----|:----|:-----|:-----|:-----|:----|:----|
| i1   | i1   | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i8   | i8   | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i16  | i16  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i32  | i32  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| i64  | i64  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui8  | ui8  | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui16 | ui16 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui32 | ui32 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| ui64 | ui64 | i8  | i16 | i32 | i64 | ui8 | ui16 | ui32 | ui64 | f32 | f64 |
| f32  | f32  | f32 | f32 | f32 | f32 | f32 | f32  | f32  | f32  | f32 | f64 |
| f64  | f64  | f64 | f64 | f64 | f64 | f64 | f64  | f64  | f64  | f32 | f64 |

Promotion rules: ambiguous (row) × known (column)

## Propagating Ambiguity

Currently, we are not propagating ambiguity through operations. I.e.,
once an operation is applied to an ambiguous type, it becomes known. We
might change this in the future.
