#' @include api.R

ops_impl <- function(e1, e2) {
  switch(
    .Generic, # nolint
    "+" = nv_add(e1, e2),
    "-" = {
      if (missing(e2)) {
        nv_negate(e1)
      } else {
        nv_sub(e1, e2)
      }
    },
    "*" = nv_mul(e1, e2),
    "/" = nv_div(e1, e2),
    "^" = nv_pow(e1, e2),
    "%%" = nv_remainder(e1, e2),
    "==" = nv_eq(e1, e2),
    "!=" = nv_ne(e1, e2),
    ">" = nv_gt(e1, e2),
    ">=" = nv_ge(e1, e2),
    "<" = nv_lt(e1, e2),
    "<=" = nv_le(e1, e2),
    "&" = nv_and(e1, e2),
    "|" = nv_or(e1, e2)
  )
}

#' @method Ops AnvilBox
#' @export
Ops.AnvilBox <- ops_impl

#' @method Ops AnvilTensor
#' @export
Ops.AnvilTensor <- ops_impl

matrix_ops_impl <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

#' @method matrixOps AnvilBox
#' @export
matrixOps.AnvilBox <- matrix_ops_impl

#' @method matrixOps AnvilTensor
#' @export
matrixOps.AnvilTensor <- matrix_ops_impl

math_impl <- function(x, ...) {
  switch(
    .Generic, # nolint
    "abs" = nv_abs(x),
    "exp" = nv_exp(x),
    "sqrt" = nv_sqrt(x),
    "log" = nv_log(x),
    "tanh" = nv_tanh(x),
    "tan" = nv_tan(x),
    "cos" = nv_cosine(x),
    "sin" = nv_sine(x),
    "floor" = nv_floor(x),
    "ceiling" = nv_ceil(x),
    "sign" = nv_sign(x),
    "round" = nv_round(x),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @method Math AnvilBox
#' @export
Math.AnvilBox <- math_impl

#' @method Math AnvilTensor
#' @export
Math.AnvilTensor <- math_impl

math2_impl <- function(x, digits, ...) {
  method <- list(...)$method
  switch(
    .Generic, # nolint
    "round" = {
      if (!missing(digits)) {
        cli_abort("Cannot specify digits")
      }
      nv_round(x, method = method)
    },
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @method Math2 AnvilBox
#' @export
Math2.AnvilBox <- math2_impl

#' @method Math2 AnvilTensor
#' @export
Math2.AnvilTensor <- math2_impl


summary_impl <- function(..., na.rm) {
  if (...length() != 1L) {
    cli_abort("Currently only one argument is supported for Summary group generic")
  }
  x <- ...elt(1L)

  dims <- seq_along(shape(x))
  switch(
    .Generic, # nolint
    "max" = nv_reduce_max(x, dims = dims, drop = TRUE),
    "min" = nv_reduce_min(x, dims = dims, drop = TRUE),
    "prod" = nv_reduce_prod(x, dims = dims, drop = TRUE),
    "sum" = nv_reduce_sum(x, dims = dims, drop = TRUE),
    "any" = nv_reduce_any(x, dims = dims, drop = TRUE),
    "all" = nv_reduce_all(x, dims = dims, drop = TRUE),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @method Summary AnvilBox
#' @export
Summary.AnvilBox <- summary_impl

#' @method Summary AnvilTensor
#' @export
Summary.AnvilTensor <- summary_impl

mean_impl <- function(x, ...) {
  nv_reduce_mean(x, dims = seq_along(shape(x)), drop = TRUE)
}

#' @method mean AnvilBox
#' @export
mean.AnvilBox <- mean_impl

#' @method mean AnvilTensor
#' @export
mean.AnvilTensor <- mean_impl

# if we don't give it the name nv_transpose, pkgdown thinks t.anvil is a package

#' @title Transpose
#' @name nv_transpose
#' @description
#' Transpose a tensor.
#' @param x ([`nv_tensor`])
#' @param permutation (`integer()` | `NULl`)\cr
#'   Permutation of dimensions. If `NULL` (default), reverses the dimensions.
#' @return [`nv_tensor`]
#' @export
#' @method t AnvilBox
t.AnvilBox <- function(x) {
  nv_transpose(x)
}

#' @method t AnvilTensor
#' @export
t.AnvilTensor <- function(x) {
  nv_transpose(x)
}
