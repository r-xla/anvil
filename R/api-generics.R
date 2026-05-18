#' @include api.R

#' @export
Ops.AnvlBox <- function(e1, e2) {
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
    "%%" = nv_mod(e1, e2),
    "==" = nv_eq(e1, e2),
    "!=" = nv_ne(e1, e2),
    ">" = nv_gt(e1, e2),
    ">=" = nv_ge(e1, e2),
    "<" = nv_lt(e1, e2),
    "<=" = nv_le(e1, e2),
    "!" = nv_not(e1),
    "&" = nv_and(e1, e2),
    "|" = nv_or(e1, e2)
  )
}

#' @export
Ops.AnvlArray <- Ops.AnvlBox

#' @export
matrixOps.AnvlBox <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

#' @export
matrixOps.AnvlArray <- matrixOps.AnvlBox

#' @export
Math.AnvlBox <- function(x, ...) {
  switch(
    .Generic, # nolint
    "abs" = nv_abs(x),
    "exp" = nv_exp(x),
    "sqrt" = nv_sqrt(x),
    "log" = nv_log(x),
    "log2" = nv_log2(x),
    "log10" = nv_log10(x),
    "tanh" = nv_tanh(x),
    "tan" = nv_tan(x),
    "cos" = nv_cos(x),
    "sin" = nv_sin(x),
    "acos" = nv_acos(x),
    "acosh" = nv_acosh(x),
    "asin" = nv_asin(x),
    "asinh" = nv_asinh(x),
    "atan" = nv_atan(x),
    "atanh" = nv_atanh(x),
    "cosh" = nv_cosh(x),
    "sinh" = nv_sinh(x),
    "digamma" = nv_digamma(x),
    "lgamma" = nv_lgamma(x),
    "trigamma" = nv_polygamma(1, x),
    "floor" = nv_floor(x),
    "ceiling" = nv_ceiling(x),
    "trunc" = nv_trunc(x),
    "sign" = nv_sign(x),
    "expm1" = nv_expm1(x),
    "log1p" = nv_log1p(x),
    "round" = nv_round(x),
    "cumsum" = nv_cumsum(x),
    "cumprod" = nv_cumprod(x),
    "cummax" = nv_cummax(x),
    "cummin" = nv_cummin(x),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @export
Math.AnvlArray <- Math.AnvlBox

#' @export
Math2.AnvlBox <- function(x, digits, ...) {
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

#' @export
Math2.AnvlArray <- Math2.AnvlBox


#' @export
Summary.AnvlBox <- function(..., na.rm) {
  if (...length() != 1L) {
    cli_abort("Currently only one argument is supported for Summary group generic")
  }
  x <- ...elt(1L)

  switch(
    .Generic, # nolint
    "max" = nv_reduce_max(x),
    "min" = nv_reduce_min(x),
    "range" = nv_concatenate(nv_reduce_min(x), nv_reduce_max(x)),
    "prod" = nv_reduce_prod(x),
    "sum" = nv_reduce_sum(x),
    "any" = nv_reduce_any(x),
    "all" = nv_reduce_all(x),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @export
Summary.AnvlArray <- Summary.AnvlBox

#' @method mean AnvlBox
#' @export
mean.AnvlBox <- function(x, trim = 0, na.rm = FALSE, ..., dims = NULL, drop = TRUE) {
  if (isTRUE(na.rm)) {
    cli_abort("{.code na.rm = TRUE} is not supported: anvl arrays do not carry {.code NA}s.")
  }
  if (!identical(trim, 0)) {
    cli_abort("{.arg trim} is not supported by {.fn mean} for anvl arrays.")
  }
  rlang::check_dots_empty()
  nv_mean(x, dims = dims, drop = drop)
}

#' @rdname nv_mean
#' @template param_x_operand
#' @param trim,na.rm Currently not supported.
#' @param ... No additional arguments.
#' @method mean AnvlArray
#' @export
mean.AnvlArray <- mean.AnvlBox

#' @method is.nan AnvlBox
#' @export
is.nan.AnvlBox <- function(x) {
  nv_is_nan(x)
}

#' @rdname nv_is_nan
#' @template param_x_operand
#' @method is.nan AnvlArray
#' @export
is.nan.AnvlArray <- is.nan.AnvlBox

#' @method is.infinite AnvlBox
#' @export
is.infinite.AnvlBox <- function(x) {
  nv_is_infinite(x)
}

#' @rdname nv_is_infinite
#' @template param_x_operand
#' @method is.infinite AnvlArray
#' @export
is.infinite.AnvlArray <- is.infinite.AnvlBox

#' @method is.finite AnvlBox
#' @export
is.finite.AnvlBox <- function(x) {
  nv_is_finite(x)
}

#' @rdname nv_is_finite
#' @template param_x_operand
#' @method is.finite AnvlArray
#' @export
is.finite.AnvlArray <- is.finite.AnvlBox

#' @method t AnvlBox
#' @export
t.AnvlBox <- function(x) {
  nd <- ndims(x)
  if (nd != 2L) {
    cli_abort("{.fn t} requires a 2-D array, but got a {nd}-D array.")
  }
  nv_transpose(x)
}

# if we don't give it the name nv_transpose, pkgdown thinks t.anvl is a package

#' @title Transpose
#' @name nv_transpose
#' @description
#' Permutes the dimensions of an array. You can also use `t()` for matrices.
#' @template param_x_operand
#' @param permutation (`integer()` | `NULL`)\cr
#'   New ordering of dimensions. If `NULL` (default), reverses the dimensions.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand` and shape `nv_shape(operand)[permutation]`.
#' @seealso [prim_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' t(x)
#' @method t AnvlArray
#' @export
t.AnvlArray <- t.AnvlBox

#' @method median AnvlBox
#' @export
median.AnvlBox <- function(x, na.rm = FALSE, ..., dim = NULL, interpolation = "linear") {
  if (isTRUE(na.rm)) {
    cli_abort("{.code na.rm = TRUE} is not supported: anvl arrays do not carry {.code NA}s.")
  }
  rlang::check_dots_empty()
  nv_median(x, dim = dim, interpolation = interpolation)
}

#' @rdname nv_median
#' @template param_x_operand
#' @param na.rm Currently not supported.
#' @param ... No additional arguments.
#' @method median AnvlArray
#' @export
median.AnvlArray <- median.AnvlBox

#' @method sort AnvlBox
#' @export
sort.AnvlBox <- function(x, decreasing = FALSE, ..., dim = NULL) {
  rlang::check_dots_empty()
  nv_sort(x, decreasing = decreasing, dim = dim)
}

#' @rdname nv_sort
#' @template param_x_operand
#' @param decreasing (`logical(1)`)\cr If `TRUE`, sort in decreasing order.
#' @param ... No additional arguments.
#' @method sort AnvlArray
#' @export
sort.AnvlArray <- sort.AnvlBox

#' @method [ AnvlBox
#' @export
`[.AnvlBox` <- function(x, ...) {
  # nargs() sees trailing missing args (e.g. the last `,` in x[1:5, , ])
  # that rlang::enquos() silently drops.
  n_args <- nargs() - 1L
  rank <- length(shape_abstract(x))
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset(x, !!!quos))
}

#' @rdname nv_subset
#' @template param_x_operand
#' @export
`[.AnvlArray` <- `[.AnvlBox`

#' @method [<- AnvlBox
#' @export
`[<-.AnvlBox` <- function(x, ..., value) {
  n_args <- nargs() - 2L
  rank <- length(shape_abstract(x))
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset_assign(x, !!!quos, value = value))
}

#' @rdname nv_subset_assign
#' @template param_x_operand
#' @export
`[<-.AnvlArray` <- `[<-.AnvlBox`

#' @method crossprod AnvlBox
#' @export
crossprod.AnvlBox <- function(x, y = NULL, ...) {
  rlang::check_dots_empty()
  nv_crossprod(x, y)
}

#' @rdname nv_crossprod
#' @param x,y Same as `lhs` and `rhs`; the names used by the base R S3 generic.
#' @param ... No additional arguments.
#' @method crossprod AnvlArray
#' @export
crossprod.AnvlArray <- crossprod.AnvlBox

#' @method tcrossprod AnvlBox
#' @export
tcrossprod.AnvlBox <- function(x, y = NULL, ...) {
  rlang::check_dots_empty()
  nv_tcrossprod(x, y)
}

#' @rdname nv_tcrossprod
#' @param x,y Same as `lhs` and `rhs`; the names used by the base R S3 generic.
#' @param ... No additional arguments.
#' @method tcrossprod AnvlArray
#' @export
tcrossprod.AnvlArray <- tcrossprod.AnvlBox

#' @method dim AnvlBox
#' @export
dim.AnvlBox <- function(x) {
  shape(x)
}

#' @method dim AnvlArray
#' @export
dim.AnvlArray <- dim.AnvlBox

#' @method length AnvlBox
#' @export
length.AnvlBox <- function(x) {
  prod(shape(x))
}

#' @method length AnvlArray
#' @export
length.AnvlArray <- length.AnvlBox

#' @method rbind AnvlBox
#' @export
rbind.AnvlBox <- function(..., deparse.level = 1) {
  nv_rbind(...)
}

#' @rdname nv_bind
#' @param deparse.level Ignored. Kept for compatibility with [base::rbind()]
#'   and [base::cbind()].
#' @method rbind AnvlArray
#' @export
rbind.AnvlArray <- rbind.AnvlBox

#' @method cbind AnvlBox
#' @export
cbind.AnvlBox <- function(..., deparse.level = 1) {
  nv_cbind(...)
}

#' @rdname nv_bind
#' @method cbind AnvlArray
#' @export
cbind.AnvlArray <- cbind.AnvlBox

#' @method solve AnvlBox
#' @export
solve.AnvlBox <- function(a, b, ...) {
  if (missing(b)) nv_inv(a, ...) else nv_solve(a, b, ...)
}

#' @rdname nv_solve
#' @param a ([`arrayish`])\cr Coefficient matrix.
#' @param b ([`arrayish`])\cr Right-hand side. If missing, returns [nv_inv()] of `a`.
#' @param ... No additional arguments.
#' @method solve AnvlArray
#' @export
solve.AnvlArray <- solve.AnvlBox

#' @method qr AnvlBox
#' @export
qr.AnvlBox <- function(x, ...) {
  nv_qr(x, ...)
}

#' @rdname nv_qr
#' @template param_x_operand
#' @param ... No additional arguments.
#' @method qr AnvlArray
#' @export
qr.AnvlArray <- qr.AnvlBox

#' @method chol AnvlBox
#' @export
chol.AnvlBox <- function(x, ..., lower = FALSE) {
  nv_chol(x, lower = lower, ...)
}

#' @rdname nv_chol
#' @template param_x_operand
#' @param lower (`logical(1)`)\cr If `TRUE`, return the lower-triangular factor.
#' @param ... No additional arguments.
#' @method chol AnvlArray
#' @export
chol.AnvlArray <- chol.AnvlBox

#' @method determinant AnvlBox
#' @export
determinant.AnvlBox <- function(x, logarithm = TRUE, ...) {
  nv_determinant(x, logarithm = logarithm, ...)
}

#' @rdname nv_determinant
#' @template param_x_operand
#' @param logarithm (`logical(1)`)\cr If `TRUE` (default), return the log
#'   of the absolute determinant.
#' @param ... No additional arguments.
#' @method determinant AnvlArray
#' @export
determinant.AnvlArray <- determinant.AnvlBox
