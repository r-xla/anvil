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
    "%%" = nv_remainder(e1, e2),
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
    "cos" = nv_cosine(x),
    "sin" = nv_sine(x),
    "floor" = nv_floor(x),
    "ceiling" = nv_ceil(x),
    "sign" = nv_sign(x),
    "expm1" = nv_expm1(x),
    "log1p" = nv_log1p(x),
    "round" = nv_round(x),
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

#' @export
Summary.AnvlArray <- Summary.AnvlBox

#' @export
mean.AnvlBox <- function(x, ...) {
  nv_reduce_mean(x, dims = seq_along(shape(x)), drop = TRUE)
}

#' @export
mean.AnvlArray <- mean.AnvlBox

#' @rdname nv_is_nan
#' @param x ([`arrayish`])\cr Operand.
#' @method is.nan AnvlBox
#' @export
is.nan.AnvlBox <- function(x) {
  nv_is_nan(x)
}

#' @method is.nan AnvlArray
#' @export
is.nan.AnvlArray <- is.nan.AnvlBox

#' @rdname nv_is_infinite
#' @param x ([`arrayish`])\cr Operand.
#' @method is.infinite AnvlBox
#' @export
is.infinite.AnvlBox <- function(x) {
  nv_is_infinite(x)
}

#' @method is.infinite AnvlArray
#' @export
is.infinite.AnvlArray <- is.infinite.AnvlBox

#' @rdname nv_is_finite
#' @param x ([`arrayish`])\cr Operand.
#' @method is.finite AnvlBox
#' @export
is.finite.AnvlBox <- function(x) {
  nv_is_finite(x)
}

#' @method is.finite AnvlArray
#' @export
is.finite.AnvlArray <- is.finite.AnvlBox

# if we don't give it the name nv_transpose, pkgdown thinks t.anvl is a package

#' @title Transpose
#' @name nv_transpose
#' @description
#' Permutes the dimensions of an array. You can also use `t()` for matrices.
#' @param x ([`arrayish`])\cr
#'   Array to transpose.
#' @param permutation (`integer()` | `NULL`)\cr
#'   New ordering of dimensions. If `NULL` (default), reverses the dimensions.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x` and shape `nv_shape(x)[permutation]`.
#' @seealso [prim_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(matrix(1:6, nrow = 2))
#' t(x)
#' @export
t.AnvlBox <- function(x) {
  nv_transpose(x)
}

#' @export
t.AnvlArray <- t.AnvlBox

#' @rdname nv_subset
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
#' @export
`[.AnvlArray` <- `[.AnvlBox`

#' @rdname nv_subset_assign
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
#' @export
`[<-.AnvlArray` <- `[<-.AnvlBox`

#' @rdname nv_crossprod
#' @method crossprod AnvlBox
#' @export
crossprod.AnvlBox <- function(x, y = NULL, ...) {
  nv_crossprod(x, y)
}

#' @method crossprod AnvlArray
#' @export
crossprod.AnvlArray <- crossprod.AnvlBox

#' @rdname nv_tcrossprod
#' @method tcrossprod AnvlBox
#' @export
tcrossprod.AnvlBox <- function(x, y = NULL, ...) {
  nv_tcrossprod(x, y)
}

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

#' @title Combine arrays by rows or columns
#' @name nv_bind
#' @description
#' Combine arrays along the row (`nv_rbind`) or column (`nv_cbind`) dimension.
#' Arguments are promoted to a common data type before being combined.
#'
#' Inputs of rank 0 (scalars) are reshaped to `c(1, 1)`. Inputs of rank 1 are
#' treated as a single row (for `nv_rbind`) or a single column (for
#' `nv_cbind`). Inputs of rank >= 2 are concatenated along dimension 1
#' (`nv_rbind`) or dimension 2 (`nv_cbind`); their other dimensions must
#' match.
#'
#' [base::rbind()] and [base::cbind()] dispatch to these via S3 when at least
#' one argument is an [`AnvlBox`] or [`AnvlArray`].
#' @param ... ([`arrayish`])\cr
#'   Arrays to combine.
#' @param deparse.level Ignored. Kept for compatibility with [base::rbind()]
#'   and [base::cbind()].
#' @return [`arrayish`]\cr
#'   A rank-2 array of the common data type.
#' @seealso [nv_concatenate()] for concatenation along an arbitrary dimension.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:3)
#' y <- nv_array(4:6)
#' nv_rbind(x, y)
#' nv_cbind(x, y)
NULL

bind_reshape_for_rbind <- function(arg) {
  arg <- as_anvl_array(arg)
  s <- shape(arg)
  if (length(s) == 0L) {
    nv_reshape(arg, c(1L, 1L))
  } else if (length(s) == 1L) {
    nv_reshape(arg, c(1L, s))
  } else {
    arg
  }
}

bind_reshape_for_cbind <- function(arg) {
  arg <- as_anvl_array(arg)
  s <- shape(arg)
  if (length(s) == 0L) {
    nv_reshape(arg, c(1L, 1L))
  } else if (length(s) == 1L) {
    nv_reshape(arg, c(s, 1L))
  } else {
    arg
  }
}

#' @rdname nv_bind
#' @export
nv_rbind <- function(...) {
  args <- lapply(list(...), bind_reshape_for_rbind)
  rlang::exec(nv_concatenate, !!!args, dimension = 1L)
}

#' @rdname nv_bind
#' @export
nv_cbind <- function(...) {
  args <- lapply(list(...), bind_reshape_for_cbind)
  rlang::exec(nv_concatenate, !!!args, dimension = 2L)
}

#' @rdname nv_bind
#' @method rbind AnvlBox
#' @export
rbind.AnvlBox <- function(..., deparse.level = 1) {
  nv_rbind(...)
}

#' @rdname nv_bind
#' @method rbind AnvlArray
#' @export
rbind.AnvlArray <- rbind.AnvlBox

#' @rdname nv_bind
#' @method cbind AnvlBox
#' @export
cbind.AnvlBox <- function(..., deparse.level = 1) {
  nv_cbind(...)
}

#' @rdname nv_bind
#' @method cbind AnvlArray
#' @export
cbind.AnvlArray <- cbind.AnvlBox
