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
    "%/%" = nv_floor(nv_div(e1, e2)),
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
    "range" = {
      mn <- nv_reduce_min(x, dims = dims, drop = TRUE)
      mx <- nv_reduce_max(x, dims = dims, drop = TRUE)
      nv_concatenate(
        prim_reshape(mn, shape = 1L),
        prim_reshape(mx, shape = 1L),
        dimension = 1L
      )
    },
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

#' @export
solve.AnvlBox <- function(a, b, ...) {
  if (missing(b)) {
    cli_abort("solve.AnvlBox requires `b`; matrix inverse is not supported")
  }
  nv_solve(a, b)
}

#' @export
solve.AnvlArray <- solve.AnvlBox

#' @export
rev.AnvlBox <- function(x) {
  rank <- length(shape(x))
  if (rank > 1L) {
    cli_abort("rev() is only defined for 1-D arrayish, got rank {rank}")
  }
  nv_reverse(x, dims = 1L)
}

#' @export
rev.AnvlArray <- rev.AnvlBox

#' @export
aperm.AnvlBox <- function(a, perm = NULL, ...) {
  nv_transpose(a, permutation = perm)
}

#' @export
aperm.AnvlArray <- aperm.AnvlBox

#' @export
head.AnvlBox <- function(x, n = 6L, ...) {
  if (length(n) != 1L) {
    cli_abort("head.AnvlBox: vector `n` is not supported; use a single integer")
  }
  len <- shape(x)[1L]
  k <- if (n < 0L) max(len + n, 0L) else min(n, len)
  nv_subset(x, array(seq_len(k)))
}

#' @export
head.AnvlArray <- head.AnvlBox

#' @export
tail.AnvlBox <- function(x, n = 6L, ...) {
  if (length(n) != 1L) {
    cli_abort("tail.AnvlBox: vector `n` is not supported; use a single integer")
  }
  len <- shape(x)[1L]
  k <- if (n < 0L) max(len + n, 0L) else min(n, len)
  nv_subset(x, array(seq.int(to = len, length.out = k)))
}

#' @export
tail.AnvlArray <- tail.AnvlBox

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
