#' @include api.R

#' @export
Ops.AnvilBox <- function(e1, e2) {
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
Ops.AnvilArray <- Ops.AnvilBox

#' @export
matrixOps.AnvilBox <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

#' @export
matrixOps.AnvilArray <- matrixOps.AnvilBox

#' @export
Math.AnvilBox <- function(x, ...) {
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
Math.AnvilArray <- Math.AnvilBox

#' @export
Math2.AnvilBox <- function(x, digits, ...) {
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
Math2.AnvilArray <- Math2.AnvilBox


#' @export
Summary.AnvilBox <- function(..., na.rm) {
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
Summary.AnvilArray <- Summary.AnvilBox

#' @export
mean.AnvilBox <- function(x, ...) {
  nv_reduce_mean(x, dims = seq_along(shape(x)), drop = TRUE)
}

#' @export
mean.AnvilArray <- mean.AnvilBox

#' @rdname nv_is_nan
#' @param x ([`arrayish`])\cr Operand.
#' @method is.nan AnvilBox
#' @export
is.nan.AnvilBox <- function(x) {
  nv_is_nan(x)
}

#' @method is.nan AnvilArray
#' @export
is.nan.AnvilArray <- is.nan.AnvilBox

#' @rdname nv_is_infinite
#' @param x ([`arrayish`])\cr Operand.
#' @method is.infinite AnvilBox
#' @export
is.infinite.AnvilBox <- function(x) {
  nv_is_infinite(x)
}

#' @method is.infinite AnvilArray
#' @export
is.infinite.AnvilArray <- is.infinite.AnvilBox

#' @rdname nv_is_finite
#' @param x ([`arrayish`])\cr Operand.
#' @method is.finite AnvilBox
#' @export
is.finite.AnvilBox <- function(x) {
  nv_is_finite(x)
}

#' @method is.finite AnvilArray
#' @export
is.finite.AnvilArray <- is.finite.AnvilBox

# if we don't give it the name nv_transpose, pkgdown thinks t.anvil is a package

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
#' @seealso [nvl_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugin_is_downloaded()
#' jit_eval({
#'   x <- nv_array(matrix(1:6, nrow = 2))
#'   t(x)
#' })
#' @export
t.AnvilBox <- function(x) {
  nv_transpose(x)
}

#' @export
t.AnvilArray <- t.AnvilBox

#' @rdname nv_subset
#' @export
`[.AnvilBox` <- function(x, ...) {
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset(x, !!!quos))
}

#' @rdname nv_subset
#' @export
`[.AnvilArray` <- `[.AnvilBox`

#' @rdname nv_subset_assign
#' @export
`[<-.AnvilBox` <- function(x, ..., value) {
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset_assign(x, !!!quos, value = value))
}

#' @rdname nv_subset_assign
#' @export
`[<-.AnvilArray` <- `[<-.AnvilBox`

#' @rdname nv_crossprod
#' @method crossprod AnvilBox
#' @export
crossprod.AnvilBox <- function(x, y = NULL, ...) {
  nv_crossprod(x, y)
}

#' @method crossprod AnvilArray
#' @export
crossprod.AnvilArray <- crossprod.AnvilBox

#' @rdname nv_tcrossprod
#' @method tcrossprod AnvilBox
#' @export
tcrossprod.AnvilBox <- function(x, y = NULL, ...) {
  nv_tcrossprod(x, y)
}

#' @method tcrossprod AnvilArray
#' @export
tcrossprod.AnvilArray <- tcrossprod.AnvilBox

# Linear-algebra generics ---------------------------------------------------
# These delegate to the corresponding nv_* implementations so that base R's
# generic dispatch (solve, qr, chol, determinant -- all S3 generics) Just
# Works on AnvilArray inputs. svd() and eigen() are not S3 generics in
# base R, so we don't add methods for them; users should call nv_svd() /
# nv_eigh() directly.

#' @rdname nv_solve
#' @method solve AnvilBox
#' @export
solve.AnvilBox <- function(a, b, ...) {
  if (missing(b)) nv_inv(a) else nv_solve(a, b)
}

#' @method solve AnvilArray
#' @export
solve.AnvilArray <- solve.AnvilBox

#' @rdname nv_qr
#' @method qr AnvilBox
#' @export
qr.AnvilBox <- function(x, ...) {
  nv_qr(x)
}

#' @method qr AnvilArray
#' @export
qr.AnvilArray <- qr.AnvilBox

#' @rdname nv_cholesky
#' @method chol AnvilBox
#' @export
chol.AnvilBox <- function(x, ...) {
  # base R's chol(A) returns upper-triangular L such that A = L^T L.
  # nv_cholesky defaults to lower; pass lower = FALSE to match base R.
  nv_cholesky(x, lower = FALSE)
}

#' @method chol AnvilArray
#' @export
chol.AnvilArray <- chol.AnvilBox

#' @rdname nv_logdet
#' @method determinant AnvilBox
#' @export
determinant.AnvilBox <- function(x, logarithm = TRUE, ...) {
  shp <- shape(x)
  if (length(shp) != 2L || shp[[1L]] != shp[[2L]]) {
    cli_abort("{.arg x} must be a square 2-D matrix")
  }
  n <- shp[[1L]]
  dt <- dtype(x)

  factored <- nvl_lu(x)
  LU <- factored[[1L]]
  pivots <- factored[[2L]]
  diag_U <- nv_extract_diag(LU)

  log_abs_det <- nv_reduce_sum(nvl_log(nvl_abs(diag_U)), dims = 1L)
  diag_sign <- nv_reduce_prod(nv_sign(diag_U), dims = 1L)
  total_sign <- nvl_mul(lu_pivot_sign(pivots, n, dt), diag_sign)

  modulus <- if (logarithm) log_abs_det else nvl_exp(log_abs_det)

  # Plain named list (not a `det`-class structure with R attributes on the
  # modulus) so the result flattens cleanly through JIT tracing. Base R's
  # `det()` is `c(z$sign * exp(z$modulus))`; the multiplication still
  # dispatches through Ops.AnvilArray and produces an AnvilArray scalar,
  # but `c()` on that result is not arrayish-aware -- users who want the
  # signed determinant should call `nv_det()` directly.
  list(modulus = modulus, sign = total_sign)
}

#' @method determinant AnvilArray
#' @export
determinant.AnvilArray <- determinant.AnvilBox
