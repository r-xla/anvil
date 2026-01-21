#' @include api.R

#' @method Ops AnvilBox
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
    "&" = nv_and(e1, e2),
    "|" = nv_or(e1, e2)
  )
}

#' @method Ops AnvilTensor
#' @export
Ops.AnvilTensor <- Ops.AnvilBox

#' @method matrixOps AnvilBox
#' @export
matrixOps.AnvilBox <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

#' @method matrixOps AnvilTensor
#' @export
matrixOps.AnvilTensor <- matrixOps.AnvilBox

#' @method Math AnvilBox
#' @export
Math.AnvilBox <- function(x, ...) {
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

#' @method Math AnvilTensor
#' @export
Math.AnvilTensor <- Math.AnvilBox

#' @method Math2 AnvilBox
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

#' @method Math2 AnvilTensor
#' @export
Math2.AnvilTensor <- Math2.AnvilBox


#' @method Summary AnvilBox
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

#' @method Summary AnvilTensor
#' @export
Summary.AnvilTensor <- Summary.AnvilBox

#' @method mean AnvilBox
#' @export
mean.AnvilBox <- function(x, ...) {
  nv_reduce_mean(x, dims = seq_along(shape(x)), drop = TRUE)
}

#' @method mean AnvilTensor
#' @export
mean.AnvilTensor <- mean.AnvilBox

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
t.AnvilTensor <- t.AnvilBox

#' @title Subset Tensor
#' @description
#' Extract elements from a tensor using `[` indexing.
#' @param x ([`tensorish`])\cr
#'   Tensor to subset.
#' @param ... Slice specifications.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop dimensions of size 1.
#' @return [`tensorish`]
#' @export
#' @method [ AnvilBox
#' @include api-subset.R
`[.AnvilBox` <- nv_subset

#' @rdname sub-.AnvilBox
#' @method [ AnvilTensor
#' @export
`[.AnvilTensor` <- nv_subset

#' @title Update Tensor Slice
#' @description
#' Update elements of a tensor using `[<-` indexing.
#' @param x ([`tensorish`])\cr
#'   Tensor to update.
#' @param ... Slice specifications.
#' @param value ([`tensorish`])\cr
#'   Values to assign.
#' @return [`tensorish`]
#' @export
#' @method [<- AnvilBox
`[<-.AnvilBox` <- function(x, ..., value) {
  call <- sys.call()
  call[[1]] <- quote(nv_subset_assign)
  eval(call, envir = parent.frame())
}

#' @rdname sub-.set.AnvilBox
#' @method [<- AnvilTensor
#' @export
`[<-.AnvilTensor` <- `[<-.AnvilBox`
