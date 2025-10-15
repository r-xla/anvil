#' @include interpreter.R

#' @export
`Ops.anvil::Box` <- function(e1, e2) {
  switch(
    .Generic, # nolint
    "+" = nv_add(e1, e2),
    "-" = {
      if (missing(e2)) {
        nv_neg(e1)
      } else {
        nv_sub(e1, e2)
      }
    },
    "*" = nv_mul(e1, e2)
  )
}

#' @export
`matrixOps.anvil::Box` <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

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
`t.anvil::Box` <- function(x) {
  nv_transpose(x)
}
