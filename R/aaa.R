#' @keywords internal
#' @include reexports.R
#' @include list_of.R
#' @include aval.R
NULL
"_PACKAGE"

## usethis namespace: start
#' @importFrom tengen shape dtype
#' @import stablehlo
#' @import pjrt
#' @import checkmate
#' @import S7
## usethis namespace: end
NULL

globals <- new.env()
globals$nvl_types <- c("nvl_tensor", "nvl_scalar")

utils::globalVariables("globals")


hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})
