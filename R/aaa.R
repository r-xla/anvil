#' @keywords internal
#' @include reexports.R
#' @include list_of.R
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
globals$nv_types <- c("AnvilTensor", "nv_scalar")

utils::globalVariables("globals")


hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})

aval <- S7::new_generic("aval", "x", function(x) {
  S7::S7_dispatch()
})
