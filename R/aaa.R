#' @keywords internal
#' @include reexports.R
#' @include list_of.R
NULL
"_PACKAGE"

## usethis namespace: start
#' @importFrom stablehlo repr BooleanType IntegerType FloatType Shape FuncId Func as_dtype FuncVariable
#' @importFrom stablehlo UnsignedType
#' @import checkmate
#' @import S7
#' @import tengen
#' @importFrom pjrt pjrt_buffer pjrt_scalar pjrt_execute pjrt_compile pjrt_program elt_type
#' @importFrom utils gethash hashtab maphash numhash
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
