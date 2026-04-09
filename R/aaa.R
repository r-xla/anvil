#' @keywords internal
NULL
"_PACKAGE"

## usethis namespace: start
#' @useDynLib anvil, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stablehlo repr Shape FuncId Func FuncValue
#' @importFrom stablehlo local_func hlo_input hlo_return hlo_tensor hlo_scalar
#' @importFrom stablehlo TensorType
#' @import checkmate
#' @import tengen
#' @importFrom pjrt pjrt_buffer pjrt_scalar pjrt_execute pjrt_compile pjrt_program elt_type
#' @importFrom utils gethash hashtab maphash numhash
#' @importFrom xlamisc seq_len0 seq_along0
#' @importFrom utils head tail
#' @importFrom cli cli_abort
#' @importFrom methods Math2 formalArgs
#' @importFrom utils capture.output
## usethis namespace: end
NULL

globals <- new.env()
globals$nv_types <- "AnvilArray"
globals$interpretation_rules <- c("stablehlo", "quickr", "reverse")
globals[["DESCRIPTOR_STASH"]] <- list()
globals[["CURRENT_DESCRIPTOR"]] <- NULL
utils::globalVariables(c("globals"))
