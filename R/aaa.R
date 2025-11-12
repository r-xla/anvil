#' @keywords internal
NULL
"_PACKAGE"

## usethis namespace: start
#' @importFrom stablehlo repr BooleanType IntegerType FloatType Shape FuncId Func as_dtype FuncVariable
#' @importFrom stablehlo local_func hlo_input hlo_return hlo_tensor hlo_scalar
#' @importFrom stablehlo UnsignedType TensorType
#' @import checkmate
#' @import S7
#' @import tengen
#' @importFrom pjrt pjrt_buffer pjrt_scalar pjrt_execute pjrt_compile pjrt_program elt_type
#' @importFrom utils gethash hashtab maphash numhash
#' @importFrom xlamisc list_of seq_len0 seq_along0
#' @importFrom utils head tail
#' @importFrom cli cli_abort
#' @importFrom methods Math2
## usethis namespace: end
NULL

globals <- new.env()
globals$nv_types <- "AnvilTensor"
globals$interpretation_rules <- c("jit", "stablehlo", "pullback", "graph", "backward")
globals[["GRAPH_STASH"]] <- list()
globals[["CURRENT_GRAPH"]] <- NULL

utils::globalVariables("globals")

hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})

aval <- S7::new_generic("aval", "x", function(x) {
  S7::S7_dispatch()
})

class_hashtab <- S7::new_S3_class("hashtab")
class_node <- S7::new_S3_class("Node")


Transformation <- new_class("Transformation")

is_transformation <- function(x) {
  inherits(x, "anvil::Transformation")
}
