#' @keywords internal
NULL
"_PACKAGE"

## usethis namespace: start
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
globals$nv_types <- "AnvilTensor"
globals$interpretation_rules <- c("stablehlo", "quickr", "backward")
globals[["DESCRIPTOR_STASH"]] <- list()
globals[["CURRENT_DESCRIPTOR"]] <- NULL
globals$backend <- "xla"

utils::globalVariables(c("globals"))

normalize_backend <- function(backend) {
  assert_string(backend)
  backend <- tolower(backend)
  assert_choice(backend, c("xla", "quickr"))
  backend
}

current_backend <- function() {
  desc <- .current_descriptor(silent = TRUE)
  if (!is.null(desc) && !is.null(desc$backend)) {
    return(desc$backend)
  }
  globals$backend
}

#' Temporarily override the active backend
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use (`"xla"` or `"quickr"`).
#' @param code Expression to evaluate with the overridden backend.
#' @return The result of evaluating `code`.
#' @keywords internal
with_backend <- function(backend, code) {
  backend <- normalize_backend(backend)
  old <- globals$backend
  globals$backend <- backend
  on.exit(globals$backend <- old, add = TRUE)
  force(code)
}
