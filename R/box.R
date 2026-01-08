#' @title Box
#' @description
#' Virtual base class for GraphBox and DebugBox.
#' Cannot be instantiated directly.
#' @return Never returns; always errors.
#' @export
Box <- function() {
  cli_abort("Box is a virtual class and cannot be instantiated directly")
}
