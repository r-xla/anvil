#' @title AnvilBox
#' @description
#' Virtual base class for GraphBox and DebugBox.
#' This class is used to represent values during graph construction and debugging.
#' Cannot be instantiated directly - use [`GraphBox()`] or [`DebugBox()`] instead.
#' @name AnvilBox
NULL

is_box <- function(x) {
  inherits(x, "AnvilBox")
}
