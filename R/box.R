#' @title AnvilBox
#' @description
#' Virtual S3 base class for [`GraphBox`] and [`DebugBox`].
#' @seealso [DebugBox], [GraphBox], [debug_box()]
#' @name AnvilBox
NULL

is_box <- function(x) {
  inherits(x, "AnvilBox")
}
