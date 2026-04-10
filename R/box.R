#' @title AnvilBox
#' @description
#' Virtual S3 base class for [`GraphBox`].
#' @seealso [GraphBox]
#' @name AnvilBox
NULL

is_box <- function(x) {
  inherits(x, "AnvilBox")
}
