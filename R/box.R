#' @title AnvlBox
#' @description
#' Virtual S3 base class for [`GraphBox`].
#' @seealso [GraphBox]
#' @name AnvlBox
NULL

is_box <- function(x) {
  inherits(x, "AnvlBox")
}
