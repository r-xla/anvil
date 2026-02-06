#' @return [`tensorish`]\cr
#'   Has the same data type as the input.
#'   When `drop = TRUE`, the shape is that of `operand` with `dims` removed.
#'   When `drop = FALSE`, the shape is that of `operand` with `dims` set to 1.
#'   It is ambiguous if the input is ambiguous.
