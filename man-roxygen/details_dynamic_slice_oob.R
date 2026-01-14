#' @details
#' ## Out-of-Bounds Behavior
#'
#' If `start_indices` would cause the slice to extend beyond the operand's bounds,
#' the indices are automatically adjusted (clamped) to keep the slice within bounds.
#' The adjustment formula is:
#'
#' ```
#' adjusted_start_indices = clamp(1, start_indices, shape(operand) - slice_sizes + 1)
#' ```
#'
#' This means:
#' - Negative or zero indices are clamped to 1 (the minimum valid 1-based index).
#' - Indices that would cause the slice to go beyond the operand are clamped to
#'   ensure the slice fits within the operand's bounds.
