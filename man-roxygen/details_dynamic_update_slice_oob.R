#' @details
#' ## Out-of-Bounds Behavior
#'
#' If `start_indices` would cause the update to extend beyond the operand's bounds,
#' the indices are automatically adjusted (clamped) to keep the update within bounds.
#' The adjustment formula is:
#'
#' ```
#' adjusted_start_indices = clamp(1, start_indices, shape(operand) - shape(update) + 1)
#' ```
#'
#' This means:
#' - Negative or zero indices are clamped to 1 (the minimum valid 1-based index).
#' - Indices that would cause the update to go beyond the operand are clamped to
#'   ensure the update fits within the operand's bounds.
#' - Only the portion of `update` that overlaps with the clamped region is written
#'   to `operand`.
