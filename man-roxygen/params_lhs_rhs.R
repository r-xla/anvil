#' @param lhs,rhs ([`tensorish`])\cr
#'   Left and right operand.
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#'   Scalars are [broadcast][nv_broadcast_scalars()] to the shape of the other operand.
