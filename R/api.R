# This is the user-facing API containing the exported tensor operations.

broadcast_shapes <- function(shape_lhs, shape_rhs) {
  if (length(shape_lhs) > length(shape_rhs)) {
    shape_rhs <- c(rep(1L, length(shape_lhs) - length(shape_rhs)), shape_rhs)
  } else if (length(shape_lhs) < length(shape_rhs)) {
    shape_lhs <- c(rep(1L, length(shape_rhs) - length(shape_lhs)), shape_lhs)
  } else if (identical(shape_lhs, shape_rhs)) {
    return(shape_lhs)
  }
  shape_out <- shape_lhs
  for (i in seq_along(shape_lhs)) {
    d_lhs <- shape_lhs[i]
    d_rhs <- shape_rhs[i]
    if (d_lhs != d_rhs && d_lhs != 1L && d_rhs != 1L) {
      stop("lhs and rhs are not broadcastable")
    }
    shape_out[i] <- max(d_lhs, d_rhs)
  }
  shape_out
}

nvl_broadcast <- function(lhs, rhs) {
  prim_broadcast_in_dim(lhs, rhs)
}

nvl_add <- function(lhs, rhs) {
  broad <- broadcas
  bind_one(prim_add, list(lhs, rhs))
}


#generic_to_primitive <- as.environment(list(
#  "+" = prim_add,
#  "*" = prim_mul,
#  "-" = prim_sub,
#  "/" = prim_div
#))

## Group Generics

#' @export
Math.nvl_tensor <- function(e1, e2) {
  args <- if (missing(e2)) list(e1) else list(e1, e2)
  bind_one(generic_to_primitive[[.Generic]], args) # nolint
}

#' @export
matrixOps.nvl_tensor <- function(e1, e2) {
  if (!missing(e2)) {
    bind_one(generic_to_primitive[[.Generic]], list(e1, e2)) # nolint
  } else {
    e1
  }
}

#' @export
Ops.nvl_tensor <- function(e1, e2) {
  if (!missing(e2)) {
    bind_one(generic_to_primitive[[.Generic]], list(e1, e2)) # nolint
  } else {
    e1
  }
}

# TODO: transpose
##' @export
#t.nvl_tensor <- function(x) {
#  bind_one(prim_transpose, list(x))
#}
