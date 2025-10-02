# This is the user-facing API containing the exported tensor operations.
#' @include interpreter.R
#' @include primitives.R

## Broadcasting ----------------------------------------------------------------

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

make_broadcast_dimensions <- function(shape_in, shape_out) {
  rank_in <- length(shape_in)
  rank_out <- length(shape_out)
  if (rank_in == rank_out) {
    return(integer())
  }
  rank_in + seq_along(rank_out - rank_in) - 1L
}

#' @title Broadcast
#' @param ... ([`nv_tensor`])
#' @param shape (`integer()`)\cr
#'   Output shape. If `NULL` (default),
#' @return ([`nv_tensor`])
#' @export
nv_broadcast_all <- function(...) {
  args <- list(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast, shape = shape)
}

#' @title Broadcast
#' @param x ([`nv_tensor`])
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return ([`nv_tensor`])
#' @export
nv_broadcast <- function(x, shape) {
  if (!identical(shape(x), shape)) {
    broadcast_dimensions <- make_broadcast_dimensions(shape(x), shape)
    nv_broadcast_in_dim(x, shape, broadcast_dimensions)
  } else {
    x
  }
}

## Binary ops ------------------------------------------------------------------

#' @name nv_binary_ops
#' @title Binary Operations
#' @description
#' Binary operations on tensors.
#' @param lhs ([`nv_tensor`])
#' @param rhs ([`nv_tensor`])
#' @return [`nv_tensor`]
NULL

#' @rdname nv_binary_ops
#' @export
nv_add <- function(lhs, rhs) {
  do.call(nvl_add, nv_broadcast_all(lhs, rhs))
}

#' @export
nv_mul <- function(lhs, rhs) {
  do.call(nvl_mul, nv_broadcast_all(lhs, rhs))
}

#' @export
nv_sub <- function(lhs, rhs) {
  do.call(nvl_sub, nv_broadcast_all(lhs, rhs))
}

## Unary ops ------------------------------------------------------------------

#' @name nv_unary_ops
#' @title Unary Operations
#' @description
#' Unary operations on tensors.
#' @param operand ([`nv_tensor`])
#' @return [`nv_tensor`]

#' @export
nv_neg <- nvl_neg


## Data Types ------------------------------------------------------------------

#' @title Tensor Data Types
#' @name data_types
#' @description
#' Data types for tensors:
#' - `dt_i1`: boolean.
#' - `dt_i{8, 16, 32, 64}`: signed integer.
#' - `dt_ui{8, 16, 32, 64}`: unsigned integer.
#' - `dt_f{32, 64}`: float.
#' @examples
NULL

#' @rdname data_types
#' @format NULL
#' @usage NULL
#' @export
dt_i1 <- BooleanType()
#' @rdname data_types
#' @format NULL
#' @usage NULL
#' @format NULL
#' @usage NULL
#' @export
dt_i8 <- IntegerType(8)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i16 <- IntegerType(16)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i32 <- IntegerType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_i64 <- IntegerType(64)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui8 <- UnsignedType(8)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui16 <- UnsignedType(16)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui32 <- UnsignedType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_ui64 <- UnsignedType(64)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_f32 <- FloatType(32)
#' @export
#' @rdname data_types
#' @format NULL
#' @usage NULL
dt_f64 <- FloatType(64)
