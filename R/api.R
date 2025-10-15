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
    # When ranks match, each input dimension maps to the same output dimension
    # StableHLO expects a mapping for every input dim
    return(seq_along(shape_out))
  }
  tail(seq_len(rank_out), rank_in)
}

#' @title Broadcast Tensors to a Common Shape
#' @description
#' Broadcast tensors to a common shape.
#'
#' @section Broadcasting Rules:
#' We follow the standard NumPy broadcasting rules:
#' 1. If the tensors have different numbers of dimensions, prepend 1s to the shape of the smaller tensor.
#' 2. For each dimensions, if:
#'    - If the sizes are the same, do nothing.
#'    - One of the tensors has size 1, expand it to the size of the other tensor.
#'    - If the sizes are different and neither is 1, raise an error.
#'
#' @param ... (`list()` of [`nv_tensor`])
#' @return (`list()` of [`nv_tensor`])
#' @export
nv_broadcast_tensors <- function(...) {
  args <- list(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast
#' @param operand ([`nv_tensor`])\cr
#'   Operand.
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return ([`nv_tensor`])
#' @export
nv_broadcast_to <- function(operand, shape) {
  if (!identical(shape(operand), shape)) {
    broadcast_dimensions <- make_broadcast_dimensions(shape(operand), shape)
    nvl_broadcast_in_dim(operand, shape, broadcast_dimensions)
  } else {
    operand
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
  do.call(nvl_add, nv_broadcast_tensors(lhs, rhs))
}

#' @rdname nv_binary_ops
#' @export
nv_mul <- function(lhs, rhs) {
  do.call(nvl_mul, nv_broadcast_tensors(lhs, rhs))
}

#' @rdname nv_binary_ops
#' @export
nv_sub <- function(lhs, rhs) {
  do.call(nvl_sub, nv_broadcast_tensors(lhs, rhs))
}

#' @rdname nv_binary_ops
#' @export
nv_div <- function(lhs, rhs) {
  do.call(nvl_div, nv_broadcast_tensors(lhs, rhs))
}

#' @rdname nv_binary_ops
#' @export
nv_pow <- function(lhs, rhs) {
  do.call(nvl_pow, nv_broadcast_tensors(lhs, rhs))
}

## Unary ops ------------------------------------------------------------------

#' @name nv_unary_ops
#' @title Unary Operations
#' @description
#' Unary operations on tensors.
#' @param operand ([`nv_tensor`])
#' @return [`nv_tensor`]

#' @rdname nv_unary_ops
#' @export
nv_neg <- nvl_neg


## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two tensors.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @section Broadcasting:
#' All dimensions but the last two are broadcasted.
#' @param lhs ([`nv_tensor`])
#' @param rhs ([`nv_tensor`])
#' @return [`nv_tensor`]
#' @export
nv_matmul <- function(lhs, rhs) {
  if (ndims(rhs) < 2L) {
    stop("lhs of matmul must have at least 2 dimensions")
  }
  if (ndims(lhs) < 2L) {
    stop("rhs of matmul must have at least 2 dimensions")
  }
  shape_leading <- broadcast_shapes(head(shape(lhs), -2L), head(shape(rhs), -2L))

  shape_lhs <- c(shape_leading, tail(shape(lhs), 2L))
  shape_rhs <- c(shape_leading, tail(shape(rhs), 2L))

  if (!identical(shape_lhs, shape(lhs))) {
    lhs <- nv_broadcast_to(lhs, shape_lhs)
  }
  if (!identical(shape_rhs, shape(rhs))) {
    rhs <- nv_broadcast_to(rhs, shape_rhs)
  }

  nvl_dot_general(
    lhs,
    rhs,
    contracting_dims = list(ndims(lhs), ndims(rhs) - 1L),
    batching_dims = list(seq_along(shape_leading), seq_along(shape_leading))
  )
}

#' @rdname nv_transpose
#' @export
nv_transpose <- function(x, permutation = NULL) {
  permutation <- permutation %??% rev(seq_len(ndims(x)))
  nvl_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshape a tensor.
#' @param operand ([`nv_tensor`])\cr
#'   The tensor.
#' @param shape (`integer()`)\cr
#'   Output shape.
#' @return ([`nv_tensor`])
#' @export
nv_reshape <- nvl_reshape

#' @title Reduction Operators
#' @description
#' Reduce a tensor along specified dimensions.
#' @param operand ([`nv_tensor`])\cr
#'   The tensor.
#' @param dims (`integer()`)\cr
#'   The dimensions along which to reduce.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions.
#' @name nv_reduce_ops
#' @export
nv_reduce_sum <- nvl_reduce_sum

## Data Types ------------------------------------------------------------------

#' @title Tensor Data Types
#' @name data_types
#' @description
#' Data types for tensors:
#' - `dt_i1`: boolean.
#' - `dt_i{8, 16, 32, 64}`: signed integer.
#' - `dt_ui{8, 16, 32, 64}`: unsigned integer.
#' - `dt_f{32, 64}`: float.
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
