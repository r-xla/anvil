# All the backward rules are only operating on GraphValues

# length(grads) == length(outputs)
p_add[["backward"]] <- function(inputs, outputs, grads, .required) {
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) grad,
    if (.required[[2L]]) grad
  )
}

p_mul[["backward"]] <- function(inputs, outputs, grads, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_mul(grad, rhs),
    if (.required[[2L]]) nvl_mul(grad, lhs)
  )
}

p_sub[["backward"]] <- function(inputs, outputs, grads, .required) {
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) grad,
    if (.required[[2L]]) nvl_neg(grad)
  )
}

p_neg[["backward"]] <- function(inputs, outputs, grads, .required) {
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_neg(grad)
  )
}

p_div[["backward"]] <- function(inputs, outputs, grads, .required) {
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_div(grad, rhs),
    if (.required[[2L]]) nvl_div(nvl_mul(grad, nvl_neg(y)), rhs)
  )
}

p_pow[["backward"]] <- function(inputs, outputs, grads, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) {
      one <- nv_full(1, dtype = dtype(lhs), shape = shape(rhs))
      nvl_mul(nvl_mul(grad, rhs), nvl_pow(lhs, nvl_sub(rhs, one)))
    },
    if (.required[[2L]]) {
      nvl_mul(grad, nvl_mul(nvl_log(lhs), y))
    }
  )
}

p_log[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_div(grad, operand)
  )
}

p_exp[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_mul(grad, y)
  )
}

p_dot_general[["backward"]] <- function(inputs, outputs, grads, contracting_dims, batching_dims, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]

  # batching dimensions
  bd_lhs <- batching_dims[[1L]]
  bd_rhs <- batching_dims[[2L]]
  # contracting dimensions
  cd_lhs <- contracting_dims[[1L]]
  cd_rhs <- contracting_dims[[2L]]
  # remaining dimensions
  rem_dims <- function(operand, b_dims, c_dims) {
    ii <- c(b_dims, c_dims)
    seq_len(ndims(operand))[if (length(ii)) -ii else TRUE]
  }
  rd_lhs <- rem_dims(lhs, bd_lhs, cd_lhs)
  rd_rhs <- rem_dims(rhs, bd_rhs, cd_rhs)

  # output dimensions
  bd_out <- seq_along(bd_lhs)
  d_lhs_out <- seq_along(rd_lhs) +
    if (length(bd_out)) bd_out[length(bd_out)] else 0L
  d_rhs_out <- seq_along(rd_rhs) +
    if (length(d_lhs_out)) d_lhs_out[length(d_lhs_out)] else 0L

  conv_perm <- function(x) {
    ids_new <- integer(length(x))
    for (i in seq_along(x)) {
      ids_new[x[i]] <- i
    }
    ids_new
  }

  cd_lhs2 <- cd_lhs[order(cd_rhs)]
  perm_lhs <- conv_perm(c(bd_lhs, rd_lhs, cd_lhs2))

  cd_rhs2 <- cd_rhs[order(cd_lhs)]
  perm_rhs <- conv_perm(c(bd_rhs, rd_rhs, cd_rhs2))

  list(
    if (.required[[1L]]) {
      grad_lhs <- nvl_dot_general(
        grad,
        rhs,
        contracting_dims = list(d_rhs_out, rd_rhs),
        batching_dims = list(bd_out, bd_rhs)
      )
      nvl_transpose(grad_lhs, perm_lhs)
    },
    if (.required[[2L]]) {
      grad_rhs <- nvl_dot_general(
        grad,
        lhs,
        contracting_dims = list(d_lhs_out, rd_lhs),
        batching_dims = list(bd_out, bd_lhs)
      )
      nvl_transpose(grad_rhs, perm_rhs)
    }
  )
}

p_transpose[["backward"]] <- function(inputs, outputs, grads, permutation, .required) {
  grad <- grads[[1L]]
  inv <- integer(length(permutation))
  for (i in seq_along(permutation)) {
    inv[permutation[[i]]] <- i
  }
  list(
    if (.required[[1L]]) nvl_transpose(grad, inv)
  )
}

p_reshape[["backward"]] <- function(inputs, outputs, grads, shape, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_reshape(grad, shape(operand))
  )
}

p_reduce_sum[["backward"]] <- function(inputs, outputs, grads, dims, drop, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) {
      bdims <- if (drop) {
        without(seq_along(shape(operand)), dims)
      } else {
        seq_along(shape(grad))
      }
      nvl_broadcast_in_dim(grad, shape(operand), bdims)
    }
  )
}

p_reduce_max[["backward"]] <- function(inputs, outputs, grads, dims, drop, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]

  list(
    if (.required[[1L]]) {
      bdims <- if (drop) {
        without(seq_along(shape(operand)), dims)
      } else {
        seq_along(shape(grad))
      }

      y <- outputs[[1L]]
      y_bc <- nvl_broadcast_in_dim(y, shape(operand), bdims)

      grad_bc <- nvl_broadcast_in_dim(grad, shape(operand), bdims)
      mask <- nvl_eq(operand, y_bc)
      mask_f <- nvl_convert(mask, dtype = dtype(grad_bc))

      count <- nvl_reduce_sum(mask_f, dims = dims, drop = drop)
      count_bc <- nvl_broadcast_in_dim(count, shape(operand), bdims)

      nvl_div(nvl_mul(grad_bc, mask_f), count_bc)
    }
  )
}

p_broadcast_in_dim[["backward"]] <- function(inputs, outputs, grads, shape_out, broadcast_dimensions, .required) {
  operand <- inputs[[1L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]

  list(
    if (.required[[1L]]) {
      # Sum grad over the axes that were introduced by broadcasting
      new_dims <- setdiff(seq_len(ndims(y)), broadcast_dimensions)
      expand_dims <- broadcast_dimensions[(shape(y)[broadcast_dimensions] != 1L) & (shape(operand) == 1L)]
      reduce_dims <- c(new_dims, expand_dims)

      x <- if (length(reduce_dims)) nvl_reduce_sum(grad, dims = reduce_dims, drop = FALSE) else grad

      # Drop the singular added dimensions
      if (length(new_dims)) {
        reshape_dims <- shape(x)
        reshape_dims <- reshape_dims[-new_dims]
        x <- nvl_reshape(x, reshape_dims)
      }

      # If broadcast_dimensions are not in increasing order, reorder the
      # remaining axes back to the original operand axis order.
      if (is.unsorted(broadcast_dimensions)) {
        x <- nvl_transpose(x, order(broadcast_dimensions))
      }
      x
    }
  )
}

# control flow backward ---------------------------------------------------------

p_select[["backward"]] <- function(inputs, outputs, grads, .required) {
  pred <- inputs[[1L]]
  true_value <- inputs[[2L]]
  grad <- grads[[1L]]
  zero <- nv_full(0L, dtype = dtype(true_value), shape = shape(true_value))

  list(
    if (.required[[1L]]) cli_abort("Predicate cannot be differentiated"),
    if (.required[[2L]]) nvl_select(pred, grad, zero),
    if (.required[[3L]]) nvl_select(nvl_not(pred), grad, zero)
  )
}

p_if[["backward"]] <- function(inputs, outputs, grads, true, false, node_map, .required) {
  cli_abort("Not yet implemented")
}

# convert backward -----------------

p_convert[["backward"]] <- function(inputs, outputs, grads, dtype, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_convert(grad, dtype(operand))
  )
}

# for comparison primitives --------------------------
# they are actually not differentiable, but instead of throwing, we
# return zeros for everything.

zero_grads <- function(inputs, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  req_lhs <- .required[[1L]]
  req_rhs <- .required[[2L]]

  zero_like <- function(x) {
    nv_full(0L, dtype = dtype(x), shape = shape(x))
  }

  list(
    if (req_lhs) zero_like(lhs),
    if (req_rhs) zero_like(rhs)
  )
}

p_eq[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}

p_ne[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}

p_gt[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}

p_ge[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}

p_lt[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}

p_le[["backward"]] <- function(inputs, outputs, grads, .required) {
  zero_grads(inputs, .required)
}
