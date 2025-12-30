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
      one <- ones_like(lhs)
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

p_sqrt[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sqrt(x) = 1 / (2 * sqrt(x))
    if (.required[[1L]]) {
      half <- nvl_fill(0.5, dtype = dtype(y), shape = shape(y))
      nvl_div(nvl_mul(grad, half), y)
    }
  )
}

p_rsqrt[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx 1/sqrt(x) = -0.5 * x^(-3/2) = -0.5 * rsqrt(x)^3
    if (.required[[1L]]) {
      neg_half <- nvl_fill(-0.5, dtype = dtype(y), shape = shape(y))
      nvl_mul(nvl_mul(grad, neg_half), nvl_mul(y, nvl_mul(y, y)))
    }
  )
}

p_tanh[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx tanh(x) = 1 - tanh(x)^2
    if (.required[[1L]]) {
      one <- nvl_fill(1, dtype = dtype(y), shape = shape(y))
      nvl_mul(grad, nvl_sub(one, nvl_mul(y, y)))
    }
  )
}

p_tan[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx tan(x) = 1 + tan(x)^2
    if (.required[[1L]]) {
      one <- nvl_fill(1, dtype = dtype(y), shape = shape(y))
      nvl_mul(grad, nvl_add(one, nvl_mul(y, y)))
    }
  )
}

p_sine[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sin(x) = cos(x)
    if (.required[[1L]]) nvl_mul(grad, nvl_cosine(operand))
  )
}

p_cosine[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cos(x) = -sin(x)
    if (.required[[1L]]) nvl_mul(grad, nvl_neg(nvl_sine(operand)))
  )
}

p_abs[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx |x| = sign(x)
    if (.required[[1L]]) nvl_mul(grad, nvl_sign(operand))
  )
}

p_max[["backward"]] <- p_min[["backward"]] <- function(inputs, outputs, grads, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]

  if (.required[[1L]] || .required[[2L]]) {
    y <- outputs[[1L]]
    mask_lhs <- nvl_convert(nvl_eq(lhs, y), dtype = dtype(grad))
    mask_rhs <- nvl_convert(nvl_eq(rhs, y), dtype = dtype(grad))
    count <- nvl_add(mask_lhs, mask_rhs)
  }

  list(
    if (.required[[1L]]) nvl_div(nvl_mul(grad, mask_lhs), count),
    if (.required[[2L]]) nvl_div(nvl_mul(grad, mask_rhs), count)
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

p_reduce_max[["backward"]] <- p_reduce_min[["backward"]] <- function(inputs, outputs, grads, dims, drop, .required) {
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
  zero <- zeros_like(true_value)

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

p_convert[["backward"]] <- function(inputs, outputs, grads, dtype, ambiguous, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  # the ambiguity is determined by the input, not the `ambiguous` parameter
  list(
    if (.required[[1L]]) nvl_convert(grad, dtype(operand), inputs[[1L]]@aval@ambiguous)
  )
}

# for comparison primitives --------------------------
# they are actually not differentiable, but instead of throwing, we
# return zeros for everything.

backward_zero_bin <- function(inputs, outputs, grads, .required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  req_lhs <- .required[[1L]]
  req_rhs <- .required[[2L]]
  list(
    if (req_lhs) zeros_like(lhs),
    if (req_rhs) zeros_like(rhs)
  )
}

p_eq[["backward"]] <- backward_zero_bin
p_ne[["backward"]] <- backward_zero_bin
p_gt[["backward"]] <- backward_zero_bin
p_ge[["backward"]] <- backward_zero_bin
p_lt[["backward"]] <- backward_zero_bin
p_le[["backward"]] <- backward_zero_bin

# zero-grads (ignores the non-differentiable points)

backward_zero_uni <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  list(
    if (.required[[1L]]) zeros_like(operand)
  )
}


p_floor[["backward"]] <- backward_zero_uni
p_ceil[["backward"]] <- backward_zero_uni
p_sign[["backward"]] <- backward_zero_uni
p_round[["backward"]] <- function(inputs, outputs, grads, method, .required) {
  backward_zero_uni(inputs, outputs, grads, .required)
}

# New primitives backward rules ------------------------------------------------

p_cbrt[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cbrt(x) = 1 / (3 * cbrt(x)^2)
    if (.required[[1L]]) {
      third <- nvl_fill(1 / 3, dtype = dtype(y), shape = shape(y))
      nvl_div(nvl_mul(grad, third), nvl_mul(y, y))
    }
  )
}

p_expm1[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx (exp(x) - 1) = exp(x)
    if (.required[[1L]]) nvl_mul(grad, nvl_exp(operand))
  )
}

p_log1p[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx log(1 + x) = 1 / (1 + x)
    if (.required[[1L]]) {
      one <- nvl_fill(1, dtype = dtype(operand), shape = shape(operand))
      nvl_div(grad, nvl_add(one, operand))
    }
  )
}

p_logistic[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    if (.required[[1L]]) {
      one <- nvl_fill(1, dtype = dtype(y), shape = shape(y))
      nvl_mul(grad, nvl_mul(y, nvl_sub(one, y)))
    }
  )
}

p_clamp[["backward"]] <- function(inputs, outputs, grads, .required) {
  min_val <- inputs[[1L]]
  operand <- inputs[[2L]]
  max_val <- inputs[[3L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]

  if (ndims(min_val) == 0L) {
    min_val <- nvl_broadcast_in_dim(min_val, shape(operand), integer())
  }
  if (ndims(max_val) == 0L) {
    max_val <- nvl_broadcast_in_dim(max_val, shape(operand), integer())
  }

  # Gradient flows through where operand equals output (not clamped)
  mask_operand <- nvl_convert(nvl_eq(operand, y), dtype = dtype(grad))
  mask_min <- nvl_convert(nvl_eq(min_val, y), dtype = dtype(grad))
  mask_max <- nvl_convert(nvl_eq(max_val, y), dtype = dtype(grad))

  list(
    if (.required[[1L]]) nvl_mul(grad, mask_min),
    if (.required[[2L]]) nvl_mul(grad, mask_operand),
    if (.required[[3L]]) nvl_mul(grad, mask_max)
  )
}

p_reverse[["backward"]] <- function(inputs, outputs, grads, dimensions, .required) {
  grad <- grads[[1L]]
  list(
    # Reverse the gradient along the same dimensions
    if (.required[[1L]]) nvl_reverse(grad, dimensions)
  )
}

p_pad[["backward"]] <- function(
  inputs,
  outputs,
  grads,
  edge_padding_low,
  edge_padding_high,
  interior_padding,
  .required
) {
  operand <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) {
      operand_shape <- shape(operand)
      # For interior_padding > 0, we need to select every (interior_padding + 1)th element
      # using slice with strides
      strides <- interior_padding + 1L
      # Start indices in 1-based (anvil convention)
      start_indices <- edge_padding_low + 1L
      # nvl_slice converts start_indices to 0-based (start_attr = start_indices - 1)
      # and keeps limit_indices as-is (limit_attr = limit_indices)
      # The positions we want (0-based): edge_padding_low, edge_padding_low + stride, ..., edge_padding_low + (n-1)*stride
      # The last position (0-based): edge_padding_low + (operand_shape - 1) * strides
      # limit (exclusive, 0-based): last_position + 1
      limit_indices <- edge_padding_low + (operand_shape - 1L) * strides + 1L
      nvl_slice(grad, start_indices, limit_indices, strides)
    },
    # padding_value gradient is not typically needed
    if (.required[[2L]]) {
      cli_abort("Gradient for padding_value not implemented")
    }
  )
}
