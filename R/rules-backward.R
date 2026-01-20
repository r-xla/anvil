# All the backward rules are only operating on GraphBoxes

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
    if (.required[[2L]]) nvl_negate(grad)
  )
}

p_negate[["backward"]] <- function(inputs, outputs, grads, .required) {
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_negate(grad)
  )
}

p_div[["backward"]] <- function(inputs, outputs, grads, .required) {
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) nvl_div(grad, rhs),
    if (.required[[2L]]) nvl_div(nvl_mul(grad, nvl_negate(y)), rhs)
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
    if (.required[[1L]]) nvl_mul(grad, nvl_negate(nvl_sine(operand)))
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
    if (.required[[1L]]) nvl_convert(grad, dtype(operand), ambiguous_abstract(operand))
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

p_cbrt[["backward"]] <- function(inputs, outputs, grads, .required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cbrt(x) = 1 / (3 * cbrt(x)^2)
    if (.required[[1L]]) {
      three <- nvl_fill(3, dtype = dtype(y), shape = shape(y))
      nvl_div(grad, nv_mul(nvl_mul(y, y), three))
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

  # because stablehlo.clamp broadcasts scalars, we need to handle this here before the eq call
  # this is an inconsistency in stablehlo, as it broadcasts scalars in clamp, but not in eq
  # (and most other functions)
  if (ndims(min_val) == 0L) {
    min_val <- nvl_broadcast_in_dim(min_val, shape(operand), integer())
  }
  if (ndims(max_val) == 0L) {
    max_val <- nvl_broadcast_in_dim(max_val, shape(operand), integer())
  }

  # the points where operand is equal to min_val or max_val are non differentiable,
  # so we just implement it like torch, which uses 1 for the gradient there.
  mask_operand <- nvl_convert(nvl_eq(operand, y), dtype = dtype(grad))

  list(
    if (.required[[1L]]) cli_abort("Gradient for min_val not implemented"),
    if (.required[[2L]]) nvl_mul(grad, mask_operand),
    if (.required[[3L]]) cli_abort("Gradient for max_val not implemented")
  )
}

p_reverse[["backward"]] <- function(inputs, outputs, grads, dims, .required) {
  grad <- grads[[1L]]
  list(
    # Reverse the gradient along the same dimensions
    if (.required[[1L]]) nvl_reverse(grad, dims)
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
  grad <- grads[[1L]]
  list(
    if (.required[[1L]]) {
      # select the non-padded elements
      out_shape <- shape(outputs[[1L]])
      strides <- interior_padding + 1L
      start_indices <- edge_padding_low + 1L
      limit_indices <- out_shape - edge_padding_high
      nvl_static_slice(grad, start_indices, limit_indices, strides)
    },
    if (.required[[2L]]) {
      cli_abort("Gradient for padding_value not implemented")
    }
  )
}

p_dynamic_slice[["backward"]] <- function(inputs, outputs, grads, slice_sizes, .required) {
  operand <- inputs[[1L]]
  start_indices <- inputs[-1L]
  grad <- grads[[1L]]

  result <- vector("list", length(inputs))

  if (.required[[1L]]) {
    # Gradient for operand: create zeros and update the slice region with the gradient
    zero <- zeros_like(operand)
    result[[1L]] <- rlang::exec(nvl_dynamic_update_slice, zero, grad, !!!start_indices)
  }

  # start_indices are not differentiable - keep as NULL (already initialized)
  # Note: result[i] <- list(NULL) is used to set to NULL without removing the element

  result
}

p_dynamic_update_slice[["backward"]] <- function(inputs, outputs, grads, .required) {
  operand <- inputs[[1L]]
  update <- inputs[[2L]]
  start_indices <- inputs[-(1:2)]
  grad <- grads[[1L]]

  result <- vector("list", length(inputs))

  if (.required[[1L]]) {
    # Gradient for operand: set the slice region to zero
    zero_update <- zeros_like(update)
    result[[1L]] <- rlang::exec(nvl_dynamic_update_slice, grad, zero_update, !!!start_indices)
  }

  if (.required[[2L]]) {
    # Gradient for update: extract the slice from the gradient
    slice_sizes <- shape(update)
    result[[2L]] <- rlang::exec(nvl_dynamic_slice, grad, !!!start_indices, slice_sizes = slice_sizes)
  }

  # start_indices are not differentiable - keep as NULL (already initialized)

  result
}

p_gather[["backward"]] <- function(inputs, outputs, grads, slice_sizes, offset_dims, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, start_index_map, index_vector_dim, indices_are_sorted, unique_indices, .required) {
  operand <- inputs[[1L]]
  start_indices <- inputs[[2L]]
  grad <- grads[[1L]]

  # Gather's backward is scatter_add: we scatter the output gradient back to the

  # positions in the operand where values were gathered from.
  # Using addition handles the case where multiple gather positions read from the same
  # source location - the gradients accumulate correctly.
  #
  # CLAMPING BEHAVIOR:
  # StableHLO's gather clamps out-of-bounds indices to valid range.
  # In the backward pass, we need to clamp indices the same way, so gradients
  # flow back to the same (clamped) positions that provided values in forward.
  #
  # The scatter parameters are derived from the gather parameters:
  # - update_window_dims = offset_dims (dimensions that came from slicing)
  # - inserted_window_dims = collapsed_slice_dims (dimensions that were collapsed)
  # - scatter_dims_to_operand_dims = start_index_map (how indices map to operand dims)
  # - update_computation = add (to accumulate gradients for overlapping reads)

  list(
    if (.required[[1L]]) {
      # Clamp indices to valid range (same as forward pass clamping behavior)
      scatter_indices <- .gather_clamp_indices(
        start_indices = start_indices,
        operand_shape = shape(operand),
        slice_sizes = slice_sizes,
        start_index_map = start_index_map,
        index_vector_dim = index_vector_dim
      )

      nvl_scatter(
        input = zeros_like(operand),
        scatter_indices = scatter_indices,
        update = grad,
        update_window_dims = offset_dims,
        inserted_window_dims = collapsed_slice_dims,
        input_batching_dims = operand_batching_dims,
        scatter_indices_batching_dims = start_indices_batching_dims,
        scatter_dims_to_operand_dims = start_index_map,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = unique_indices,
        # Use addition to accumulate gradients when multiple gather positions
        # read from the same source location
        update_computation = function(old, new) nvl_add(old, new)
      )
    },
    if (.required[[2L]]) zeros_like(start_indices)
  )
}

# Helper: Clamp indices to valid range
.gather_clamp_indices <- function(
  start_indices,
  operand_shape,
  slice_sizes,
  start_index_map,
  index_vector_dim
) {
  indices_shape <- shape(start_indices)
  n_index_coords <- length(start_index_map)

  if (n_index_coords == 0L) {
    return(start_indices)
  }

  # Build min and max bounds for each coordinate
  min_bounds <- rep(1L, n_index_coords)
  max_bounds <- integer(n_index_coords)
  for (coord_idx in seq_len(n_index_coords)) {
    operand_dim <- start_index_map[coord_idx]
    operand_size <- operand_shape[operand_dim]
    slice_size_for_dim <- slice_sizes[operand_dim]
    max_bounds[coord_idx] <- max(1L, operand_size - slice_size_for_dim + 1L)
  }

  if (index_vector_dim <= length(indices_shape)) {
    # Explicit index vector dimension - build bounds tensors
    bounds_shape <- rep(1L, length(indices_shape))
    bounds_shape[index_vector_dim] <- n_index_coords

    min_tensor <- nvl_broadcast_in_dim(
      nvl_fill(1L, dtype = dtype(start_indices), shape = integer()),
      indices_shape,
      integer()
    )

    max_tensor_vals <- nvl_reshape(
      nv_convert(nv_tensor(max_bounds, dtype = "i64"), dtype = dtype(start_indices)),
      bounds_shape
    )
    max_tensor <- nvl_broadcast_in_dim(max_tensor_vals, indices_shape, seq_along(indices_shape))

    nvl_clamp(min_tensor, start_indices, max_tensor)
  } else {
    # Implicit index vector (single coordinate)
    min_tensor <- nvl_fill(1L, dtype = dtype(start_indices), shape = indices_shape)
    max_tensor <- nvl_fill(max_bounds[1L], dtype = dtype(start_indices), shape = indices_shape)
    nvl_clamp(min_tensor, start_indices, max_tensor)
  }
}

p_scatter[["backward"]] <- function(
  inputs,
  outputs,
  grads,
  update_window_dims,
  inserted_window_dims,
  input_batching_dims,
  scatter_indices_batching_dims,
  scatter_dims_to_operand_dims,
  index_vector_dim,
  indices_are_sorted,
  unique_indices,
  update_computation_graph,
  .required
) {
  input <- inputs[[1L]]
  scatter_indices <- inputs[[2L]]
  update <- inputs[[3L]]
  grad <- grads[[1L]]

  # Scatter backward:
  # For simple assignment (update_computation = function(old, new) new):
  # - d(output)/d(input) at positions NOT in indices = identity
  # - d(output)/d(input) at positions IN indices = 0 (overwritten values don't contribute)
  # - d(output)/d(scatter_indices) = 0 (indices are not differentiable)
  # - d(output)/d(update) = gradient at positions where updates were written
  #
  # NOTE: This implementation currently only supports unique_indices = TRUE.
  # For non-unique indices (overlapping writes), the backward pass is more complex
# because XLA doesn't guarantee which update "wins" at each position.

  if (!unique_indices) {
    cli_abort(
      c(
        "Scatter backward is only implemented for unique_indices = TRUE.",
        "i" = "With overlapping indices, the forward pass behavior is non-deterministic,",
        "i" = "making gradient computation ill-defined.",
        "i" = "Consider restructuring your code to avoid duplicate scatter indices."
      )
    )
  }

  # Compute slice_sizes for the gather operation (inverse of scatter)
  # This determines the shape of slices gathered from the gradient tensor
  update_shape <- shape(update)
  input_shape <- shape(input)

  # Build slice_sizes: for dimensions in update_window_dims, use update shape;
  # for dimensions in inserted_window_dims, use 1
  slice_sizes <- integer(length(input_shape))
  update_window_pos <- 1L
  for (i in seq_along(input_shape)) {
    if (i %in% inserted_window_dims) {
      slice_sizes[i] <- 1L
    } else if (i %in% input_batching_dims) {
      slice_sizes[i] <- 1L
    } else {
      slice_sizes[i] <- update_shape[update_window_dims[update_window_pos]]
      update_window_pos <- update_window_pos + 1L
    }
  }

  list(
    # Gradient for input: zero out positions that were overwritten
    if (.required[[1L]]) {
      zeros_update <- zeros_like(update)
      nvl_scatter(
        input = grad,
        scatter_indices = scatter_indices,
        update = zeros_update,
        update_window_dims = update_window_dims,
        inserted_window_dims = inserted_window_dims,
        input_batching_dims = input_batching_dims,
        scatter_indices_batching_dims = scatter_indices_batching_dims,
        scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = TRUE,
        update_computation = function(old, new) new
      )
    },
    # Gradient for scatter_indices: not differentiable
    if (.required[[2L]]) zeros_like(scatter_indices),
    # Gradient for update: gather from gradient at update positions
    if (.required[[3L]]) {
      # The gather dimension numbers are derived from scatter dimension numbers:
      # - offset_dims = update_window_dims
      # - collapsed_slice_dims = inserted_window_dims
      # - start_index_map = scatter_dims_to_operand_dims
      # - operand_batching_dims = input_batching_dims
      # - start_indices_batching_dims = scatter_indices_batching_dims
      nvl_gather(
        operand = grad,
        start_indices = scatter_indices,
        slice_sizes = slice_sizes,
        offset_dims = update_window_dims,
        collapsed_slice_dims = inserted_window_dims,
        operand_batching_dims = input_batching_dims,
        start_indices_batching_dims = scatter_indices_batching_dims,
        start_index_map = scatter_dims_to_operand_dims,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = TRUE
      )
    }
  )
}

