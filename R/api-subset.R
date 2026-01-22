#' @include api.R
#' @include primitives.R

# Subset Specification Classes =================================================
#
# These S3 classes represent the different types of subset specifications:
# - SubsetFull: select all elements (missing argument or `:`)
# - SubsetRange: select a contiguous range (e.g., `2:5`)
# - SubsetIndices: select one or more elements by index (e.g., `3`, `list(1, 3, 5)`)

#' @noRd
SubsetFull <- function(size) {
  structure(list(start = 1L, size = size), class = "SubsetFull")
}

#' @noRd
SubsetRange <- function(start, end) {
  structure(list(start = start, size = end - start + 1L), class = "SubsetRange")
}

#' @noRd
SubsetIndices <- function(index, drop) {
  # index can be: integer (static), or tensor (dynamic)
  static <- is.numeric(index)
  if (static) {
    size <- length(index)
  } else {
    nd <- ndims_abstract(index)
    size <- if (nd == 0L) 1L else shape_abstract(index)[1L]
  }
  structure(list(index = index, size = size, drop = drop, static = static), class = "SubsetIndices")
}

# Type checking helpers
is_subset_full <- function(x) inherits(x, "SubsetFull")
is_subset_range <- function(x) inherits(x, "SubsetRange")
is_subset_indices <- function(x) inherits(x, "SubsetIndices")

#' Get start index for a subset specification
#' @param s A SubsetSpec object
#' @param dtype The dtype to use for the tensor (default "i64")
#' @return A scalar tensor with the start index
#' @noRd
subset_start <- function(s, dtype = "i64") {
  if (is_subset_indices(s)) s$index else nv_scalar(s$start, dtype = dtype)
}

#' Convert subset specs to gather parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @param rank Integer, the rank of the operand tensor
#' @return A list with all parameters needed for nvl_gather:
#'   - start_indices: tensor of start indices
#'   - slice_sizes: integer vector
#'   - offset_dims: integer vector
#'   - collapsed_slice_dims: integer vector
#'   - start_index_map: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - gather_dims: integer vector (for post-processing reshape)
#'   - gather_shape: integer vector (for post-processing reshape)
#' @noRd
subset_specs_to_gather <- function(subsets, rank) {
  # Identify gather dimensions (SubsetIndices with multiple elements)
  gather_dims <- which(vapply(subsets, function(s) {
    is_subset_indices(s) && s$size > 1L
  }, logical(1L)))

  is_gather <- length(gather_dims) > 0L

  if (is_gather) {
    # For gather, slice_sizes is 1 for gather dims, normal for others
    slice_sizes <- vapply(seq_len(rank), function(i) {
      if (i %in% gather_dims) 1L else subsets[[i]]$size
    }, integer(1L))

    # Collapsed dims: gather dims plus any other dims with drop = TRUE
    collapsed_slice_dims <- c(
      gather_dims,
      which(vapply(seq_len(rank), function(i) {
        !(i %in% gather_dims) && isTRUE(subsets[[i]]$drop)
      }, logical(1L)))
    )
    collapsed_slice_dims <- sort(unique(collapsed_slice_dims))

    gather_shape <- vapply(gather_dims, function(i) subsets[[i]]$size, integer(1L))

    # Build start_indices using expand_grid for all combinations
    start_indices_per_dim <- lapply(seq_len(rank), function(i) {
      s <- subsets[[i]]
      if (i %in% gather_dims) {
        s$index
      } else {
        nv_convert(subset_start(s), dtype = "i64")
      }
    })
    start_indices <- do.call(nv_expand_grid, start_indices_per_dim)
    start_indices <- nv_convert(start_indices, dtype = "i32")
  } else {
    # Simple slice case
    slice_sizes <- vapply(subsets, \(s) s$size, integer(1L))
    collapsed_slice_dims <- which(vapply(subsets, \(s) isTRUE(s$drop), logical(1L)))
    gather_shape <- integer(0L)

    # Build start_indices by concatenating scalar starts
    start_parts <- lapply(subsets, function(s) {
      nv_convert(subset_start(s), dtype = "i32")
    })
    if (rank == 1L) {
      start_indices <- nv_reshape(start_parts[[1L]], shape = c(1L, 1L))
    } else {
      reshaped <- lapply(start_parts, \(t) nv_reshape(t, shape = c(1L, 1L)))
      start_indices <- do.call(nv_concatenate, c(reshaped, list(dimension = 2L)))
    }
  }

  n_remaining <- rank - length(collapsed_slice_dims)
  offset_dims <- seq_len(n_remaining) + 1L

  list(
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    start_index_map = seq_len(rank),
    index_vector_dim = 2L,
    indices_are_sorted = !is_gather,
    unique_indices = !is_gather,
    gather_dims = gather_dims,
    gather_shape = gather_shape
  )
}

#' Convert subset specs to scatter parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @param rank Integer, the rank of the input tensor
#' @param update_shape Integer vector, the shape of the update tensor
#' @return A list with all parameters needed for nvl_scatter:
#'   - scatter_indices: tensor of scatter indices
#'   - update_window_dims: integer vector
#'   - inserted_window_dims: integer vector
#'   - scatter_dims_to_operand_dims: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - slice_sizes: integer vector (for shape validation)
#' @noRd
subset_specs_to_scatter <- function(subsets, rank, update_shape) {
  # Check for gather-type indices (not supported in scatter)
  has_multi_indices <- any(vapply(subsets, function(s) {
    is_subset_indices(s) && s$size > 1L
  }, logical(1L)))
  if (has_multi_indices) {
    cli_abort("Gather indices (list() with multiple elements or 1D tensor indices) are not supported in subset assignment")
  }

  slice_sizes <- vapply(subsets, \(s) s$size, integer(1L))

  # Build scatter_indices by concatenating start positions

  start_tensors <- lapply(subsets, function(s) {
    nv_convert(subset_start(s), dtype = "i64")
  })
  scatter_indices <- do.call(nv_concatenate, c(start_tensors, list(dimension = 1L)))

  # Determine update_window_dims and inserted_window_dims based on update_shape

  # update_window_dims: dimensions of the update that correspond to window dimensions
  # inserted_window_dims: operand dimensions not present in the update (size-1 dims that were "inserted")
  n_update_dims <- length(update_shape)

  if (n_update_dims == 0L) {
    # Scalar update: all operand dims are inserted
    update_window_dims <- integer(0L)
    inserted_window_dims <- seq_len(rank)
  } else if (identical(as.integer(update_shape), slice_sizes)) {
    # Update shape matches slice shape exactly
    update_window_dims <- seq_len(rank)
    inserted_window_dims <- integer(0L)
  } else {
    # Update has fewer dims than slice - need to figure out which dims were dropped
    # The dropped dims should be size-1 in slice_sizes
    size_one_dims <- which(slice_sizes == 1L)
    if (length(size_one_dims) != rank - n_update_dims) {
      cli_abort("Update shape {shape2string(update_shape)} is incompatible with slice shape {shape2string(slice_sizes)}")
    }
    inserted_window_dims <- size_one_dims
    update_window_dims <- setdiff(seq_len(rank), size_one_dims)

    # Verify the remaining dims match
    expected_update_shape <- slice_sizes[update_window_dims]
    if (!identical(as.integer(update_shape), expected_update_shape)) {
      cli_abort("Update shape {shape2string(update_shape)} does not match expected shape {shape2string(expected_update_shape)}")
    }
  }

  list(
    scatter_indices = scatter_indices,
    update_window_dims = update_window_dims,
    inserted_window_dims = inserted_window_dims,
    scatter_dims_to_operand_dims = seq_len(rank),
    index_vector_dim = 1L,
    indices_are_sorted = TRUE,
    unique_indices = TRUE,
    slice_sizes = slice_sizes
  )
}

# Helper functions for subset operations ======================================

#' Parse subset specifications and fill unspecified dimensions
#' @param quos List of quosures (from enquos)
#' @param operand_shape Shape of the operand tensor
#' @return List of SubsetSpec objects
#' @noRd
parse_subset_specs <- function(quos, operand_shape) {
  rank <- length(operand_shape)

  if (length(quos) > rank) {
    cli_abort("Too many subset specifications: got {length(quos)}, expected at most {rank}")
  }

  subsets <- lapply(seq_along(quos), function(i) {
    parse_subset_spec(quos[[i]], operand_shape[i])
  })

  # Trailing subsets don't need to be specified, so we fill them with full selections
  if (length(subsets) < rank) {
    for (i in seq(length(subsets) + 1L, rank)) {
      subsets[[i]] <- SubsetFull(operand_shape[i])
    }
  }

  # Convert R indices to tensors with appropriate dtype
  # First, find if any tensor indices exist and get their dtype
  tensor_dtype <- NULL
  for (s in subsets) {
    if (is_subset_indices(s) && is_tensorish(s$index)) {
      tensor_dtype <- dtype_abstract(s$index)
      break
    }
  }
  index_dtype <- tensor_dtype %||% as_dtype("i64")

  # Convert R integer indices to tensors and ensure consistent dtype
  for (i in seq_along(subsets)) {
    s <- subsets[[i]]
    if (is_subset_indices(s)) {
      idx <- s$index
      if (is.numeric(idx)) {
        # Convert R integer(s) to tensor
        if (length(idx) == 1L) {
          subsets[[i]]$index <- nv_scalar(idx, dtype = index_dtype)
        } else {
          subsets[[i]]$index <- nv_tensor(idx, dtype = index_dtype)
        }
      } else if (is_tensorish(idx) && dtype_abstract(idx) != index_dtype) {
        # Convert tensor to consistent dtype
        subsets[[i]]$index <- nv_convert(idx, dtype = index_dtype)
      }
    }
  }

  subsets
}

#' Parse a single subset specification
#' @param quo Quosure to parse
#' @param dim_size Size of the dimension being indexed
#' @return A SubsetSpec object (SubsetFull, SubsetRange, or SubsetIndices)
#' @noRd
parse_subset_spec <- function(quo, dim_size) {
  is_integerish <- function(x) test_integerish(x, len = 1L, any.missing = FALSE)

  # Missing argument - select all
  if (rlang::quo_is_missing(quo)) {
    return(SubsetFull(dim_size))
  }

  e <- rlang::quo_get_expr(quo)

  # Range (a:b) - must check before evaluating since `:` has special meaning
  if (is.call(e) && identical(e[[1]], quote(`:`))) {
    env <- rlang::quo_get_env(quo)
    start <- rlang::eval_tidy(rlang::new_quosure(e[[2]], env))
    end <- rlang::eval_tidy(rlang::new_quosure(e[[3]], env))
    if (is_integerish(start) && is_integerish(end)) {
      start <- as.integer(start)
      end <- as.integer(end)
      if (start > end) {
        cli_abort("Range start ({start}) must be less than or equal to end ({end})")
      }
      return(SubsetRange(start, end))
    }
    cli_abort("Ranges must be static")
  }

  # Evaluate the quosure
  e <- rlang::eval_tidy(quo)

  if (identical(e, `:`)) {
    return(SubsetFull(dim_size))
  }

  # Single integer - drops dimension
  if (is_integerish(e)) {
    return(SubsetIndices(as.integer(e), drop = TRUE))
  }

  # R vectors of length > 1 - not allowed (use list() instead)
  if (is.numeric(e) && length(e) > 1L) {
    cli_abort(
      "Vectors of length > 1 are not allowed as subset indices. Use list() to select multiple elements, e.g. x[list(1, 3), ] instead of x[c(1, 3), ]"
    )
  }

  # list() - preserves dimensions (keep as R integer vector, convert to tensor later)
  if (is.list(e) && !is.object(e)) {
    if (length(e) == 0L) {
      cli_abort("Empty list() indices are not allowed")
    }
    if (!all(vapply(e, is_integerish, logical(1L)))) {
      cli_abort("All list() elements must be scalar integers")
    }

    indices <- vapply(e, as.integer, integer(1L))
    return(SubsetIndices(indices, drop = FALSE))
  }

  # AnvilRange (dynamic range) - not supported
  if (inherits(e, "AnvilRange")) {
    cli_abort("Ranges must be static")
  }

  # Tensor indices
  if (is_tensorish(e) && !is.atomic(e)) {
    if (!is_anvil_tensor(e)) {
      cli_abort("Expected AnvilTensor for dynamic index")
    }
    dt <- dtype_abstract(e)
    if (!(inherits(dt, "IntegerType") || inherits(dt, "UnsignedType"))) {
      cli_abort("Dynamic indices must be integers, but got {.cls {class(dt)[1]}}")
    }
    nd <- ndims_abstract(e)
    if (nd > 1L) {
      cli_abort("Dynamic indices must be at most 1D, but got {nd}D tensor")
    }
    # Scalar tensor drops dimension, 1D tensor preserves
    return(SubsetIndices(e, drop = nd == 0L))
  }

  cli_abort("Invalid subset expression")
}

#' @title Subset a Tensor
#' @description
#' Extracts a subset from a tensor using gather semantics.
#' Supports both static and dynamic indexing.
#'
#' Dimension dropping behavior:
#' - R literal indices (e.g., `x[1, ]`) automatically drop that dimension
#' - Scalar tensor indices (0D tensors) also drop the dimension
#' - To preserve a dimension when selecting a single element, use `list()`: `x[list(1), ]`
#' - To select multiple non-contiguous elements, use `list()`: `x[list(1, 3, 5), ]`
#' - Ranges (e.g., `x[1:3, ]`) never drop dimensions
#' - 1D tensor indices never drop dimensions
#'
#' @param x ([`tensorish`])\cr
#'   Input tensor to subset.
#' @param ... Subset specifications. Can be:
#'   - Ranges (e.g., `2:5`) for contiguous slices. Must be static.
#'   - Single integers (e.g., `3`) for single elements (drops dimension)
#'   - `list(i)` for single element without dropping dimension
#'   - `list(i, j, ...)` for multiple non-contiguous elements
#'   - Scalar tensors for dynamic indexing (does not drop dimension)
#'   - Missing (empty or `:`) to select all elements in that dimension
#' @return [`tensorish`]
#' @export
nv_subset <- function(x, ...) {
  if (!is_tensorish(x)) {
    cli_abort(c(
      "Argument x must be tensorish",
      "x" = "Got {.cls {class(x)[1]}}"
    ))
  }
  operand_shape <- shape_abstract(x)
  quos <- rlang::enquos(...)
  rank <- length(operand_shape)

  subsets <- parse_subset_specs(quos, operand_shape)
  params <- subset_specs_to_gather(subsets, rank)

  out <- nvl_gather(
    operand = x,
    start_indices = params$start_indices,
    slice_sizes = params$slice_sizes,
    offset_dims = params$offset_dims,
    collapsed_slice_dims = params$collapsed_slice_dims,
    operand_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = params$start_index_map,
    index_vector_dim = params$index_vector_dim,
    indices_are_sorted = params$indices_are_sorted,
    unique_indices = params$unique_indices
  )

  if (length(params$gather_dims) > 0L) {
    # Reshape gather output to proper multi-dimensional shape
    current_shape <- shape_abstract(out)
    if (length(params$gather_dims) > 1L && length(current_shape) > 0L) {
      new_shape <- c(params$gather_shape, current_shape[-1L])
      out <- nv_reshape(out, new_shape)
    }
  } else {
    # Remove the batch dimension (size 1, at position 1)
    result_shape <- shape(out)
    if (length(result_shape) > 0L && result_shape[1L] == 1L) {
      out <- nv_reshape(out, shape = result_shape[-1L])
    }
  }

  out
}

#' @title Update Subset
#' @description
#' Updates elements of a tensor.
#' This has copy-on-write semantics just like for standard R arrays.
#' @param x ([`tensorish`])\cr
#'   Input tensor to update.
#' @param ... Subset specifications. Can be:
#'   - Ranges (e.g., `2:5`) for contiguous slices. Must be static.
#'   - Single integers (e.g., `3`) for single elements
#'   - Missing (selects all) for full dimension
#' @param value ([`tensorish`])\cr
#'   Values to write. Shape must match the subset shape.
#' @return [`tensorish`]
#' @export
#' @examples
#' \dontrun{
#' # Update contiguous slice
#' x <- nv_tensor(1:10)
#' x <- nv_subset_assign(x, 2:4, value = nv_tensor(c(20, 30, 40)))
#' }
nv_subset_assign <- function(x, ..., value) {
  if (!is_tensorish(x)) {
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  if (!is_tensorish(value, literal = TRUE)) {
    cli_abort("Expected tensorish value, but got {.cls {class(value)[1]}}")
  }

  lhs_shape <- shape_abstract(x)
  quos <- rlang::enquos(...)
  rank <- length(lhs_shape)

  subsets <- parse_subset_specs(quos, lhs_shape)

  # Get slice_sizes first to handle scalar broadcast
  slice_sizes <- vapply(subsets, \(s) s$size, integer(1L))

  value_shape <- shape_abstract(value)
  if (!ndims_abstract(value)) {
    value <- nv_broadcast_to(value, slice_sizes)
    value_shape <- slice_sizes
  }

  params <- subset_specs_to_scatter(subsets, rank, value_shape)

  if (!identical(params$slice_sizes, as.integer(value_shape))) {
    # fmt: skip
    cli_abort("Subset shape {shape2string(params$slice_sizes)} does not match value shape {shape2string(value_shape)}") # nolint
  }

  if (dtype(x) != dtype(value)) {
    if (!promotable_to(dtype(value), dtype(x))) {
      cli_abort(
        "Value type {dtype2string(dtype(value))} is not promotable to left-hand side type {dtype2string(dtype(x))}"
      )
    }
    value <- nv_convert(value, dtype = dtype(x))
  }

  nvl_scatter(
    input = x,
    scatter_indices = params$scatter_indices,
    update = value,
    update_window_dims = params$update_window_dims,
    inserted_window_dims = params$inserted_window_dims,
    input_batching_dims = integer(),
    scatter_indices_batching_dims = integer(),
    scatter_dims_to_operand_dims = params$scatter_dims_to_operand_dims,
    index_vector_dim = params$index_vector_dim,
    indices_are_sorted = params$indices_are_sorted,
    unique_indices = params$unique_indices,
    update_computation = function(old, new) new
  )
}
