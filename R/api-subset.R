#' @include api.R
#' @include primitives.R

#' @title Subset a Tensor
#' @description
#' Extracts a subset from a tensor using gather semantics.
#' Supports both static and dynamic indexing.
#' @param x ([`tensorish`])\cr
#'   Input tensor to subset.
#' @param ... Subset specifications. Can be:
#'   - Ranges (e.g., `2:5`) for contiguous slices. Must be static.
#'   - Single integers (e.g., `3`) for single elements
#'   - Scalar tensors for dynamic indexing
#'   - Missing (empty or `:`) to select all elements in that dimension
#' @param drop (`logical(1)`)\cr
#'   Whether to drop dimensions with size 1 that result from single-index selections.
#' @return [`tensorish`]
#' @export
nv_subset <- function(x, ..., drop = FALSE) {
  assert_flag(drop)
  is_integerish <- function(x) {
    test_integerish(x, len = 1L, any.missing = FALSE)
  }
  get_slice <- function(e, d) {
    # Check for missing argument (empty name from alist)
    if (is.symbol(e) && (identical(as.character(e), "")) || identical(e, quote(`:`))) {
      return(list(static = TRUE, start = 1L, size = d, slice = TRUE, indexed = FALSE))
    }
    if (is_integerish(e)) {
      e <- as.integer(e)
      return(list(static = TRUE, start = e, size = 1L, slice = FALSE, indexed = TRUE))
    }
    if (is.call(e) && identical(e[[1]], quote(`:`))) {
      start <- eval(e[[2]], envir = parent.frame(2L))
      end <- eval(e[[3]], envir = parent.frame(2L))
      # TODO: We could also support Literal Tensors here
      if (is_integerish(start) && is_integerish(end)) {
        start <- as.integer(start)
        end <- as.integer(end)
        return(list(static = TRUE, start = start, size = end - start + 1L, slice = TRUE, indexed = TRUE))
      }
      cli_abort("Ranges must be static")
    }
    # If it's an expression (symbol or call), evaluate it
    if (is.symbol(e) || is.call(e)) {
      e <- eval(e, envir = parent.frame(2L))
    }
    # Check for AnvilRange (result of x:y where x or y is AnvilBox/AnvilTensor)
    if (inherits(e, "AnvilRange")) {
      cli_abort("Ranges must be static")
    }
    if (is_tensorish(e) && (!is.atomic(e))) {
      # only ints are allowed as atomics and they are covered by first case
      return(list(static = FALSE, start = e, size = 1L, slice = FALSE, indexed = TRUE))
    }
    cli_abort("Invalid slice expression")
  }
  if (!is_tensorish(x)) {
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  operand_shape <- shape_abstract(x)
  rank <- length(operand_shape)
  # Use substitute() instead of enexprs() to support forwarding from [ operator
  exprs <- eval(substitute(alist(...)))

  if (length(exprs) > rank) {
    cli_abort("Too many indices: got {length(exprs)}, expected at most {rank}")
  }

  slices <- lapply(seq_along(exprs), function(i) get_slice(exprs[[i]], operand_shape[i]))
  # Fill remaining dimensions with full selections
  if (length(slices) < rank) {
    for (i in (length(slices) + 1L):rank) {
      slices[[i]] <- list(static = TRUE, start = 1L, size = operand_shape[i], slice = TRUE, indexed = FALSE)
    }
  }

  static <- all(vapply(slices, \(x) x$static, logical(1L)))

  # slice_sizes: for each operand dimension
  slice_sizes <- vapply(slices, \(x) x$size, integer(1L))

  # Dimensions to collapse (single index, not range, when drop=TRUE)
  # These are dims where slice=FALSE (single element selected)
  collapsed_slice_dims <- if (drop) {
    which(!vapply(slices, \(x) x$slice, logical(1L)))
  } else {
    integer()
  }

  # For simple contiguous slicing, we use a single gather "batch" entry
  # start_indices shape will be [1, rank] - one row with start position for each dimension
  # index_vector_dim = 2 (1-based) means the index vector is in the second dimension

  # start_index_map: maps each position in the index vector to operand dimensions
  # We map all positions to their corresponding operand dimensions
  start_index_map <- seq_len(rank)

  # offset_dims: which output dimensions correspond to the slice (after the batch dim)
  # The output will have shape [1, slice_dims...] (batch dim first, then slice dims)
  # After removing collapsed dims, offset_dims are positions 2, 3, ... in output (1-based)
  n_remaining <- rank - length(collapsed_slice_dims)
  offset_dims <- seq_len(n_remaining) + 1L  # 1-based, starting at position 2

  if (static) {
    # All indices are static - create a constant start_indices tensor
    start_indices_vec <- vapply(slices, \(x) x$start - 1L, integer(1L))  # 0-based
    start_indices <- nv_tensor(matrix(start_indices_vec, nrow = 1L), dtype = "i32")
    index_vector_dim <- 2L  # 1-based
  } else {
    # Dynamic indices - build start_indices from tensor values
    start_parts <- lapply(slices, function(s) {
      if (is.numeric(s$start)) {
        nv_scalar(as.integer(s$start - 1L), dtype = "i32")  # 0-based
      } else {
        # Dynamic tensor - subtract 1 to convert to 0-based
        nv_convert(s$start - nv_scalar(1L, dtype = dtype(s$start)), dtype = "i32")
      }
    })
    # Stack into [1, rank] tensor
    if (rank == 1L) {
      start_indices <- nv_reshape(start_parts[[1L]], shape = c(1L, 1L))
    } else {
      # Reshape each to [1, 1] and concatenate along dim 2
      reshaped <- lapply(start_parts, \(x) nv_reshape(x, shape = c(1L, 1L)))
      start_indices <- do.call(nv_concatenate, c(reshaped, list(dimension = 2L)))
    }
    index_vector_dim <- 2L  # 1-based
  }

  gather_dim_numbers <- GatherDimensionNumbers(
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    operand_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = start_index_map,
    index_vector_dim = index_vector_dim
  )

  out <- nvl_gather(
    operand = x,
    start_indices = start_indices,
    gather_dimension_numbers = gather_dim_numbers,
    slice_sizes = slice_sizes,
    indices_are_sorted = TRUE
  )

  # Remove the batch dimension (size 1, at position 1) from gather result
  result_shape <- shape(out)
  if (length(result_shape) > 0L && result_shape[1L] == 1L) {
    out <- nv_reshape(out, shape = result_shape[-1L])
  }

  out
}

#' @title Update Subset
#' @description
#' Updates elements of a tensor with new values using scatter.
#' Supports both contiguous slices (e.g., `2:5`) and scattered indices (e.g., `c(1, 3, 6)`).
#' @param x ([`tensorish`])\cr
#'   Input tensor to update.
#' @param ... Subset specifications. Can be:
#'   - Ranges (e.g., `2:5`) for contiguous slices. Must be static.
#'   - Single integers (e.g., `3`) for single elements
#'   - Integer vectors (e.g., `c(1, 3, 6)`) for scattered indices
#'   - Missing (selects all) for full dimension
#' @param value ([`tensorish`])\cr
#'   Values to write. Shape must match the subset shape.
#' @return [`tensorish`]
#' @export
#' @examples
#' \dontrun{
#' # Update contiguous slice
#' x <- nv_tensor(1:10)
#' x <- nv_update_subset(x, 2:4, value = nv_tensor(c(20, 30, 40)))
#'
#' # Update scattered indices
#' x <- nv_tensor(1:10)
#' x <- nv_update_subset(x, c(1, 3, 6), value = nv_tensor(c(10, 30, 60)))
#' }
nv_update_subset <- function(x, ..., value) {
  is_integerish <- function(x) {
    test_integerish(x, len = 1L, any.missing = FALSE)
  }
  is_integerish_vector <- function(x) {
    test_integerish(x, min.len = 1L, any.missing = FALSE)
  }

  get_subset <- function(e, d) {
    # Check for missing argument (empty name from alist) - select all
    if (is.symbol(e) && (identical(as.character(e), "")) || identical(e, quote(`:`))) {
      return(list(type = "full", start = 1L, size = d, indices = NULL))
    }
    # Single integer
    if (is_integerish(e)) {
      e <- as.integer(e)
      return(list(type = "single", start = e, size = 1L, indices = e))
    }
    # Range (a:b)
    if (is.call(e) && identical(e[[1]], quote(`:`))) {
      start <- eval(e[[2]], envir = parent.frame(2L))
      end <- eval(e[[3]], envir = parent.frame(2L))
      if (is_integerish(start) && is_integerish(end)) {
        start <- as.integer(start)
        end <- as.integer(end)
        return(list(type = "range", start = start, size = end - start + 1L, indices = start:end))
      }
      cli_abort("Ranges must be static")
    }
    # Evaluate expression if needed
    if (is.symbol(e) || is.call(e)) {
      e <- eval(e, envir = parent.frame(2L))
    }
    # Integer vector - scattered indices
    if (is_integerish_vector(e)) {
      e <- as.integer(e)
      return(list(type = "scattered", start = NULL, size = length(e), indices = e))
    }
    # Check for AnvilRange or dynamic tensor
    if (inherits(e, "AnvilRange")) {
      cli_abort("Ranges must be static")
    }
    if (is_tensorish(e) && (!is.atomic(e))) {
      cli_abort("Dynamic indices not yet supported")
    }
    cli_abort("Invalid subset expression")
  }

  if (!is_tensorish(x)) {
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  if (!is_tensorish(value)) {
    cli_abort("Expected tensorish value, but got {.cls {class(value)[1]}}")
  }

  shape <- shape_abstract(x)
  exprs <- eval(substitute(alist(...)))

  if (length(exprs) > length(shape)) {
    cli_abort("Too many indices: got {length(exprs)}, expected at most {length(shape)}")
  }

  # Fill in missing dimensions with "full" selection
  if (length(exprs) < length(shape)) {
    exprs <- c(exprs, replicate(length(shape) - length(exprs), quote(`:`), simplify = FALSE))
  }

  # Parse subset specifications
  subsets <- lapply(seq_along(exprs), function(i) get_subset(exprs[[i]], shape[i]))

  # Check if any dimension has scattered indices
  has_scattered <- any(vapply(subsets, \(x) x$type == "scattered", logical(1L)))

  value_shape <- shape_abstract(value)

  if (has_scattered) {
    # Scattered index mode: Generate all combinations of indices
    # Each scatter operation updates a single scalar value

    indices_list <- lapply(subsets, \(x) x$indices)
    # Create all combinations using expand.grid
    indices_grid <- do.call(expand.grid, c(indices_list, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))
    n_scatter_ops <- nrow(indices_grid)

    # Verify value shape
    expected_shape <- vapply(subsets, \(x) x$size, integer(1L))
    if (!identical(expected_shape, value_shape)) {
      cli_abort("Value shape [{paste(value_shape, collapse = ', ')}] does not match subset shape [{paste(expected_shape, collapse = ', ')}]")
    }

    # scatter_indices shape: [n_scatter_ops, rank]
    scatter_indices_mat <- as.matrix(indices_grid) - 1L  # Convert to 0-based
    scatter_indices <- nv_tensor(scatter_indices_mat)

    # Flatten value: [expected_shape] -> [n_scatter_ops]
    # The update tensor has only the batch dimension (no window dimensions)
    value_with_dims <- nv_reshape(value, shape = n_scatter_ops)

    # For scattered indices updating single elements:
    # - update_window_dims: empty (no window, just batch dimension)
    # - inserted_window_dims: all input dimensions (they're not in the update)
    # - scatter_dims_to_operand_dims: identity mapping
    scatter_dim_numbers <- ScatterDimensionNumbers(
      update_window_dims = integer(),  # No window dims (1-based)
      inserted_window_dims = seq_len(length(shape)),  # All input dims inserted (1-based)
      scatter_dims_to_operand_dims = seq_len(length(shape)),  # Identity mapping (1-based)
      index_vector_dim = 2L  # Indices in dimension 2 (1-based)
    )
  } else {
    # Contiguous slice mode: Single scatter operation with window dimensions
    start_indices <- vapply(subsets, \(x) x$start, integer(1L))
    slice_sizes <- vapply(subsets, \(x) x$size, integer(1L))

    # Verify value shape
    if (!identical(slice_sizes, value_shape)) {
      cli_abort("Value shape [{paste(value_shape, collapse = ', ')}] does not match slice shape [{paste(slice_sizes, collapse = ', ')}]")
    }

    # Create a single scatter index
    scatter_indices_vec <- start_indices - 1L  # Convert to 0-based
    scatter_indices <- nv_tensor(matrix(scatter_indices_vec, nrow = 1L))

    # Add batch dimension: [slice_shape] -> [1, slice_shape]
    value_with_dims <- nv_reshape(value, shape = c(1L, value_shape))

    # All value dimensions are window dimensions (no insertion)
    scatter_dim_numbers <- ScatterDimensionNumbers(
      update_window_dims = seq_len(length(value_shape)) + 1L,  # Dims start after batch (1-based)
      inserted_window_dims = integer(),  # No inserted dimensions
      scatter_dims_to_operand_dims = seq_len(length(shape)),  # Identity mapping (1-based)
      index_vector_dim = 2L  # Indices in dimension 2 (1-based)
    )
  }

  # Use scatter to update
  result <- nvl_scatter(
    input = x,
    scatter_indices = scatter_indices,
    update = value_with_dims,
    scatter_dimension_numbers = scatter_dim_numbers,
    update_computation = function(old, new) new
  )

  result[[1L]]
}

#' @title Scatter Update
#' @description
#' General scatter operation for updating a tensor at specified indices.
#' This exposes the full capabilities of the scatter operation.
#' @param input ([`tensorish`])\cr
#'   Input tensor to scatter into.
#' @param indices ([`tensorish`])\cr
#'   Scatter indices tensor.
#' @param updates ([`tensorish`])\cr
#'   Update values tensor.
#' @param update_window_dims (`integer()`)\cr
#'   Dimensions of updates that are window dimensions (1-based).
#' @param inserted_window_dims (`integer()`)\cr
#'   Dimensions to insert in the input (1-based).
#' @param scatter_dims_to_operand_dims (`integer()`)\cr
#'   Mapping from scatter dimensions to operand dimensions (1-based).
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension containing the index vector (1-based).
#' @param update_computation (`function`)\cr
#'   Binary function to combine old and new values. Defaults to replacement.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are sorted.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether all indices are unique.
#' @return [`tensorish`]
#' @export
#' @examples
#' \dontrun{
#' # Replace values at specific indices
#' x <- nv_tensor(c(1L, 2L, 3L, 4L))
#' indices <- nv_tensor(matrix(c(1L, 3L), nrow = 1L))
#' updates <- nv_tensor(c(10L, 30L))
#' result <- nv_scatter_update(
#'   input = x,
#'   indices = indices,
#'   updates = updates,
#'   update_window_dims = 2L,
#'   inserted_window_dims = integer(),
#'   scatter_dims_to_operand_dims = 1L,
#'   index_vector_dim = 2L
#' )
#'
#' # Add to existing values
#' result <- nv_scatter_update(
#'   input = x,
#'   indices = indices,
#'   updates = updates,
#'   update_window_dims = 2L,
#'   inserted_window_dims = integer(),
#'   scatter_dims_to_operand_dims = 1L,
#'   index_vector_dim = 2L,
#'   update_computation = function(old, new) old + new
#' )
#' }
nv_scatter_update <- function(
  input,
  indices,
  updates,
  update_window_dims,
  inserted_window_dims = integer(),
  scatter_dims_to_operand_dims,
  index_vector_dim,
  update_computation = function(old, new) new,
  indices_are_sorted = FALSE,
  unique_indices = FALSE
) {
  if (!is_tensorish(input)) {
    cli_abort("Expected tensorish input, but got {.cls {class(input)[1]}}")
  }
  if (!is_tensorish(indices)) {
    cli_abort("Expected tensorish indices, but got {.cls {class(indices)[1]}}")
  }
  if (!is_tensorish(updates)) {
    cli_abort("Expected tensorish updates, but got {.cls {class(updates)[1]}}")
  }

  scatter_dim_numbers <- ScatterDimensionNumbers(
    update_window_dims = as.integer(update_window_dims),
    inserted_window_dims = as.integer(inserted_window_dims),
    scatter_dims_to_operand_dims = as.integer(scatter_dims_to_operand_dims),
    index_vector_dim = as.integer(index_vector_dim)
  )

  result <- nvl_scatter(
    input = input,
    scatter_indices = indices,
    update = updates,
    scatter_dimension_numbers = scatter_dim_numbers,
    indices_are_sorted = indices_are_sorted,
    unique_indices = unique_indices,
    update_computation = update_computation
  )

  result[[1L]]
}
