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
#' Updates elements of a tensor.
#' Supports both contiguous slices (e.g., `2:5`) and scattered indices (e.g., `c(1, 3, 6)`).
#' This has copy-on-write semantics just like for standard R arrays.
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
#' x <- nv_subset_assign(x, 2:4, value = nv_tensor(c(20, 30, 40)))
#'
#' # Update scattered indices
#' x <- nv_tensor(1:10)
#' x <- nv_subset_assign(x, c(1, 3, 6), value = nv_tensor(c(10, 30, 60)))
#' }
nv_subset_assign <- function(x, ..., value) {
  is_integerish <- function(x) {
    test_integerish(x, len = 1L, any.missing = FALSE)
  }
  is_integerish_vector <- function(x) {
    test_integerish(x, min.len = 1L, any.missing = FALSE)
  }

  # TODO:

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
    e <- eval(e, envir = parent.frame(2L))
    if (is_integerish_vector(e)) {
      e <- as.integer(e)
      e <- nv_tensor(e, dtype = "i64")
      return(list(type = "scattered", start = NULL, size = shape_abstract(e), indices = e))
    }
    if (is_anvil_tensor(e)) {
      dt <- dtype_abstract(e)
      if (!(inherits(dt, "IntegerType") || inherits(dt, "UnsignedType"))) {
        cli_abort("Dynamic indices must be integers, but got {.cls {class(e)[1]}}")
      }
      nd <- ndims_abstract(e)
      if (nd > 1L) {
        cli_abort("Dynamic indices must be at most 1D, but got {.cls {class(e)[1]}}")
      }
      if (dt != as_dtype("i64")) {
        e <- nv_convert(e, dtype = "i64")
      }
      if (nd == 0L) {
        return(list(type = "single", start = e, size = 1L, indices = e))
      }
      return(list(type = "scattered", start = NULL, size = shape(e), indices = e))
    }
    cli_abort("Invalid subset expression")
  }

  if (!is_tensorish(x)) {
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  if (!is_tensorish(value)) {
    cli_abort("Expected tensorish value, but got {.cls {class(value)[1]}}")
  }

  lhs_shape <- shape_abstract(x)
  exprs <- eval(substitute(alist(...)))

  if (length(exprs) != length(lhs_shape)) {
    cli_abort("Expected {length(lhs_shape)} indices, but got {length(exprs)}")
  }

  # Parse subset specifications
  subsets <- lapply(seq_along(exprs), function(i) get_subset(exprs[[i]], lhs_shape[i]))

  # Check if any dimension has scattered indices
  has_scattered <- any(vapply(subsets, \(x) x$type == "scattered", logical(1L)))

  value_shape <- shape_abstract(value)

  subset_shape <- vapply(subsets, \(s) s$size, integer(1L))
  if (is_lit(value)) {
    value <- nv_broadcast_to(value, subset_shape)
  } else if (!identical(subset_shape, value_shape)) {
    # fmt: skip
    cli_abort("Subset shape {shape2string(subset_shape)} does not match value shape {shape2string(value_shape)}") # nolint
  }

  # TODO: Convert rhs
  # TODO: Support scalar on rhs and broadcast it to subset shape
  is_scattered <- vapply(subsets, \(s) s$type == "scattered", logical(1L))
  if (sum(is_scattered) >= 2L) {
    cli_abort("There can be at most one scattered subset")
  }

  # TODO:
  # Set unique_indices and indices_are_sorted correctly

  if (dtype(x) != dtype(value)) {
    if (!promotable_to(dtype(value), dtype(x))) {
      cli_abort("Value type {dtype2string(dtype(value))} is not promotable to left-hand side type {dtype2string(dtype(x))}")
    }
    value <- nv_convert(value, dtype = dtype(x))
  }

  # TODO: Ensure that for static subsets the update does not go out of bounds

  if (!has_scattered) {
    # In this case, our scatter contains a single vector that indicates the starting position in each dimension
    start_position <- rlang::exec(nv_concatenate, !!!lapply(subsets, \(s) s$start), dimension = 1L)

    nvl_scatter(
      input = x,
      scatter_indices = start_position,
      update = value,
      update_window_dims = seq_along(subset_shape),
      inserted_window_dims = integer(),
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = seq_along(lhs_shape),
      index_vector_dim = ndims_abstract(value),
      indices_are_sorted = FALSE,
      unique_indices = FALSE,
      update_computation = function(old, new) new
    )
  } else {
    # first, we build the complete start index
    scatter_vals <- subsets[[which(is_scattered)]]$indices
    n <- shape(scatter_vals)
    # scatter_indices either has shape [n, subsets], where the i-th column contains the
    # scatter values (where is is the index of the scattered subset)
    # and the other columns contain the start indices for the slices/indexes

    scatter_indices_list <- lapply(seq_along(subsets), function(i) {
      if (is_scattered[i]) {
        nv_reshape(scatter_vals, shape = c(n, 1L))
      } else {
        nv_broadcast_to(subsets[[i]]$start, c(n, 1L))
      }
    })
    scatter_indices <- rlang::exec(nv_concatenate, !!!scatter_indices_list, dimension = 2L)

    nvl_scatter(
      input = x,
      scatter_indices = scatter_indices,
      update = value,
      update_window_dims = seq_along(lhs_shape)[!is_scattered],
      inserted_window_dims = seq_along(subsets)[is_scattered],
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = seq_along(lhs_shape),
      # we always add the index dim explicitly
      index_vector_dim = 2L,
      indices_are_sorted = FALSE,
      unique_indices = FALSE,
      update_computation = function(old, new) new
    )
  }
}
