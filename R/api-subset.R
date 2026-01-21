#' @include api.R
#' @include primitives.R

# TODO: If the input is a known constant wrapepd in ConcreteTensor, we can also infer things like unique_indices etc. (?)

#' @title Compute All Combinations of Start Indices
#' @description
#' Given a list of start indices per dimension, computes all combinations
#' (cartesian product) of these indices. This is used for scatter/gather
#' operations where multiple dimensions have scattered indices.
#'
#' @param start_indices_per_dim A list where each element is a 1-D tensor
#'   containing the start indices for that dimension.
#'
#' @return A 2-D tensor of shape `(n_combinations, n_dims)` where each row
#'   is one combination of start indices.
#'
#' @details
#' For example, if `start_indices_per_dim = list(tensor(c(1, 3)), tensor(c(2, 4)), tensor(7), tensor(2))`:
#' - Dimension 1 has indices: 1, 3 (2 values)
#' - Dimension 2 has indices: 2, 4 (2 values)
#' - Dimension 3 has index: 7 (1 value)
#' - Dimension 4 has index: 2 (1 value)
#'
#' The result is a tensor with shape (4, 4):
#' ```
#' [[1, 2, 7, 2],
#'  [1, 4, 7, 2],
#'  [3, 2, 7, 2],
#'  [3, 4, 7, 2]]
#' ```
#'
#' @export
nv_meshgrid_start_indices <- function(start_indices_per_dim) {
  n_dims <- length(start_indices_per_dim)
  if (n_dims == 0L) {
    cli_abort("start_indices_per_dim must have at least one element")
  }

  # Get the number of indices per dimension
  lengths <- vapply(start_indices_per_dim, function(x) {
    sh <- shape_abstract(x)
    if (length(sh) == 0L) {
      1L
    } else if (length(sh) == 1L) {
      sh[1L]
    } else {
      cli_abort("Each element must be a scalar or 1-D tensor")
    }
  }, integer(1L))

  # Total number of combinations
  n_combinations <- prod(lengths)

  # Output shape for the meshgrid (before flattening)
  meshgrid_shape <- lengths

  # For each dimension, broadcast its indices to the full meshgrid shape
  # Then reshape and concatenate
  index_columns <- lapply(seq_len(n_dims), function(i) {
    idx_tensor <- start_indices_per_dim[[i]]
    n_idx <- lengths[i]

    # Ensure it's 1-D
    if (length(shape_abstract(idx_tensor)) == 0L) {
      idx_tensor <- nv_reshape(idx_tensor, c(1L))
    }

    # Create broadcast shape: 1s everywhere except dimension i
    bcast_shape <- rep(1L, n_dims)
    bcast_shape[i] <- n_idx

    # Reshape and broadcast to full meshgrid shape
    idx_reshaped <- nv_reshape(idx_tensor, bcast_shape)
    idx_broadcast <- nv_broadcast_to(idx_reshaped, meshgrid_shape)

    # Flatten to 1-D column
    nv_reshape(idx_broadcast, c(n_combinations))
  })

  # Stack columns into (n_combinations, n_dims) tensor
  # Use concatenate with reshape to stack
  stacked <- lapply(index_columns, function(col) {
    nv_reshape(col, c(n_combinations, 1L))
  })

  do.call(function(...) nv_concatenate(..., dimension = 2L), stacked)
}

# Helper functions for subset operations ======================================

#' Parse subset specifications and fill unspecified dimensions
#' @param exprs List of subset expressions (from alist)
#' @param operand_shape Shape of the operand tensor
#' @param allow_vectors Whether to allow vector indices (for scatter)
#' @param require_all Whether all dimensions must be specified (for scatter)
#' @param parent_frame The parent frame for evaluation
#' @return List of parsed subset specifications
#' @noRd
parse_subset_specs <- function(
  exprs,
  operand_shape,
  allow_vectors = FALSE,
  require_all = FALSE,
  parent_frame = parent.frame()
) {
  rank <- length(operand_shape)

  if (length(exprs) > rank) {
    cli_abort("Too many indices: got {length(exprs)}, expected at most {rank}")
  }

  if (require_all && length(exprs) != rank) {
    cli_abort("Expected {rank} indices, but got {length(exprs)}")
  }

  # Parse provided subset specifications
  # We need to pass the parent frame through properly
  slices <- lapply(seq_along(exprs), function(i) {
    # parent_frame is already the correct frame - it was passed to parse_subset_specs
    # We need to add 1 to account for the lapply wrapper
    parse_subset_spec(exprs[[i]], operand_shape[i], allow_vectors = allow_vectors, parent_frame = parent_frame)
  })

  # Fill remaining dimensions with full selections
  if (length(slices) < rank) {
    for (i in (length(slices) + 1L):rank) {
      slices[[i]] <- list(type = "full", start = 1L, size = operand_shape[i], indices = NULL, drop = FALSE)
    }
  }

  slices
}

#' Parse a single subset specification for gather/scatter operations
#' @param e Expression to parse
#' @param d Dimension size
#' @param allow_vectors Whether to allow vector indices (for scatter)
#' @param parent_frame The parent frame for evaluation
#' @return List with type, start, size, indices, drop
#' @noRd
parse_subset_spec <- function(e, d, allow_vectors = FALSE, parent_frame = parent.frame()) {
  is_integerish <- function(x) {
    test_integerish(x, len = 1L, any.missing = FALSE)
  }
  is_integerish_vector <- function(x) {
    test_integerish(x, min.len = 1L, any.missing = FALSE)
  }

  # Check for missing argument (empty name from alist) - select all
  if (is.symbol(e) && (identical(as.character(e), "") || identical(e, quote(`:`)))) {
    return(list(type = "full", start = 1L, size = d, indices = NULL, drop = FALSE))
  }

  # Single integer literal - drops dimension
  if (is_integerish(e)) {
    e <- as.integer(e)
    return(list(type = "single", start = e, size = 1L, indices = e, drop = TRUE))
  }

  # Range (a:b)
  if (is.call(e) && identical(e[[1]], quote(`:`))) {
    start <- eval(e[[2]], envir = parent_frame)
    end <- eval(e[[3]], envir = parent_frame)
    if (is_integerish(start) && is_integerish(end)) {
      start <- as.integer(start)
      end <- as.integer(end)
      if (start > end) {
        cli_abort("Range start ({start}) must be less than or equal to end ({end})")
      }
      return(list(type = "range", start = start, size = end - start + 1L, indices = start:end, drop = FALSE))
    }
    cli_abort("Ranges must be static")
  }

  # Handle list() calls specially - these preserve dimensions
  if (is.call(e) && identical(e[[1]], quote(list))) {
    # Evaluate the list elements
    list_elements <- lapply(as.list(e)[-1L], function(el) eval(el, envir = parent_frame))

    if (length(list_elements) == 0L) {
      cli_abort("Empty list() indices are not allowed")
    }

    # Check all elements are integerish scalars
    if (!all(vapply(list_elements, is_integerish, logical(1L)))) {
      cli_abort("All list() elements must be scalar integers")
    }

    indices <- as.integer(unlist(list_elements))

    if (length(indices) == 1L) {
      # Single element in list - don't drop dimension
      return(list(type = "single", start = indices, size = 1L, indices = indices, drop = FALSE))
    }

    # Multiple elements in list - gather operation
    e_tensor <- nv_tensor(indices, dtype = "i64")
    return(list(type = "gather", start = NULL, size = length(indices), indices = e_tensor, drop = FALSE))
  }

  # Evaluate expressions (symbols or calls)
  if (is.symbol(e) || is.call(e)) {
    e <- eval(e, envir = parent_frame)
  }

  # Check for AnvilRange (result of x:y where x or y is AnvilBox/AnvilTensor)
  if (inherits(e, "AnvilRange")) {
    cli_abort("Ranges must be static")
  }

  # R vectors of length > 1 are not allowed for subset (use list() instead)
  if (is_integerish_vector(e) && length(e) > 1L && !allow_vectors) {
    cli_abort(
      "Vectors of length > 1 are not allowed as subset indices. Use list() to select multiple elements, e.g. x[list(1, 3), ] instead of x[c(1, 3), ]"
    )
  }

  # Integer vector (for scatter only)
  if (allow_vectors && is_integerish_vector(e)) {
    e_int <- as.integer(e)
    e_tensor <- nv_tensor(e_int, dtype = "i64")
    return(list(type = "scattered", start = NULL, size = shape_abstract(e_tensor), indices = e_tensor, drop = FALSE))
  }

  # Tensor indices - don't drop dimension (dynamic)
  if (is_tensorish(e) && (!is.atomic(e))) {
    if (is_anvil_tensor(e)) {
      dt <- dtype_abstract(e)
      if (!(inherits(dt, "IntegerType") || inherits(dt, "UnsignedType"))) {
        cli_abort("Dynamic indices must be integers, but got {.cls {class(e)[1]}}")
      }
      nd <- ndims_abstract(e)
      if (nd > 1L) {
        cli_abort("Dynamic indices must be at most 1D, but got {nd}D tensor")
      }
      if (dt != as_dtype("i64")) {
        e <- nv_convert(e, dtype = "i64")
      }
      if (nd == 0L) {
        # Scalar tensor indices drop dimension (like R literals)
        return(list(type = "single", start = e, size = 1L, indices = e, drop = TRUE))
      }
      # 1D tensor indices
      tensor_size <- shape_abstract(e)[1L]
      if (tensor_size == 1L) {
        # Single-element 1D tensor - treat as single index (don't drop dimension)
        return(list(type = "single", start = e, size = 1L, indices = e, drop = FALSE))
      }
      # Multi-element 1D tensor - treat as gather
      if (allow_vectors) {
        return(list(type = "scattered", start = NULL, size = tensor_size, indices = e, drop = FALSE))
      } else {
        return(list(type = "gather", start = NULL, size = tensor_size, indices = e, drop = FALSE))
      }
    }
    # For gather, single tensor indices
    return(list(type = "single", start = e, size = 1L, indices = e, drop = FALSE))
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
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  operand_shape <- shape_abstract(x)
  # Use substitute() instead of enexprs() to support forwarding from [ operator
  exprs <- eval(substitute(alist(...)))

  # Parse subset specifications and fill unspecified dimensions
  # parent.frame() here refers to the caller of nv_subset (e.g., `[.AnvilTensor`)
  # We pass it to parse_subset_specs which will forward it to parse_subset_spec
  slices <- parse_subset_specs(
    exprs,
    operand_shape,
    allow_vectors = FALSE,
    require_all = FALSE,
    parent_frame = parent.frame()
  )

  rank <- length(operand_shape)

  # Check for gather type (multiple allowed now via meshgrid)
  is_gather <- vapply(slices, \(s) s$type == "gather", logical(1L))
  gather_dims <- which(is_gather)

  if (length(gather_dims) > 0L) {
    # Handle gather case (list with multiple elements in one or more dimensions)
    nv_subset_gather(x, slices, operand_shape, rank, gather_dims)
  } else {
    # Handle simple slice case (no multi-element gather)
    nv_subset_slice(x, slices, operand_shape, rank)
  }
}

#' Simple slice subset (no multi-element gather)
#' @noRd
nv_subset_slice <- function(x, slices, operand_shape, rank) {
  # Determine if all indices are static (type "single" or "range" with numeric start)
  static <- all(vapply(
    slices,
    function(s) {
      s$type %in% c("full", "range") || (s$type == "single" && is.numeric(s$start))
    },
    logical(1L)
  ))

  # slice_sizes: for each operand dimension
  slice_sizes <- vapply(slices, \(x) x$size, integer(1L))

  # Dimensions to collapse based on drop field from parsed specs
  collapsed_slice_dims <- which(vapply(slices, \(x) isTRUE(x$drop), logical(1L)))

  # start_index_map: maps each position in the index vector to operand dimensions
  start_index_map <- seq_len(rank)

  # offset_dims: which output dimensions correspond to the slice (after the batch dim)
  n_remaining <- rank - length(collapsed_slice_dims)
  offset_dims <- seq_len(n_remaining) + 1L # 1-based, starting at position 2

  if (static) {
    # All indices are static - create a constant start_indices tensor
    # Keep indices as 1-based - nvl_gather handles conversion to 0-based
    start_indices_vec <- vapply(slices, \(x) x$start, integer(1L))
    start_indices <- nv_tensor(matrix(start_indices_vec, nrow = 1L), dtype = "i32")
    index_vector_dim <- 2L # 1-based
  } else {
    # Dynamic indices - build start_indices from tensor values
    # Keep indices as 1-based - nvl_gather handles conversion to 0-based
    start_parts <- lapply(slices, function(s) {
      if (is.numeric(s$start)) {
        nv_scalar(as.integer(s$start), dtype = "i32")
      } else {
        nv_convert(s$start, dtype = "i32")
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
    index_vector_dim <- 2L # 1-based
  }

  # For simple slicing, the indices are trivially sorted (single starting position)
  # and unique (we only have one slice starting position)
  out <- nvl_gather(
    operand = x,
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    operand_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = start_index_map,
    index_vector_dim = index_vector_dim,
    indices_are_sorted = static,
    unique_indices = static
  )

  # Remove the batch dimension (size 1, at position 1) from gather result
  result_shape <- shape(out)
  if (length(result_shape) > 0L && result_shape[1L] == 1L) {
    out <- nv_reshape(out, shape = result_shape[-1L])
  }

  out
}

#' Gather subset (list with multiple elements in one or more dimensions)
#' @noRd
nv_subset_gather <- function(x, slices, operand_shape, rank, gather_dims) {
  # Build list of start indices per dimension

  # For gather dims: use the indices tensor
  # For non-gather dims: use a single start index (as scalar tensor)
  start_indices_per_dim <- lapply(seq_len(rank), function(i) {
    s <- slices[[i]]
    if (i %in% gather_dims) {
      # Gather dimension - use the indices tensor (already 1-based)
      s$indices
    } else {
      # Non-gather dimension - use start index as scalar
      if (is.numeric(s$start)) {
        nv_scalar(as.integer(s$start), dtype = "i64")
      } else {
        nv_convert(s$start, dtype = "i64")
      }
    }
  })

  # Compute the shape of the gather dimensions (for reshaping output)
  # This is the meshgrid shape before flattening
  gather_shape <- vapply(gather_dims, function(i) slices[[i]]$size, integer(1L))

  # Use meshgrid to get all combinations of start indices
  # Result shape: [n_combinations, rank]
  start_indices <- nv_meshgrid_start_indices(start_indices_per_dim)
  start_indices <- nv_convert(start_indices, dtype = "i32")

  n_combinations <- shape_abstract(start_indices)[1L]

  # For gather, slice_sizes is 1 for gather dims, and normal for others
  slice_sizes <- vapply(seq_len(rank), function(i) {
    if (i %in% gather_dims) 1L else slices[[i]]$size
  }, integer(1L))

  # Collapsed dims: gather dims are always collapsed (we're selecting single elements)
  # Plus any other dims with drop = TRUE
  collapsed_slice_dims <- c(
    gather_dims,
    which(vapply(seq_len(rank), function(i) {
      !(i %in% gather_dims) && isTRUE(slices[[i]]$drop)
    }, logical(1L)))
  )
  collapsed_slice_dims <- sort(unique(collapsed_slice_dims))

  # offset_dims: non-collapsed dims go after the batch dim
  n_remaining <- rank - length(collapsed_slice_dims)
  offset_dims <- seq_len(n_remaining) + 1L

  # Conservatively assume indices may not be sorted or unique
  indices_are_sorted <- FALSE
  unique_indices <- FALSE

  out <- nvl_gather(
    operand = x,
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    operand_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = seq_len(rank),
    index_vector_dim = 2L,
    indices_are_sorted = indices_are_sorted,
    unique_indices = unique_indices
  )

  # The gather output has shape [n_combinations, ...remaining_dims...]
  # We need to reshape the first dimension (n_combinations) back to the
  # meshgrid shape (gather_shape) to get proper multi-dimensional output
  #
  # For x[list(1,3), list(2,4)] on a matrix:
  # - gather produces shape [4] (flattened combinations)
  # - we reshape to [2, 2] (the gather dimension sizes)

  current_shape <- shape_abstract(out)
  if (length(gather_dims) > 1L && length(current_shape) > 0L) {
    # Replace the first dimension (n_combinations) with the gather_shape
    new_shape <- c(gather_shape, current_shape[-1L])
    out <- nv_reshape(out, new_shape)
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
  if (!is_tensorish(x)) {
    cli_abort("Expected tensorish, but got {.cls {class(x)[1]}}")
  }
  if (!is_tensorish(value, literal = TRUE)) {
    cli_abort("Expected tensorish value, but got {.cls {class(value)[1]}}")
  }

  lhs_shape <- shape_abstract(x)
  exprs <- eval(substitute(alist(...)))

  # Parse subset specifications (require all dimensions to be specified)
  subsets <- parse_subset_specs(
    exprs,
    lhs_shape,
    allow_vectors = TRUE,
    require_all = TRUE,
    parent_frame = parent.frame()
  )

  # Check if any dimension has scattered indices
  has_scattered <- any(vapply(subsets, \(x) x$type == "scattered", logical(1L)))

  value_shape <- shape_abstract(value)

  subset_shape <- vapply(subsets, \(s) s$size, integer(1L))
  if (!ndims_abstract(value)) {
    value <- nv_broadcast_to(value, subset_shape)
  } else if (!identical(subset_shape, value_shape)) {
    # fmt: skip
    cli_abort("Subset shape {shape2string(subset_shape)} does not match value shape {shape2string(value_shape)}") # nolint
  }

  is_scattered <- vapply(subsets, \(s) s$type == "scattered", logical(1L))
  if (sum(is_scattered) >= 2L) {
    cli_abort("There can be at most one scattered subset")
  }

  if (dtype(x) != dtype(value)) {
    if (!promotable_to(dtype(value), dtype(x))) {
      cli_abort(
        "Value type {dtype2string(dtype(value))} is not promotable to left-hand side type {dtype2string(dtype(x))}"
      )
    }
    value <- nv_convert(value, dtype = dtype(x))
  }

  if (!has_scattered) {
    # In this case, our scatter contains a single vector that indicates the starting position in each dimension
    # Convert start positions to tensors and concatenate into a 1D vector
    start_tensors <- lapply(subsets, function(s) {
      if (is.numeric(s$start)) {
        nv_scalar(s$start, dtype = "i64")
      } else {
        nv_convert(s$start, dtype = "i64")
      }
    })

    # Always concatenate into a 1D tensor, even for single dimension
    start_position <- do.call(nv_concatenate, c(start_tensors, list(dimension = 1L)))

    # Determine if indices are static (all starts are numeric, not tensors)
    all_static <- all(vapply(subsets, function(s) is.numeric(s$start), logical(1L)))

    # For a single contiguous slice write, the indices are trivially sorted and unique
    nvl_scatter(
      input = x,
      scatter_indices = start_position,
      update = value,
      update_window_dims = seq_along(subset_shape),
      inserted_window_dims = integer(),
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = seq_along(lhs_shape),
      index_vector_dim = 1L,
      indices_are_sorted = all_static,
      unique_indices = all_static,
      update_computation = function(old, new) new
    )
  } else {
    # first, we build the complete start index
    scatter_vals <- subsets[[which(is_scattered)]]$indices
    n <- shape(scatter_vals)
    # scatter_indices either has shape [n, subsets], where the i-th column contains the
    # scatter values (where i is the index of the scattered subset)
    # and the other columns contain the start indices for the slices/indexes

    scatter_indices_list <- lapply(seq_along(subsets), function(i) {
      if (is_scattered[i]) {
        nv_reshape(scatter_vals, shape = c(n, 1L))
      } else {
        nv_broadcast_to(subsets[[i]]$start, c(n, 1L))
      }
    })
    scatter_indices <- do.call(nv_concatenate, c(scatter_indices_list, list(dimension = 2L)))

    # For scattered indices, we conservatively assume indices may not be sorted or unique
    # since the scatter indices could come from a dynamic tensor.
    # A future optimization could check if indices are static R vectors and determine
    # sorted/unique properties at trace time.
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
