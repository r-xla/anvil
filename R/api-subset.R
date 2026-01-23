#' @include jit.R

SubsetFull <- function(size) {
  structure(list(size = size), class = "SubsetFull")
}

SubsetRange <- function(start, end) {
  structure(list(start = start, size = end - start + 1L), class = "SubsetRange")
}

SubsetIndex <- function(index) {
  static <- is.numeric(index)
  if (static) {
    if (length(index) != 1L) cli_abort("Internal error")
  } else {
    if (ndims_abstract(index) != 0L) cli_abort("Internal error")
  }
  structure(list(index = index, size = 1L, static = static), class = "SubsetIndex")
}

SubsetIndices <- function(indices) {
  static <- is.numeric(indices)
  if (static) {
    size <- length(indices)
  } else {
    nd <- ndims_abstract(indices)
    if (nd != 1L) {
      cli_abort("Internal error")
    }
    size <- shape_abstract(indices)[1L]
  }
  structure(list(indices = indices, size = size, static = static), class = "SubsetIndices")
}

is_subset_full <- function(x) inherits(x, "SubsetFull")
is_subset_range <- function(x) inherits(x, "SubsetRange")
is_subset_index <- function(x) inherits(x, "SubsetIndex")
is_subset_indices <- function(x) inherits(x, "SubsetIndices")

subset_start_positions <- function(subsets) {
  subset_start_position <- function(s) {
    if (is_subset_index(s)) {
      s$index
    } else if (inherits(s, "SubsetIndices")) {
      s$indices
    } else if (is_subset_full(s)) {
      1L
    } else if (is_subset_range(s)) {
      s$start
    } else {
      cli_abort("Internal error")
    }
  }

  lapply(subsets, subset_start_position)
}

# Inputs: list of 1-D tensors, each of shape [n_i].
# Non-multi-index dims have n_i = 1, multi-index dims have n_i > 1.
# Returns a tensor of shape [rank] or [gather_shape..., rank].
dynamic_start_indices <- function(starts) {
  rank <- length(starts)
  sizes <- vapply(starts, function(s) shape_abstract(s)[1L], integer(1L))
  multi_index_dims <- which(sizes > 1L)

  if (length(multi_index_dims) == 0L) {
    start <- do.call(nv_concatenate, c(starts, list(dimension = 1L)))
    return(start)
  }

  # consider s_1, ..., s_n
  # where some s_i are multi-index, some are not
  # We wan't to create a  tensor S of shape [size(s_i1), ..., size(s_im), n]
  # where i1, ..., im are the multi-index dimensions.
  # We can look at this as j tensors of shape [size(s_i1), ..., size(s_im), 1] (S[.., j]) that we create
  # for j that are not in {i1, ..., im} we simply broadcast the singular start index.
  # The other J slices (S[.., j]) for j \in {i1, ..., im} form
  # a cartesian product of the multi-index dimensions.
  # But we compute each layer S[.., j] individually, where for j = s_i1, the indices are constant
  # across every of the m dimensions except for the one corresponding to i1

  multi_index_sizes <- sizes[multi_index_dims]
  n_gather <- length(multi_index_dims)

  slices <- vector("list", rank)
  multi_index_i <- 1L
  for (d in seq_len(rank)) {
    if (identical(shape_abstract(starts[[d]]), 1L)) {
      slices[[d]] <- nv_broadcast_to(starts[[d]], c(multi_index_sizes, 1L))
    } else {
      slices[[d]] <- nvl_broadcast_in_dim(starts[[d]], c(multi_index_sizes, 1L), multi_index_i)
      multi_index_i <- multi_index_i + 1L
    }
  }
  out <- do.call(nv_concatenate, c(slices, list(dimension = n_gather + 1L)))
  out
}

.static_start_indices <- jit(function(...) {
  dynamic_start_indices(list(...))
})

static_start_indices <- function(starts) {
  starts <- lapply(starts, nv_tensor, dtype = "i32")
  do.call(.static_start_indices, starts)
}


#' Build a tensor of start indices (aka scatter_indices) from subset specs
#'
#' For each subset spec, extracts the start index and combines them into a tensor.
#' The dtype is determined automatically: i32 for static R ints, or the maximum
#' integer type present among dynamic tensor indices. Conversion is only performed
#' when at least one subset is dynamic.
#'
#' - Without multi_index_dims: returns a 1D tensor of shape [rank] (all starts are scalar).
#' - With multi_index_dims: returns a tensor of shape [gather_shape..., rank] where the
#'   gather dimensions' indices are broadcast across the cartesian product.
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A tensor of start indices
#' @noRd
subset_specs_start_indices <- function(subsets) {
  starts <- subset_start_positions(subsets)
  all_static <- all(vapply(starts, is.numeric, logical(1L)))
  if (all_static) {
    static_start_indices(starts)
  } else {
    # Convert R integers to 1D tensors, reshape 0D tensors to 1D
    starts <- lapply(starts, function(s) {
      if (is.numeric(s)) {
        nv_tensor(s, dtype = "i32")
      } else if (ndims_abstract(s) == 0L) {
        nv_reshape(s, 1L)
      } else {
        s
      }
    })
    dynamic_start_indices(starts)
  }
}

#' Convert subset specs to gather parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A list with all parameters needed for nvl_gather:
#'   - start_indices: tensor of start indices (shape [gather_shape..., rank] or [1, rank])
#'   - slice_sizes: integer vector
#'   - offset_dims: integer vector
#'   - collapsed_slice_dims: integer vector
#'   - start_index_map: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - multi_index_subset: logical
#' @noRd
subset_specs_to_gather <- function(subsets) {
  rank <- length(subsets)

  # Identify gather dimensions (SubsetIndices with multiple elements)
  multi_index_dims <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size > 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_dims) > 0L

  # slice_sizes: 1 for multi_index_dims, the size for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_dims) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  collapsed_slice_dims <- sort(c(
    multi_index_dims,
    which(vapply(
      seq_len(rank),
      function(i) {
        !(i %in% multi_index_dims) && is_subset_index(subsets[[i]])
      },
      logical(1L)
    ))
  ))

  start_indices <- subset_specs_start_indices(subsets)

  # offset_dims: positions in the output for non-collapsed operand dims.
  # The output interleaves batch (gather) dims and offset (slice) dims
  # in the order of the original operand dimensions.
  subset_index_dims <- which(vapply(subsets, is_subset_index, logical(1L)))
  surviving_dims <- setdiff(seq_len(rank), subset_index_dims)
  multi_among_surviving <- which(surviving_dims %in% multi_index_dims)
  offset_dims <- setdiff(seq_along(surviving_dims), multi_among_surviving)

  index_vector_dim <- length(multi_index_dims) + 1L

  list(
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    start_index_map = seq_len(rank),
    index_vector_dim = index_vector_dim,
    indices_are_sorted = !multi_index_subset,
    unique_indices = !multi_index_subset,
    multi_index_subset = multi_index_subset
  )
}

#' Convert subset specs to scatter parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A list with all parameters needed for nvl_scatter:
#'   - scatter_indices: tensor of scatter indices
#'   - update_window_dims: integer vector
#'   - inserted_window_dims: integer vector
#'   - scatter_dims_to_operand_dims: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - update_shape: integer vector (expected shape of the update tensor)
#' @noRd
subset_specs_to_scatter <- function(subsets) {
  rank <- length(subsets)

  multi_index_dims <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size > 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_dims) > 0L

  # slice_sizes: 1 for gather dims (individually addressed), normal for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_dims) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  scatter_indices <- subset_specs_start_indices(subsets)

  if (multi_index_subset) {
    # scatter_indices shape: [gather_shape..., rank]
    n_gather <- length(multi_index_dims)
    inserted_window_dims <- multi_index_dims
    update_window_dims <- setdiff(seq_len(rank), multi_index_dims)
    update_shape <- vapply(
      seq_len(rank),
      function(i) {
        if (i %in% multi_index_dims) subsets[[i]]$size else slice_sizes[i]
      },
      integer(1L)
    )
    index_vector_dim <- n_gather + 1L
  } else {
    # scatter_indices shape: [rank] (no batch dims)
    update_window_dims <- seq_len(rank)
    inserted_window_dims <- integer(0L)
    update_shape <- slice_sizes
    index_vector_dim <- 1L
  }

  list(
    scatter_indices = scatter_indices,
    update_window_dims = update_window_dims,
    inserted_window_dims = inserted_window_dims,
    scatter_dims_to_operand_dims = seq_len(rank),
    index_vector_dim = index_vector_dim,
    indices_are_sorted = !multi_index_subset,
    unique_indices = TRUE,
    update_shape = update_shape,
    multi_index_subset = multi_index_subset
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
    return(SubsetIndex(as.integer(e)))
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
    return(SubsetIndices(indices))
  }

  # AnvilRange (dynamic range) - not supported
  if (inherits(e, "AnvilRange")) {
    cli_abort("Ranges must be static")
  }

  # Tensor indices (AnvilTensor or GraphBox)
  if (is_tensorish(e) && !is.atomic(e)) {
    dt <- dtype_abstract(e)
    if (!(inherits(dt, "IntegerType") || inherits(dt, "UnsignedType"))) {
      cli_abort("Dynamic indices must be integers, but got {.cls {class(dt)[1]}}")
    }
    nd <- ndims_abstract(e)
    if (nd > 1L) {
      cli_abort("Dynamic indices must be at most 1D, but got {nd}D tensor")
    }
    # Scalar tensor drops dimension, 1D tensor preserves
    if (nd == 0L) {
      return(SubsetIndex(e))
    }
    return(SubsetIndices(e))
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

  subsets <- parse_subset_specs(quos, operand_shape)
  params <- subset_specs_to_gather(subsets)

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
    cli_abort("Expected tensorish `x`, but got {.cls {class(x)[1]}}")
  }
  if (!is_tensorish(value)) {
    cli_abort("Expected tensorish `value`, but got {.cls {class(value)[1]}}")
  }
  if (dtype_abstract(x) != dtype_abstract(value)) {
    dt_x <- dtype_abstract(x)
    dt_value <- dtype_abstract(value)
    if (!promotable_to(dt_value, dt_x)) {
      cli_abort(
        "Value type {dtype2string(dt_value)} is not promotable to left-hand side type {dtype2string(dt_x)}"
      )
    }
    value <- nv_convert(value, dtype = dt_x)
  }

  lhs_shape <- shape_abstract(x)
  # because we do NSE to determine `:`-calls
  quos <- rlang::enquos(...)

  subsets <- parse_subset_specs(quos, lhs_shape)
  params <- subset_specs_to_scatter(subsets)

  if (!ndims_abstract(value)) {
    value <- nv_broadcast_to(value, params$update_shape)
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
