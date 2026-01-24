# set utils
set <- function() {
  hashtab()
}

set_has <- function(set, key) {
  !identical(gethash(set, key, NA), NA)
}

set_add <- function(set, key) {
  set[[key]] <- NULL
}

dtype_from_buffer <- function(x) {
  d <- as.character(dtype(x))
  as_dtype(d)
}

hashkeys <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
    val[[idx]] <<- k
  })
  val
}

hashvalues <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
    val[[idx]] <<- v
  })
  val
}

is_nv_type <- function(x) {
  any(sapply(globals$nv_types, function(type, x) inherits(x, type), x))
}


transpose_list <- function(.l) {
  if (length(.l) == 0L) {
    return(list())
  }
  res <- .mapply(list, .l, list())
  if (length(res) == length(.l[[1L]])) {
    names(res) <- names(.l[[1L]])
  }
  res
}

# these functions also work with primitives etc.
formalArgs2 <- function(f) {
  names(formals2(f))
}

formals2 <- function(f) {
  formals(args(f))
}


# We assume little endian
minmax_raw <- function(bits, signed = TRUE) {
  stopifnot(bits %% 8 == 0, bits >= 8)
  n <- bits %/% 8
  if (!signed) {
    return(list(
      min = as.raw(rep(0x00, n)),
      max = as.raw(rep(0xFF, n))
    ))
  }
  hi_min <- as.raw(0x80) # 1000 0000
  hi_max <- as.raw(0x7F) # 0111 1111
  zeros <- as.raw(rep(0x00, n - 1))
  ff <- as.raw(rep(0xFF, n - 1))
  list(min = c(zeros, hi_min), max = c(ff, hi_max))
}


nv_minval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(-Inf, dtype = dtype, device = device)
  } else if (dtype %in% c("i1", "pred")) {
    nv_scalar(FALSE, dtype = "pred", device = device)
  } else {
    nv_scalar(globals$ranges_raw[[dtype]]$min, dtype = dtype, device = device)
  }
}

nv_maxval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(Inf, dtype = dtype, device = device)
  } else if (dtype %in% c("i1", "pred")) {
    nv_scalar(TRUE, dtype = "pred", device = device)
  } else {
    nv_scalar(globals$ranges_raw[[dtype]]$max, dtype = dtype, device = device)
  }
}

without <- function(x, indices) {
  if (length(indices)) {
    x[-indices]
  } else {
    x
  }
}

zero_env <- function() {
  new.env(size = 0L, parent = emptyenv())
}

shape2string <- function(x, parenthesize = TRUE) {
  if (is_shape(x)) {
    x <- x$dims
  }
  if (parenthesize) {
    sprintf("(%s)", paste0(x, collapse = ","))
  } else {
    paste0(x, collapse = ",")
  }
}

shapes2string <- function(shapes) {
  paste0(sapply(shapes, shape2string), sep = ", ")
}

zeros <- function(dtype, shape, ambiguous) {
  nvl_fill(0L, dtype = dtype, shape = shape, ambiguous = ambiguous)
}

ones <- function(dtype, shape, ambiguous) {
  nvl_fill(1L, dtype = dtype, shape = shape, ambiguous = ambiguous)
}


# TODO: Better handle ambiguity
zeros_like <- function(x, ambiguous) {
  zeros(dtype(x), shape(x), ambiguous)
}

# TODO: Handle ambiguity
ones_like <- function(x, ambiguous) {
  ones(dtype(x), shape(x), ambiguous)
}

#' @title Abstract Properties
#' @name abstract_properties
#' @description
#' Calls the extractor after converting the input to an [`AbstractTensor`].
#' @param x ([`tensorish`])\cr
#' @export
shape_abstract <- function(x) {
  shape(to_abstract(x))
}

#' @rdname abstract_properties
#' @export
ndims_abstract <- function(x) {
  length(shape_abstract(x))
}

#' @rdname abstract_properties
#' @export
ambiguous_abstract <- function(x) {
  to_abstract(x)$ambiguous
}

#' @rdname abstract_properties
#' @export
dtype_abstract <- function(x) {
  dtype(to_abstract(x))
}

#' @export
#' @rdname abstract_properties
ambiguous_abstract <- function(x) {
  to_abstract(x)$ambiguous
}

dtype2string <- function(dtype, ambiguous = FALSE) {
  paste0(repr(dtype), if (ambiguous) "?")
}

is_lit <- function(x) {
  test_scalar(x) && (is.numeric(x) || is.logical(x))
}

cache_size <- function(f) {
  environment(f)$cache$size
}

# Clamp gather start indices to valid ranges, matching XLA's forward pass behavior.
# This ensures that out-of-bounds indices are clamped to [1, operand_size - slice_size + 1]
# for each dimension.
gather_clamp_indices <- function(
  start_indices,
  operand_shape,
  slice_sizes,
  start_index_map,
  index_vector_dim
) {
  # slice_sizes are in the order of the operand_shape, so we need to reverse the start_index_map
  if (length(operand_shape) != length(slice_sizes)) {
    cli_abort("operand_shape and slice_sizes must have the same length")
  }

  indices_shape <- shape(start_indices)
  n_index_coords <- length(start_index_map)

  if (n_index_coords == 0L) {
    return(start_indices)
  }

  # Build max bounds for each coordinate
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

    # The max bound is the same for a given slice along the index_vector_dim
    max_tensor_vals <- nvl_reshape(
      nv_convert(nv_tensor(max_bounds, dtype = "i64"), dtype = dtype(start_indices)),
      bounds_shape
    )
    max_tensor <- nv_broadcast_to(max_tensor_vals, indices_shape)

    nvl_clamp(min_tensor, start_indices, max_tensor)
  } else {
    # Implicit index vector (single coordinate)
    min_tensor <- nvl_fill(1L, dtype = dtype(start_indices), shape = integer())
    max_tensor <- nvl_fill(max_bounds[1L], dtype = dtype(start_indices), shape = integer())
    nvl_clamp(min_tensor, start_indices, max_tensor)
  }
}

# Compute gather slice_sizes from scatter parameters.
# This inverts a scatter into a gather: for each operand dimension, the slice
# size is 1 for inserted/batching dims, or the update's window size otherwise.
scatter_to_gather_slice_sizes <- function(
  update_shape,
  input_shape,
  update_window_dims,
  inserted_window_dims,
  input_batching_dims
) {
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
  slice_sizes
}
