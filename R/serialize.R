#' @title Tensor serialization and I/O
#' @name nv_serialization
#' @description
#' Read and write tensors using the safetensors format.
#' to/from raw vectors in memory.
#'
#' @param tensors (named `list` of [`AnvilTensor`])\cr
#'   Named list of tensors.
#' @param path (`character(1)`)\cr
#'   Path to the safetensors file.
#' @param con (connection)\cr
#'   A connection object to read/write from.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device for the tensor (`"cpu"`, `"cuda"`, ...).
#'   Default is to use the CPU.
#'
#' @return
#' - `nv_write()`: `NULL` (invisibly)
#' - `nv_read()`: Named list of [`AnvilTensor`] objects
#' - `nv_serialize()`: Raw vector containing serialized tensors
#' - `nv_unserialize()`: Named list of [`AnvilTensor`] objects
#'
#' @details
#' These functions wrap the safetensors format functionality provided by the
#' \CRANpkg{safetensors} package.
#'
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(array(1:6, dim = c(2, 3)))
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' reloaded <- nv_unserialize(raw_data)
#' reloaded
nv_write <- function(tensors, path) {
  checkmate::assert_list(tensors, names = "unique", types = "AnvilTensor")
  checkmate::assert_string(path)

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  nv_serialize(tensors, con = con)
  invisible(NULL)
}

#' @rdname nv_serialization
#' @export
nv_read <- function(path, device = NULL) {
  checkmate::assert_string(path)
  checkmate::assert_file_exists(path)
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  nv_unserialize(con, device = device)
}

#' @rdname nv_serialization
#' @export
nv_serialize <- function(tensors, con = NULL) {
  checkmate::assert_list(tensors, names = "unique", types = "AnvilTensor")

  # Extract ambiguity information to store in metadata
  ambiguity_info <- lapply(tensors, function(t) if (ambiguous(t)) TRUE else FALSE)
  names(ambiguity_info) <- names(tensors)

  # Serialize using base R serialize() as raw bytes, encoded as hex string
  raw_bytes <- serialize(ambiguity_info, connection = NULL)
  hex_string <- paste(sprintf("%02x", as.integer(raw_bytes)), collapse = "")
  metadata <- list(`__ambiguity_info__` = hex_string)

  # Unwrap AnvilTensors to get underlying PJRTBuffers for safetensors
  tensors_unwrapped <- lapply(tensors, unwrap_if_tensor)

  if (is.null(con)) {
    safetensors::safe_serialize(tensors_unwrapped, metadata = metadata)
  } else {
    safetensors::safe_save_file(tensors_unwrapped, con, metadata = metadata)
  }
}

#' @rdname nv_serialization
#' @export
nv_unserialize <- function(con, device = NULL) {
  result <- safetensors::safe_load_file(con, framework = "pjrt", device = device)

  # Extract metadata to restore ambiguity information
  # Metadata is nested under __metadata__ key
  metadata_attr <- attr(result, "metadata", exact = TRUE)
  metadata <- if (!is.null(metadata_attr)) metadata_attr$`__metadata__` else NULL

  # Parse ambiguity info using base R (unserialize from hex string)
  ambiguity_info <- if (!is.null(metadata) && !is.null(metadata$`__ambiguity_info__`)) {
    hex_string <- metadata$`__ambiguity_info__`
    # Convert hex string back to raw bytes
    hex_pairs <- strsplit(hex_string, "(?<=..)", perl = TRUE)[[1]]
    raw_bytes <- as.raw(strtoi(hex_pairs, base = 16L))
    unserialize(raw_bytes)
  } else {
    NULL
  }

  # Wrap each tensor with correct ambiguity
  result_wrapped <- lapply(names(result), function(name) {
    tensor <- result[[name]]
    # Check if there's ambiguity metadata for this tensor
    is_ambiguous <- if (!is.null(ambiguity_info) && !is.null(ambiguity_info[[name]])) {
      isTRUE(ambiguity_info[[name]])
    } else {
      FALSE
    }
    ensure_nv_tensor(tensor, ambiguous = is_ambiguous)
  })
  names(result_wrapped) <- names(result)
  result_wrapped
}
