#' @title Save tensors to a file
#'
#' @description
#' Saves a named list of tensors to a file in the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' This is a convenience wrapper around [`nv_serialize()`] that opens and closes
#' a file connection.
#'
#' @param tensors (named `list` of [`AnvilTensor`])\cr
#'   Named list of tensors to save. Names must be unique.
#' @param path (`character(1)`)\cr
#'   File path to write to.
#'
#' @returns `NULL` (invisibly).
#' @seealso [nv_read()], [nv_serialize()], [nv_unserialize()]
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(array(1:6, dim = c(2, 3)))
#' x
#' path <- tempfile(fileext = ".safetensors")
#' nv_save(list(x = x), path)
#' nv_read(path)
nv_save <- function(tensors, path) {
  checkmate::assert_list(tensors, names = "unique", types = "AnvilTensor")
  checkmate::assert_string(path)

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  nv_serialize(tensors, con = con)
  invisible(NULL)
}

#' @title Read tensors from a file
#'
#' @description
#' Loads tensors from a file in the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' This is a convenience wrapper around [`nv_unserialize()`] that opens and
#' closes a file connection.
#'
#' @param path (`character(1)`)\cr
#'   Path to the safetensors file.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device on which to place the loaded tensors (`"cpu"`, `"cuda"`, ...).
#'   Default is to use the CPU.
#'
#' @returns Named `list` of [`AnvilTensor`] objects.
#' @seealso [nv_save()], [nv_serialize()], [nv_unserialize()]
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(array(1:6, dim = c(2, 3)))
#' x
#' path <- tempfile(fileext = ".safetensors")
#' nv_save(list(x = x), path)
#' nv_read(path)
nv_read <- function(path, device = NULL) {
  checkmate::assert_string(path)
  checkmate::assert_file_exists(path)
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  nv_unserialize(con, device = device)
}

#' @title Serialize tensors to raw bytes
#'
#' @description
#' Serializes a named list of tensors into the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' The ambiguity of the tensors is stored in the metadata and preserved in write-read roundtrips.
#'
#' @param tensors (named `list` of [`AnvilTensor`])\cr
#'   Named list of tensors to serialize. Names must be unique.
#' @param con (`NULL` | connection)\cr
#'   An optional connection to write to.
#'   If `NULL` (default), a raw vector is returned.
#'
#' @returns A [`raw`] vector if `con` is `NULL`, otherwise `NULL` (invisibly).
#' @seealso [nv_unserialize()], [nv_save()], [nv_read()]
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(array(1:6, dim = c(2, 3)))
#' x
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' nv_unserialize(raw_data)
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

#' @title Deserialize tensors from raw bytes
#'
#' @description
#' Deserializes tensors from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' The data type, shape, and [ambiguity][ambiguous()] of each tensor are
#' restored from the serialized data.
#'
#' @param con (connection | [`raw`])\cr
#'   A connection or raw vector to read from.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device on which to place the loaded tensors (`"cpu"`, `"cuda"`, ...).
#'   Default is to use the CPU.
#'
#' @returns Named `list` of [`AnvilTensor`] objects.
#' @seealso [nv_serialize()], [nv_save()], [nv_read()]
#' @export
#' @examplesIf pjrt::plugin_is_downloaded("cpu")
#' x <- nv_tensor(array(1:6, dim = c(2, 3)))
#' x
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' nv_unserialize(raw_data)
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
