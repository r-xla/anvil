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
#'   A connection object to read serialized tensors from.
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
  safetensors::safe_save_file(tensors, path)
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
nv_serialize <- function(tensors) {
  checkmate::assert_list(tensors, names = "unique", types = "AnvilTensor")
  safetensors::safe_serialize(tensors)
}

#' @rdname nv_serialization
#' @export
nv_unserialize <- function(con, device = NULL) {
  x <- safetensors::safe_load_file(con, framework = "pjrt", device = device)
  lapply(x, ensure_nv_tensor)
}
