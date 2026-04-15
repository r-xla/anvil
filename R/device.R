#' @title Create a Device
#' @description
#' Constructs a backend-specific device object.
#'
#' A device identifies a compute resources, such as CPU, or a specific GPU.
#' It is relevant for data allocation (e.g. via [nv_array()]) but also compilation ([jit]).
#'
#' @param x (`character(1)`)\cr
#'   Identifier for the device.
#'   E.g. `"cpu"`, `"cuda"`, or `"cuda:<n>"` (for the n-th GPU).
#' @param backend (`character(1)`)\cr
#'   The backend for which to create the device.
#' @return A backend-specific device object (e.g. `PJRTDevice` for `"xla"`,
#'   [`quickr_device`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvilBackend()`].
#' @examplesIf pjrt::plugin_is_downloaded()
#' # Create CPU device for xla backend:
#' nv_device("cpu", "xla")
#' # Create CPU device for quickr backend:
#' nv_device("cpu", "quickr")
#' @export
nv_device <- function(x, backend = default_backend()) {
  backend <- assert_backend(backend)
  globals$backends[[backend]]$new_device(x)
}
