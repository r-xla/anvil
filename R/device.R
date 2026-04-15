#' @title Create a Device
#' @description
#' Constructs a device object.
#' There are different ways to specify this:
#' 1. ``
#' Constructs a backend-specific device object for a given device type
#' (e.g. `"cpu"`, `"cuda"`). The returned device can be passed to array
#' constructors like [nv_fill()] or [nv_iota()] to control where the
#' resulting array is allocated and which backend runs the operation.
#' @param type (`character(1)`)\cr
#'   Device type, e.g. `"cpu"` or `"cuda"`. Supported values depend on the
#'   selected backend.
#' @param backend (`character(1)`)\cr
#'   Backend that produces the device. Defaults to [`default_backend()`].
#' @return A backend-specific device object (e.g. `PJRTDevice` for `"xla"`,
#'   [`QuickrDevice`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvilBackend()`].
#' @examplesIf pjrt::plugin_is_downloaded()
#' # Create CPU device for XLA backend:
#' nv_device("xla:cpu")
#' nv_device("xla:1")
#' @export
nv_device <- function(type, backend = default_backend()) {
  #TODO: This should probably be more like AnvilArray (?)
  backend <- assert_backend(backend)
  globals$backends[[backend]]$new_device(type)
}
