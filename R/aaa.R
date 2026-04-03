#' @keywords internal
NULL
"_PACKAGE"

## usethis namespace: start
#' @importFrom stablehlo repr Shape FuncId Func FuncValue
#' @importFrom stablehlo local_func hlo_input hlo_return hlo_tensor hlo_scalar
#' @importFrom stablehlo TensorType
#' @import checkmate
#' @import tengen
#' @importFrom pjrt pjrt_buffer pjrt_scalar pjrt_execute pjrt_compile pjrt_program elt_type
#' @importFrom utils gethash hashtab maphash numhash
#' @importFrom xlamisc seq_len0 seq_along0
#' @importFrom utils head tail
#' @importFrom cli cli_abort
#' @importFrom methods Math2 formalArgs
#' @importFrom utils capture.output
## usethis namespace: end
NULL

globals <- new.env()
globals$nv_types <- "AnvilArray"
globals$interpretation_rules <- c("stablehlo", "quickr", "reverse")
globals[["DESCRIPTOR_STASH"]] <- list()
globals[["CURRENT_DESCRIPTOR"]] <- NULL
backend_config_fields <- c(
  "constructor",
  "dtype",
  "shape",
  "ambiguous",
  "as_array",
  "as_raw",
  "platform",
  "device",
  "print_data"
)

#' Create a backend configuration
#'
#' @param ... Named functions implementing the backend interface.
#'   Required fields: `constructor`, `dtype`, `shape`, `ambiguous`,
#'   `as_array`, `as_raw`, `platform`, `device`, `print_data`.
#' @return A `BackendConfig` object.
#' @keywords internal
BackendConfig <- function(...) {
  config <- list(...)
  missing <- setdiff(backend_config_fields, names(config))
  if (length(missing)) {
    cli_abort("Backend config is missing required fields: {.val {missing}}")
  }
  extra <- setdiff(names(config), backend_config_fields)
  if (length(extra)) {
    cli_abort("Backend config has unknown fields: {.val {extra}}")
  }
  structure(config, class = "BackendConfig")
}

QuickrDeviceCpu <- function() {
  structure("cpu", class = "QuickrDeviceCpu")
}

#' @export
format.QuickrDeviceCpu <- function(x, ...) "QuickrDeviceCpu"

#' @export
print.QuickrDeviceCpu <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

PlainDeviceCpu <- function() {
  structure("cpu", class = "PlainDeviceCpu")
}

#' @export
format.PlainDeviceCpu <- function(x, ...) "PlainDeviceCpu"

#' @export
print.PlainDeviceCpu <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

globals$backends <- list(
  xla = BackendConfig(
    constructor = function(data, dtype, shape, device, ambiguous) {
      if (is.null(dtype) && !inherits(data, "PJRTBuffer")) {
        dtype <- as.character(default_dtype(data))
      }
      buf <- pjrt_buffer(data, dtype, device = device, shape = shape)
      structure(
        list(data = buf, ambiguous = ambiguous, backend = "xla"),
        class = "AnvilArray"
      )
    },
    dtype = function(x) as_dtype(as.character(pjrt::elt_type(x$data))),
    shape = function(x) tengen::shape(x$data),
    ambiguous = function(x) x$ambiguous,
    as_array = function(x) tengen::as_array(x$data),
    as_raw = function(x, row_major) tengen::as_raw(x$data, row_major = row_major),
    platform = function(x) pjrt::platform(x$data),
    device = function(x) device(x$data),
    print_data = function(x, footer) print(x$data, header = FALSE, footer = footer)
  ),
  quickr = BackendConfig(
    constructor = function(data, dtype, shape, device, ambiguous) {
      if (is.null(dtype)) {
        dtype <- default_dtype(data)
      }
      if (is_dtype(dtype)) {
        dtype <- as.character(dtype)
      }
      if (is.null(shape)) {
        shape <- if (!is.null(dim(data))) {
          as.integer(dim(data))
        } else if (length(data) == 1L) {
          1L
        } else {
          as.integer(length(data))
        }
      }
      data <- switch(
        substr(dtype, 1, 1),
        "f" = as.double(data),
        "i" = ,
        "u" = as.integer(data),
        "b" = as.logical(data),
        as.double(data)
      )
      if (length(shape) >= 1L) {
        dim(data) <- shape
      }
      structure(
        list(data = data, dtype = as_dtype(dtype), shape = shape, ambiguous = ambiguous, backend = "quickr"),
        class = "AnvilArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
    as_array = function(x) x$data,
    as_raw = function(x, row_major) as.raw(x$data),
    platform = function(x) "cpu",
    device = function(x) QuickrDeviceCpu(),
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    }
  ),
  plain = BackendConfig(
    constructor = function(data, dtype, shape, device, ambiguous) {
      if (is.null(dtype)) {
        dtype <- as.character(default_dtype(data))
      }
      if (is.null(shape)) {
        shape <- if (!is.null(dim(data))) {
          as.integer(dim(data))
        } else if (length(data) == 1L) {
          1L
        } else {
          as.integer(length(data))
        }
      }
      data <- switch(
        substr(dtype, 1, 1),
        "f" = as.double(data),
        "i" = ,
        "u" = as.integer(data),
        "b" = as.logical(data),
        as.double(data)
      )
      structure(
        list(data = data, dtype = as_dtype(dtype), shape = shape, ambiguous = ambiguous, backend = "plain"),
        class = "AnvilArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
    as_array = function(x) x$data,
    as_raw = function(x, row_major) cli_abort("as_raw not supported for plain backend"),
    platform = function(x) "cpu",
    device = function(x) PlainDeviceCpu(),
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    }
  )
)
utils::globalVariables(c("globals"))

normalize_backend <- function(backend) {
  assert_string(backend)
  backend <- tolower(backend)
  assert_choice(backend, names(globals$backends))
  backend
}

current_backend <- function() {
  desc <- .current_descriptor(silent = TRUE)
  if (!is.null(desc) && !is.null(desc$backend)) {
    return(desc$backend)
  }
  normalize_backend(getOption("anvil.default_backend", "xla"))
}

#' Temporarily set the default backend
#'
#' Sets the `anvil.default_backend` option for the duration of the
#' calling scope. This affects `nv_array()`, `nv_scalar()`, and `jit()`.
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use (`"xla"` or `"quickr"`).
#' @param envir The environment to scope the change to.
#' @return The previous value of the option (invisibly).
#' @export
local_backend <- function(backend, envir = parent.frame()) {
  backend <- normalize_backend(backend)
  withr::local_options(anvil.default_backend = backend, .local_envir = envir)
}
