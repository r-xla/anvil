#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar `i32` seed value.
#' @template param_device
#' @return [`nv_array`] of dtype `ui64` and shape `(2)`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' state
#' @export
nv_rng_state <- function(seed, device = default_device()) {
  seed <- nv_array(seed, dtype = as_dtype("i32"), shape = integer(), device = device)
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}

#' @title Prepare Inputs of an API Function
#' @description
#' Convenience wrapper for the beginning of an `nv_*` API function. It:
#'
#' 1. Infers a common device from any concrete [`AnvilArray`] input and
#'    converts [`arrayish`] R inputs (length-1 vectors, R arrays) onto it.
#'    If none of the inputs is concrete, the R inputs are left as-is and
#'    placed on the default device when they eventually hit [`nv_array()`].
#' 2. Ensures that all concrete [`AnvilArray`] inputs share the same device
#'    (and therefore the same backend). Cross-device inputs cause an error.
#'
#' During tracing (inside [`jit()`]) the conversion in step 1 is skipped
#' because inputs are abstract.
#' @param ... ([`arrayish`])\cr
#'   Inputs to prepare.
#' @return (`list`)\cr
#'   The inputs in the same order, each converted to an [`AnvilArray`] (or
#'   left as-is for tracing boxes / abstract literals).
#' @examplesIf pjrt::plugins_downloaded()
#' nv_align_inputs(nv_array(1:3), 1L)
#' @export
nv_align_inputs <- function(...) {
  args <- list(...)
  # In eager mode we place R literals on the device inferred from concrete
  # inputs. In jit/tracing mode, device placement is handled at the input/output
  # boundary and when boxing constants, so we can short-circuit here.
  if (currently_tracing()) {
    return(args)
  }

  devices <- list()
  for (a in args) {
    if (is_anvil_array(a) && backend(a) != "plain") {
      devices[[length(devices) + 1L]] <- device(a)
    }
  }

  dev <- if (length(devices) == 0L) {
    default_device()
  } else {
    first_dev <- devices[[1L]]
    for (d in devices[-1L]) {
      if (eq_device(first_dev, d)) {
        next
      }
      if (backend(first_dev) != backend(d)) {
        cli_abort(c(
          "Found inputs from multiple backends",
          i = "Found backends {.val {backend(first_dev)}} and {.val {backend(d)}}"
        ))
      }
      cli_abort(c(
        "Found inputs living on multiple devices, which is currently not supported",
        i = "Found devices {.val {as.character(first_dev)}} and {.val {as.character(d)}}"
      ))
    }
    first_dev
  }

  lapply(args, function(a) {
    if (is_anvil_array(a)) {
      return(a)
    }
    if (is_valid_r(a)) {
      return(as_anvil_array(a, device = dev))
    }
    a
  })
}

# Convert an R-valid value (scalar numeric/logical, or R array of those) into
# an [`AnvilArray`] on the given device. AnvilArrays are returned unchanged
# (with a consistency check on the device, performed by [`nv_array()`]).
as_anvil_array <- function(x, device = NULL) {
  if (is_valid_lit(x)) {
    nv_scalar(x, device = device)
  } else if (is_valid_array(x)) {
    nv_array(x, device = device)
  } else {
    nv_array(x, device = device)
  }
}
