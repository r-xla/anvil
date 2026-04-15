#' @include backend.R
NULL

jit_xla_inputs <- function(args_flat, is_static_flat, device) {
  device_strs <- character()
  first_device <- NULL
  avals_in <- Map(
    function(x, is_static) {
      if (is_static) {
        return(x)
      }
      if (!is_anvil_array(x)) {
        cli_abort("Expected AnvilArray, but got {.cls {class(x)[1]}}")
      }
      if (backend(x) != "xla") {
        cli_abort("Expected {.val xla} backend, but got {.val {backend(x)}} backend.")
      }
      dev <- device(x)
      device_strs <<- c(device_strs, as.character(dev))
      if (is.null(first_device)) {
        first_device <<- dev
      }
      nv_abstract(dtype(x), shape(x), ambiguous = ambiguous(x))
    },
    args_flat,
    is_static_flat
  )

  unique_strs <- unique(device_strs)
  if (length(unique_strs) > 1) {
    cli_abort("Inputs live on different devices: {.val {unique_strs}}.")
  }

  inferred_device <- if (!is.null(first_device)) {
    first_device
  } else if (!is.null(device)) {
    device
  } else {
    NULL
  }

  list(avals_in = avals_in, device = inferred_device)
}

jit_call_xla <- function(exec, out_node, consts_flat, args_flat, is_static_flat, ambiguous_out = NULL) {
  args_nonstatic <- args_flat[!is_static_flat]
  args_unwrapped <- lapply(args_nonstatic, \(a) a$data)
  out_vals <- rlang::exec(
    pjrt::pjrt_execute,
    exec,
    !!!consts_flat,
    !!!args_unwrapped,
    simplify = FALSE
  )
  jit_wrap_outputs(out_vals, out_node, ambiguous_out, "xla")
}

jit_xla_impl <- function(f, static, cache, donate, device) {
  function() {
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static, "xla")
    inputs <- jit_xla_inputs(prep$args_flat, prep$is_static_flat, device)

    device_key <- if (!is.null(inputs$device)) as.character(inputs$device) else NULL
    cache_key <- list(prep$in_tree, inputs$avals_in, device_key)
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(jit_call_xla(
        cache_hit[[1]], # executable
        cache_hit[[2]], # out tree
        cache_hit[[3]], # constants
        prep$args_flat,
        prep$is_static_flat,
        cache_hit[[4]] # ambiguity
      ))
    }

    compiled <- compile_to_xla(
      f,
      args_flat = inputs$avals_in,
      in_tree = prep$in_tree,
      donate = donate,
      device = inputs$device
    )
    cache$set(
      cache_key,
      list(compiled$exec, compiled$out_tree, compiled$const_arrays, compiled$ambiguous_out)
    )

    jit_call_xla(
      compiled$exec,
      compiled$out_tree,
      compiled$const_arrays,
      prep$args_flat,
      prep$is_static_flat,
      compiled$ambiguous_out
    )
  }
}

#' @title Trace, lower, and compile a function to an XLA executable
#' @description
#' Takes a function, traces it into a computational graph, lowers it to StableHLO,
#' and compiles it to a PJRT executable. Returns the compiled executable along with
#' metadata needed for execution.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param args_flat (`list`)\cr
#'   Flat list of abstract input values.
#' @param in_tree (`Node`)\cr
#'   Tree structure of the inputs.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`NULL` | `character(1)`)\cr
#'   Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from traced arrays.
#' @return A `list` with elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_arrays`: Constants needed at execution time.
#'   - `ambiguous_out`: Logical vector indicating which outputs are ambiguous (`NULL` if none are).
#' @keywords internal
compile_to_xla <- function(f, args_flat, in_tree, donate = character(), device = NULL) {
  desc <- local_descriptor()
  graph <- trace_fn(f, desc = desc, toplevel = TRUE, args_flat = args_flat, in_tree = in_tree)

  # FIXME: This should also respect the devices of args_flat
  traced_devices <- unique(vapply(desc$devices, as.character, character(1)))
  if (length(traced_devices) > 1) {
    cli_abort("Tensors live on different devices: {traced_devices}.")
  }

  if (!is.null(device) && length(traced_devices) && traced_devices[[1L]] != pjrt::as_pjrt_device(device)) {
    cli_abort(
      "Provided device {.val {device}} does not match traced device {.val {traced_devices}}."
    )
  }
  device <- if (!is.null(device)) {
    pjrt::as_pjrt_device(device)
  } else if (length(traced_devices)) {
    desc$devices[[1L]]
  } else {
    pjrt::as_pjrt_device(Sys.getenv("PJRT_PLATFORM", "cpu"))
  }

  compile_graph_to_xla(graph, donate = donate, device = device)
}

compile_graph_to_xla <- function(graph, donate = character(), device = NULL) {
  device <- if (!is.null(device)) {
    pjrt::as_pjrt_device(device)
  } else {
    pjrt::as_pjrt_device(Sys.getenv("PJRT_PLATFORM", "cpu"))
  }

  graph <- inline_scalarish_constants(graph)
  graph <- remove_unused_constants(graph)

  out <- stablehlo(graph, donate = donate)
  func <- out[[1L]]
  constants <- out[[2L]]

  const_arrays <- lapply(constants, \(const) {
    if (!is_concrete_tensor(const$aval)) {
      cli_abort("Internal error: Not all constants are concrete arrays")
    }
    arr <- const$aval$data
    if (backend(arr) == "plain") {
      pjrt_buffer(as_array(arr), dtype = dtype(arr), device = device, shape = shape(arr))
    } else {
      unwrap_if_array(arr)
    }
  })

  out_tree <- graph$out_tree

  ambiguous_out <- vapply(graph$outputs, \(x) x$aval$ambiguous, logical(1))
  if (!any(ambiguous_out)) {
    ambiguous_out <- NULL
  }

  src <- stablehlo::repr(func)
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, device = device)

  list(exec = exec, out_tree = out_tree, const_arrays = const_arrays, ambiguous_out = ambiguous_out)
}

#' @title Ahead-of-time compile a function to XLA
#' @description
#' Compiles a function to an XLA executable via tracing.
#'
#' Returns a callable R function that executes the compiled binary.
#' Unlike [`jit()`], compilation happens eagerly at
#' definition time rather than on first call, so the input shapes and dtypes must be
#' specified upfront via abstract arrays (see [`nv_abstract()`]).
#' @details
#' Traces `f` with the given abstract `args` (via [`trace_fn()`]), lowers the resulting graph
#' via [`stablehlo()`] and then compiles it to an XLA executable via [`pjrt::pjrt_compile()`].
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvilArray`]s.
#' @param args (`list`)\cr
#'   List of abstract array specifications (e.g. from [`nv_abstract()`]) describing the
#'   expected shapes and dtypes of `f`'s arguments.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`character(1)` | `PJRTDevice`)\cr
#'   Target device such as `"cpu"` (default) or `"cuda"`.
#' @return (`function`)\cr
#'   A function that accepts [`AnvilArray`] arguments (matching the flat inputs)
#'   and returns the result as [`AnvilArray`]s.
#' @seealso [`jit()`] for lazy compilation, [`compile_to_xla()`] for the lower-level API.
#' @export
#' @examplesIf pjrt::plugin_is_downloaded()
#' f_compiled <- xla(function(x, y) x + y,
#'   args = list(x = nv_abstract("f32", c(2, 2)), y = nv_abstract("f32", c(2, 2)))
#' )
#' a <- nv_array(array(1:4, c(2, 2)), dtype = "f32")
#' b <- nv_array(array(5:8, c(2, 2)), dtype = "f32")
#' f_compiled(a, b)
xla <- function(f, args, donate = character(), device = NULL) {
  # FIXME: Also use device inference from trace_fn
  device <- device %||% Sys.getenv("PJRT_PLATFORM", "cpu")
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  compiled <- compile_to_xla(f, args_flat = args_flat, in_tree = in_tree, donate = donate, device = device)
  exec <- compiled$exec
  out_tree <- compiled$out_tree
  const_arrays <- compiled$const_arrays
  ambiguous_out <- compiled$ambiguous_out

  f_xla <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    args_flat <- flatten(args)
    args_flat <- lapply(args_flat, autoconvert_input, backend = "xla")
    args_unwrapped <- lapply(args_flat, \(a) a$data)
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!const_arrays,
      !!!args_unwrapped,
      simplify = FALSE
    )
    jit_wrap_outputs(out_vals, out_tree, ambiguous_out, "xla")
  }
  formals(f_xla) <- formals2(f)
  f_xla
}

#' XLA backend
#'
#' Constructs the XLA backend, which stores array data in PJRT buffers (via
#' [`pjrt::pjrt_buffer()`]) and compiles jitted functions to XLA executables
#' via [`stablehlo()`] and [`pjrt::pjrt_compile()`]. This is the default
#' backend.
#'
#' @section Data representation:
#' An [`AnvilArray`] with `backend = "xla"` wraps a [`pjrt::pjrt_buffer()`]
#' stored in the `$data` field. The buffer owns the memory holding the tensor
#' values and may live on any device supported by PJRT (CPU, CUDA, Metal,
#' ...). Calling [`as_array()`] transfers the buffer contents back to an R
#' array; calling [`nv_array()`] on an R object uploads it to the requested
#' device.
#'
#' Each `AnvilArray` therefore has an associated device, queryable via
#' [`device()`]. A device is a [`pjrt::as_pjrt_device()`] object (e.g. the
#' platform `"cpu"` or `"cuda"`, optionally with an index such as `"cuda:1"`).
#' When `device` is `NULL` in [`nv_array()`] or the [`jit()`] wrapper, the
#' device defaults to the `PJRT_PLATFORM` environment variable (falling back
#' to `"cpu"`), or is inferred from the existing inputs of a jitted call.
#' Operations require all inputs to live on the same device.
#'
#' @section XLA JIT arguments:
#' * `donate` (`character()`, default `character()`): names of arguments whose
#'   underlying buffers may be donated to (i.e., reused/consumed by) the
#'   compiled XLA executable. Donated buffers must not be used again by the
#'   caller after the call; this can reduce memory usage and copies for large
#'   inputs. Must not overlap with `static`.
#' * `device` (`NULL` | `character(1)` | [`pjrt::PJRTDevice`][pjrt::as_pjrt_device],
#'   default `NULL`): target device (e.g. `"cpu"`, `"cuda"`) on which the
#'   function is compiled and executed. When `NULL`, the device is inferred
#'   from the inputs; if inputs live on different devices an error is raised.
#'
#' @return An [`AnvilBackend`] object with subclass `"AnvilBackendXla"`.
#' @seealso [`AnvilBackend()`], [`AnvilBackendQuickr()`], [`local_backend()`], [`jit()`].
#' @export
AnvilBackendXla <- function() {
  backend <- AnvilBackend(
    data_constructor = function(data, dtype, shape, device, ambiguous) {
      buf <- pjrt_buffer(data, dtype = dtype, device = device, shape = shape)
      structure(
        list(data = buf, ambiguous = ambiguous, backend = "xla"),
        class = "AnvilArray"
      )
    },
    dtype = function(x) tengen::dtype(x$data),
    shape = function(x) tengen::shape(x$data),
    ambiguous = function(x) x$ambiguous,
    as_array = function(x) tengen::as_array(x$data),
    as_raw = function(x, row_major) tengen::as_raw(x$data, row_major = row_major),
    platform = function(x) pjrt::platform(x$data),
    device = function(x) device(x$data),
    new_device = function(x) pjrt::pjrt_device(x),
    print_data = function(x, footer) print(x$data, header = FALSE, footer = footer),
    jit = function(f, static, cache, donate = character(), device = NULL) {
      assert_subset(donate, formalArgs2(f))
      # static is checked in jit() itself
      common <- intersect(donate, static)
      if (length(common)) {
        cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
      }
      jit_xla_impl(f, static, cache, donate, device)
    }
  )
  class(backend) <- c("AnvilBackendXla", class(backend))
  backend
}

register_backend("xla", AnvilBackendXla())
