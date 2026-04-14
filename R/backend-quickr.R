#' @include backend.R
NULL

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

quickr_jit_aval <- function(x) {
  if (!is_anvil_array(x)) {
    cli_abort("Expected AnvilArray, but got {.cls {class(x)[1]}}")
  }
  if (backend(x) != "quickr") {
    cli_abort("Expected {.val quickr} backend, but got {.val {backend(x)}} backend.")
  }
  nv_abstract(dtype(x), shape(x), ambiguous = ambiguous(x))
}

jit_quickr_inputs <- function(args_flat, is_static_flat) {
  avals_in <- Map(
    function(x, is_static) {
      if (is_static) {
        return(x)
      }
      quickr_jit_aval(x)
    },
    args_flat,
    is_static_flat
  )

  list(avals_in = avals_in, device = NULL)
}

jit_call_quickr <- function(fun, r_args) {
  do.call(fun, r_args)
}

jit_quickr_impl <- function(f, static, cache, unwrap) {
  function() {
    # calling a jitted function within another jitted function --> re-trace the original closure
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static)
    inputs <- jit_quickr_inputs(prep$args_flat, prep$is_static_flat)

    cache_key <- list(prep$in_tree, inputs$avals_in, inputs$device)
    r_args <- lapply(unname(prep$args), function(a) {
      if (is_anvil_array(a)) as_array(a) else a
    })
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(jit_call_quickr(cache_hit, r_args))
    }

    compiled <- compile_to_quickr(f, args_flat = inputs$avals_in, in_tree = prep$in_tree, unwrap = unwrap)
    cache$set(cache_key, compiled$fun)
    jit_call_quickr(compiled$fun, r_args)
  }
}

compile_to_quickr <- function(f, args_flat, in_tree, unwrap = FALSE) {
  desc <- local_descriptor()
  graph <- trace_fn(f, desc = desc, toplevel = TRUE, args_flat = args_flat, in_tree = in_tree)
  list(fun = graph_to_quickr_function(graph, unwrap = unwrap))
}

register_backend(
  "quickr",
  AnvilBackend(
    data_constructor = function(data, dtype, shape, device, ambiguous) {
      if (is.null(dtype)) {
        dtype <- if (is.double(data)) FloatType(64) else default_dtype(data)
      }
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
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
      dtype_chr <- as.character(dtype)
      data <- switch(
        substr(dtype_chr, 1, 1),
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
        list(data = data, dtype = dtype, shape = shape, ambiguous = ambiguous, backend = "quickr"),
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
    },
    jit = function(f, static, cache, unwrap = FALSE) {
      assert_flag(unwrap)
      jit_quickr_impl(f, static, cache, unwrap)
    }
  )
)
