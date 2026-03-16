#' @include rules-quickr.R
NULL

quickr_restore_leaf <- function(value, shape) {
  shape <- as.integer(shape)

  if (!length(shape)) {
    return(value[[1L]])
  }

  if (length(shape) == 1L) {
    value_dims <- dim(value)
    if (is.null(value_dims) || !identical(value_dims, shape)) {
      return(array(value, dim = shape))
    }
  }

  value
}

quickr_restore_output <- function(value, out_tree, out_infos) {
  if (inherits(out_tree, "LeafNode") && length(out_infos) == 1L) {
    return(quickr_restore_leaf(value, out_infos[[1L]]$shape))
  }

  if (!is.list(value) || length(value) != length(out_infos)) {
    cli_abort("Internal error: expected {length(out_infos)} quickr outputs, got {length(value)}")
  }

  leaves <- .mapply(
    function(x, info) quickr_restore_leaf(x, info$shape),
    list(value, out_infos),
    NULL
  )
  unflatten(out_tree, leaves)
}

graph_to_quickr_prepare <- function(graph) {
  in_tree <- graph$in_tree
  needs_flatten <- inherits(in_tree, "ListNode") && any(vapply(in_tree$nodes, inherits, logical(1L), "ListNode"))

  needs_output_wrapper <- !(inherits(graph$out_tree, "LeafNode") && length(graph$outputs) == 1L)
  out_infos <- lapply(graph$outputs, function(node) {
    list(dtype = as.character(dtype(node)), shape = shape(node))
  })
  is_static_flat <- graph$is_static_flat
  has_static <- !is.null(is_static_flat) && isTRUE(any(is_static_flat))

  needs_wrapper <- isTRUE(needs_output_wrapper) ||
    length(graph$constants) ||
    isTRUE(needs_flatten) ||
    isTRUE(has_static)

  r_fun <- graph_to_quickr_r_fun_impl(graph, include_declare = TRUE)

  list(
    r_fun = r_fun,
    needs_flatten = needs_flatten,
    out_infos = out_infos,
    needs_wrapper = needs_wrapper
  )
}

quickr_assert_static_args <- function(args_flat, is_static_flat, static_args_flat) {
  if (is.null(is_static_flat) || !isTRUE(any(is_static_flat))) {
    return(invisible(NULL))
  }
  if (is.null(static_args_flat)) {
    cli_abort("This graph is missing traced static argument values. Retrace the graph before lowering to quickr.")
  }

  static_runtime <- args_flat[is_static_flat]
  if (length(static_runtime) != length(static_args_flat)) {
    cli_abort(
      "Internal error: expected {length(static_args_flat)} static args, got {length(static_runtime)}"
    )
  }

  mismatch <- vapply(
    seq_along(static_args_flat),
    function(i) !identical(static_runtime[[i]], static_args_flat[[i]]),
    logical(1L)
  )
  if (any(mismatch)) {
    cli_abort(
      "Static arguments must match the values used to trace the graph. Retrace the graph for new static values."
    )
  }

  invisible(NULL)
}

graph_to_quickr_make_wrapper <- function(
  graph,
  r_fun,
  inner_fun,
  out_infos,
  needs_flatten
) {
  r_arg_names <- names(formals(r_fun)) %||% character()
  n_user <- length(graph$inputs)
  leaf_arg_names <- r_arg_names[seq_len(n_user)]

  is_static_flat <- graph$is_static_flat
  has_static <- !is.null(is_static_flat) && isTRUE(any(is_static_flat))
  use_in_tree_formals <- isTRUE(needs_flatten) || isTRUE(has_static)

  const_args <- list()
  if (length(graph$constants)) {
    const_arg_names <- r_arg_names[(n_user + 1L):(n_user + length(graph$constants))]
    const_vals <- lapply(graph$constants, function(node) {
      as_array(node$aval$data)
    })
    const_args <- stats::setNames(const_vals, const_arg_names)
  }

  wrapper <- function() {}
  if (isTRUE(use_in_tree_formals)) {
    in_tree <- graph$in_tree
    top_names <- in_tree$names %||% rep("", length(in_tree$nodes))
    top_names <- vapply(
      top_names,
      function(x) {
        if (is.null(x) || !nzchar(x)) "x" else x
      },
      character(1L)
    )
    top_names <- make.unique(make.names(top_names))
    formals(wrapper) <- as.pairlist(stats::setNames(rep(list(quote(expr = )), length(top_names)), top_names))
  } else {
    formals(wrapper) <- formals(r_fun)[seq_len(n_user)]
  }

  # Don't retain the full call frame (graph, r_fun, etc.) via parent environments.
  wrapper_env <- new.env(parent = environment(graph_to_quickr_function))
  wrapper_env$inner <- inner_fun
  wrapper_env$out_tree <- graph$out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$leaf_arg_names <- leaf_arg_names
  wrapper_env$use_in_tree_formals <- use_in_tree_formals
  wrapper_env$top_names <- if (isTRUE(use_in_tree_formals)) top_names else NULL
  wrapper_env$is_static_flat <- is_static_flat
  wrapper_env$static_args_flat <- graph$static_args_flat
  wrapper_env$const_args <- const_args
  wrapper_env$restore_output <- quickr_restore_output

  body(wrapper) <- quote({
    if (isTRUE(use_in_tree_formals)) {
      args_top <- mget(top_names, envir = environment(), inherits = FALSE)
      args <- flatten(args_top)
      if (!is.null(is_static_flat)) {
        if (length(args) != length(is_static_flat)) {
          cli_abort("Expected {length(is_static_flat)} flattened inputs, got {length(args)}")
        }
        quickr_assert_static_args(args, is_static_flat, static_args_flat)
        args <- args[!is_static_flat]
      }
      args <- stats::setNames(args, leaf_arg_names)
    } else if (!length(leaf_arg_names)) {
      args <- list()
    } else {
      args <- mget(leaf_arg_names, envir = environment(), inherits = FALSE)
    }

    value <- do.call(inner, c(const_args, args))
    restore_output(value, out_tree, out_infos)
  })

  environment(wrapper) <- wrapper_env
  wrapper
}

graph_to_quickr_make_rank1_wrapper <- function(r_fun, inner_fun, out_shape) {
  stopifnot(length(out_shape) == 1L)

  arg_names <- names(formals(r_fun)) %||% character()
  inner_call <- as.call(c(list(as.name("inner")), lapply(arg_names, as.name)))
  wrapper <- function() {}
  formals(wrapper) <- formals(r_fun)

  wrapper_env <- new.env(parent = environment(graph_to_quickr_function))
  wrapper_env$inner <- inner_fun
  wrapper_env$out_shape <- as.integer(out_shape)

  body(wrapper) <- rlang::expr(array(!!inner_call, dim = out_shape))

  environment(wrapper) <- wrapper_env
  wrapper
}

#' Convert an AnvilGraph to a plain R function
#'
#' Lowers a supported subset of `AnvilGraph` objects to a plain R function (no
#' compilation) suitable for `quickr::quick()`. The returned function expects
#' plain R scalars/vectors/arrays and returns plain R values/arrays.
#'
#' Most users will prefer [`jit()`] with `backend = "quickr"`. This function is
#' the lower-level graph API.
#'
#' @param graph ([`AnvilGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @seealso [`jit()`] with `backend = "quickr"` for tracing and compiling a
#'   regular R function in one step.
#' @export
graph_to_r_function <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvilGraph}")
  }

  prep <- graph_to_quickr_prepare(graph)
  r_fun <- prep$r_fun
  # The lowered function is intended to be runnable as plain R code as well as
  # compilable by {quickr}. When {quickr} is installed, its `declare()` can
  # modify arguments at runtime (e.g. stripping dims), which breaks plain R
  # execution. Keep the `declare(type(...))` call for compilation, but make it
  # a no-op when evaluating the lowered function in R.
  environment(r_fun)$declare <- function(...) invisible(NULL)

  if (!isTRUE(prep$needs_wrapper)) {
    return(r_fun)
  }

  graph_to_quickr_make_wrapper(
    graph = graph,
    r_fun = r_fun,
    inner_fun = r_fun,
    out_infos = prep$out_infos,
    needs_flatten = prep$needs_flatten
  )
}

graph_to_quickr_r_function <- graph_to_r_function

#' Convert an AnvilGraph to a quickr-compiled function
#'
#' Lowers a supported subset of `AnvilGraph` objects to a plain R function and
#' compiles it with `quickr::quick()`.
#'
#' The returned function expects plain R scalars/vectors/arrays (not
#' [`AnvilTensor`]) and returns plain R values/arrays.
#'
#' If the graph returns multiple outputs (e.g. a nested list), the compiled
#' function returns the same structure by rebuilding the output tree in R.
#'
#' Currently supported primitives are:
#' `fill`, `iota`, `reverse`, `concatenate`, `pad`, `gather`, `scatter`, `convert`, `add`, `sub`, `mul`, `divide`,
#' `negate`, `abs`, `sqrt`, `log`, `log1p`, `exp`, `expm1`, `logistic`, `sine`, `cosine`, `tan`, `tanh`,
#' `floor`, `ceil`, `power`, `maximum`, `minimum`, `equal`,
#' `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `and`, `or`,
#' `xor`, `not`, `select`, `broadcast_in_dim`, `dot_general`, `transpose`,
#' `reshape`, `sum`, `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `reduce_any`, `reduce_all`,
#' `static_slice`, `dynamic_slice`, `dynamic_update_slice`.
#' Higher-order primitives supported: `if`, `while`.
#'
#' Supported dtypes are `f32`, `f64`, `i32`, and `pred`.
#' The code generator currently supports tensors up to rank 5. Some primitives
#' are more restricted (e.g. `transpose` currently only handles rank-2 tensors).
#'
#' Most users will prefer [`jit()`] with `backend = "quickr"`. This function is
#' the lower-level graph API.
#'
#' @param graph ([`AnvilGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @seealso [`jit()`] with `backend = "quickr"` for tracing and compiling a
#'   regular R function in one step.
#' @keywords internal
#' @examplesIf pjrt::plugin_is_downloaded() && requireNamespace("quickr", quietly = TRUE)
#' # Simple: two scalar inputs
#' fn <- function(x, y) x + y
#'
#' graph <- trace_fn(
#'   fn,
#'   args = list(
#'     x = nv_scalar(0.0, dtype = "f64"),
#'     y = nv_scalar(0.0, dtype = "f64")
#'   )
#' )
#'
#' f_quickr <- graph_to_quickr_function(graph)
#' f_pjrt <- jit(fn)
#'
#' out_quickr <- f_quickr(0.5, 1.25)
#' out_pjrt <- as_array(f_pjrt(nv_scalar(0.5, dtype = "f64"), nv_scalar(1.25, dtype = "f64")))
#' stopifnot(isTRUE(all.equal(out_quickr, out_pjrt, tolerance = 1e-12)))
#'
#' # Nested inputs + nested outputs
#' fn2 <- function(x) {
#'   total <- x$a + x$b$u - x$b$v
#'   list(
#'     total = total,
#'     parts = list(a = x$a, b = list(u = x$b$u, v = x$b$v))
#'   )
#' }
#'
#' graph2 <- trace_fn(
#'   fn2,
#'   args = list(
#'     x = list(
#'       a = nv_scalar(0.0, dtype = "f64"),
#'       b = list(
#'         u = nv_scalar(0.0, dtype = "f64"),
#'         v = nv_scalar(0.0, dtype = "f64")
#'       )
#'     )
#'   )
#' )
#'
#' f2_quickr <- graph_to_quickr_function(graph2)
#' f2_pjrt <- jit(fn2)
#'
#' to_r <- function(x) {
#'   if (inherits(x, "AnvilTensor")) {
#'     as_array(x)
#'   } else if (is.list(x)) {
#'     lapply(x, to_r)
#'   } else {
#'     x
#'   }
#' }
#'
#' x2_r <- list(a = 0.5, b = list(u = 1.25, v = 0.75))
#' x2_nv <- rapply(x2_r, function(x) nv_scalar(x, dtype = "f64"), how = "replace")
#'
#' out2_quickr <- f2_quickr(x2_r)
#' out2_pjrt <- to_r(f2_pjrt(x2_nv))
#' stopifnot(isTRUE(all.equal(out2_quickr, out2_pjrt, tolerance = 1e-12)))
graph_to_quickr_function <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvilGraph}")
  }

  assert_quickr_installed("{.fn graph_to_quickr_function}")

  prep <- graph_to_quickr_prepare(graph)
  inner_quick <- quickr_eager_compile(prep$r_fun)

  if (!isTRUE(prep$needs_wrapper)) {
    out_shape <- as.integer(prep$out_infos[[1L]]$shape)
    if (length(out_shape) == 1L) {
      return(graph_to_quickr_make_rank1_wrapper(
        r_fun = prep$r_fun,
        inner_fun = inner_quick,
        out_shape = out_shape
      ))
    }
    return(inner_quick)
  }

  graph_to_quickr_make_wrapper(
    graph = graph,
    r_fun = prep$r_fun,
    inner_fun = inner_quick,
    out_infos = prep$out_infos,
    needs_flatten = prep$needs_flatten
  )
}
