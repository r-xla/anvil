#' @include graph-to-quickr-r.R
NULL

quickr_decode_leaf <- function(seg, shape, dtype) {
  base <- if (dtype %in% c("pred", "i1")) {
    seg != 0
  } else if (dtype == "i32") {
    as.integer(seg)
  } else {
    as.double(seg)
  }

  if (!length(shape)) {
    return(base[[1L]])
  }
  if (length(shape) == 2L) {
    return(matrix(base, nrow = shape[[1L]], ncol = shape[[2L]]))
  }
  array(base, dim = shape)
}

graph_to_quickr_prepare <- function(graph) {
  in_tree <- graph$in_tree
  needs_flatten <- inherits(in_tree, "ListNode") && any(vapply(in_tree$nodes, inherits, logical(1L), "ListNode"))

  needs_pack <- !(inherits(graph$out_tree, "LeafNode") && length(graph$outputs) == 1L)
  out_infos <- lapply(graph$outputs, function(node) {
    list(dtype = as.character(dtype(node)), shape = shape(node))
  })
  is_static_flat <- graph$is_static_flat
  has_static <- !is.null(is_static_flat) && isTRUE(any(is_static_flat))

  needs_wrapper <- isTRUE(needs_pack) || length(graph$constants) || isTRUE(needs_flatten) || isTRUE(has_static)

  r_fun <- graph_to_quickr_r_fun_impl(graph, include_declare = TRUE, pack_output = needs_pack) # nolint

  list(
    r_fun = r_fun,
    needs_flatten = needs_flatten,
    needs_pack = needs_pack,
    out_infos = out_infos,
    needs_wrapper = needs_wrapper
  )
}

graph_to_quickr_make_wrapper <- function(
  graph,
  r_fun,
  inner_fun,
  needs_pack,
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

  out_lens <- vapply(
    out_infos,
    function(info) {
      if (!length(info$shape)) 1L else Reduce(`*`, as.integer(info$shape), init = 1L)
    },
    integer(1L)
  )

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
  wrapper_env$out_lens <- out_lens
  wrapper_env$needs_pack <- needs_pack
  wrapper_env$leaf_arg_names <- leaf_arg_names
  wrapper_env$use_in_tree_formals <- use_in_tree_formals
  wrapper_env$top_names <- if (isTRUE(use_in_tree_formals)) top_names else NULL
  wrapper_env$is_static_flat <- is_static_flat
  wrapper_env$const_args <- const_args
  wrapper_env$decode_leaf <- quickr_decode_leaf

  body(wrapper) <- quote({
    if (isTRUE(use_in_tree_formals)) {
      args_top <- mget(top_names, envir = environment(), inherits = FALSE)
      args <- flatten(args_top)
      if (!is.null(is_static_flat)) {
        if (length(args) != length(is_static_flat)) {
          cli_abort("Expected {length(is_static_flat)} flattened inputs, got {length(args)}")
        }
        args <- args[!is_static_flat]
      }
      args <- stats::setNames(args, leaf_arg_names)
    } else if (!length(leaf_arg_names)) {
      args <- list()
    } else {
      args <- mget(leaf_arg_names, envir = environment(), inherits = FALSE)
    }

    packed <- do.call(inner, c(const_args, args))

    if (!isTRUE(needs_pack)) {
      return(packed)
    }

    leaves <- vector("list", length(out_infos))
    pos <- 0L
    for (i in seq_along(out_infos)) {
      len <- out_lens[[i]]
      seg <- packed[pos + seq_len(len)]
      pos <- pos + len
      leaves[[i]] <- decode_leaf(seg, out_infos[[i]]$shape, out_infos[[i]]$dtype)
    }

    unflatten(out_tree, leaves)
  })

  environment(wrapper) <- wrapper_env
  wrapper
}

#' Convert an AnvilGraph to a quickr-compatible R function
#'
#' Lowers a supported subset of `AnvilGraph` objects to a plain R function (no
#' compilation) suitable for `quickr::quick()`. The returned function expects
#' plain R scalars/vectors/arrays and returns plain R values/arrays.
#'
#' @param graph ([`AnvilGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
graph_to_quickr_r_function <- function(graph) {
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
    needs_pack = prep$needs_pack,
    out_infos = prep$out_infos,
    needs_flatten = prep$needs_flatten
  )
}

#' Convert an AnvilGraph to a quickr-compiled function
#'
#' Lowers a supported subset of `AnvilGraph` objects to a plain R function and
#' compiles it with `quickr::quick()`.
#'
#' The returned function expects plain R scalars/vectors/arrays (not
#' [`AnvilTensor`]) and returns plain R values/arrays.
#'
#' If the graph returns multiple outputs (e.g. a nested list), the compiled
#' function returns the same structure by packing/unpacking values for `quickr`.
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
#' @param graph ([`AnvilGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @export
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
    return(inner_quick)
  }

  graph_to_quickr_make_wrapper(
    graph = graph,
    r_fun = prep$r_fun,
    inner_fun = inner_quick,
    needs_pack = prep$needs_pack,
    out_infos = prep$out_infos,
    needs_flatten = prep$needs_flatten
  )
}
