eval_graph_pjrt <- function(graph, ...) {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  args <- list(...)

  args_nv <- lapply(args, function(x) {
    if (inherits(x, "AnvilTensor")) {
      return(x)
    }
    dt <- if (is.logical(x)) "pred" else "f32"
    if (is.null(dim(x))) {
      if (length(x) == 1L) {
        nv_scalar(x, dtype = dt)
      } else {
        nv_tensor(x, dtype = dt, shape = c(length(x)))
      }
    } else {
      nv_tensor(x, dtype = dt, shape = dim(x))
    }
  })

  out <- stablehlo(graph)
  func <- out[[1L]]
  constants <- out[[2L]]

  const_tensors <- lapply(constants, function(const) {
    if (!is_concrete_tensor(const@aval)) {
      cli::cli_abort("Internal error: non-concrete constant in graph")
    }
    const@aval@data
  })

  src <- stablehlo::repr(func)
  program <- pjrt::pjrt_program(src = src, format = "mlir")
  exec <- pjrt::pjrt_compile(program)
  out_vals <- rlang::exec(pjrt::pjrt_execute, exec, !!!const_tensors, !!!args_nv, simplify = FALSE)
  out_vals <- lapply(out_vals, nv_tensor)
  out_nv <- unflatten(graph@out_tree, out_vals)

  as_array(out_nv)
}

