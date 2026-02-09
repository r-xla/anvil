compile_graph_pjrt <- function(graph) {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  unwrap_if_tensor_for_test <- function(x) {
    if (inherits(x, "AnvilTensor")) {
      x$tensor
    } else {
      x
    }
  }

  flatten_args_for_test <- function(x) {
    if (is.list(x)) {
      if (!length(x)) {
        return(list())
      }
      Reduce(c, lapply(unname(x), flatten_args_for_test))
    } else {
      list(x)
    }
  }

  out <- stablehlo(graph)
  func <- out[[1L]]
  constants <- out[[2L]]

  const_tensors <- lapply(constants, function(const) {
    if (!is_concrete_tensor(const$aval)) {
      cli::cli_abort("Internal error: non-concrete constant in graph")
    }
    unwrap_if_tensor_for_test(const$aval$data)
  })

  src <- stablehlo::repr(func)
  program <- pjrt::pjrt_program(src = src, format = "mlir")
  exec <- pjrt::pjrt_compile(program)

  input_nodes <- graph$inputs
  out_tree <- graph$out_tree

  as_r <- function(x) {
    if (inherits(x, "AnvilTensor")) {
      return(as_array(x))
    }
    if (is.list(x)) {
      return(lapply(x, as_r))
    }
    x
  }

  function(...) {
    args <- flatten_args_for_test(list(...))
    if (length(args) != length(input_nodes)) {
      cli::cli_abort("Expected {length(input_nodes)} inputs, got {length(args)}")
    }

    args_nv <- Map(
      function(x, gval) {
        if (inherits(x, "AnvilTensor")) {
          return(x)
        }
        expected_shape <- gval$aval$shape$dims
        expected_dtype <- as.character(gval$aval$dtype)
        if (expected_dtype == "i1") {
          expected_dtype <- "pred"
        }
        if (!length(expected_shape)) {
          if (length(x) != 1L) {
            cli::cli_abort("Expected scalar input")
          }
          nv_scalar(x, dtype = expected_dtype)
        } else {
          nv_tensor(x, dtype = expected_dtype, shape = expected_shape)
        }
      },
      args,
      input_nodes
    )

    args_unwrapped <- lapply(args_nv, unwrap_if_tensor_for_test)
    out_vals <- rlang::exec(pjrt::pjrt_execute, exec, !!!const_tensors, !!!args_unwrapped, simplify = FALSE)
    out_vals <- lapply(out_vals, nv_tensor)
    out_nv <- unflatten(out_tree, out_vals)
    as_r(out_nv)
  }
}

eval_graph_pjrt <- function(graph, ...) {
  run <- compile_graph_pjrt(graph)
  run(...)
}
