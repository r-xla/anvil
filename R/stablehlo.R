# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

#' @title Lower a function to StableHLO
#' @description
#' Immediately lower a flattened function to a StableHLO Func object.
#' @param f (`function` | `Gr`)\cr
#'   Flattened function to lower.
#' @return (`list`) with elements:
#'   - `func`: The StableHLO `Func` object
#'   - `out_tree`: The output tree structure
#' @export
stablehlo <- function(graph, static = character()) {
  # Node -> FuncVariable
  env <- hashtab()
  func <- stablehlo::local_func(id = "main")
  for (node in c(graph@constants, graph@inputs)) {
    vt <- st2vt(node@aval)
    id <- stablehlo::ValueId()
    fi <- stablehlo::FuncInput(id, vt)
    func@inputs@items <- c(func@inputs@items, fi)
    fvar <- stablehlo::FuncVariable(id, vt, func)
    env[[node]] <- fvar
  }

  do_call <- function(call) {
    prim <- call@primitive
    params <- call@params
    inputs <- lapply(call@inputs, \(x) env[[x]])
    if (any(vapply(inputs, is.null, logical(1L)))) {
      #browser()
    }
    fvars_out <- rlang::exec(prim[["stablehlo"]], !!!c(inputs, params))
    if (length(call@outputs) != length(fvars_out)) {
      cli_abort("Expected {length(call@outputs)} outputs, but got {length(fvars_out)}")
    }
    for (i in seq_along(fvars_out)) {
      env[[call@outputs[[i]]]] <- fvars_out[[i]]
    }
  }

  for (call in graph@calls) {
    do_call(call)
  }

  outputs <- lapply(graph@outputs, \(x) env[[x]])
  func <- do.call(stablehlo::hlo_return, outputs)

  constants <- graph@constants

  list(func, constants)
}
