# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

HloEnv <- S7::new_class(
  "HloEnv",
  properties = list(
    parent = NULL | new_S3_class("anvil::HloEnv"),
    gval_to_fvar = new_property(class_hashtab, default = quote(hashtab()))
  )
)

env_add <- function(env, gval, fvar) {
  env@gval_to_fvar[[gval]] <- fvar
  invisible(env)
}

env_get <- function(env, gval) {
  fvar <- env@gval_to_fvar[[gval]]
  if (!is.null(fvar)) {
    return(fvar)
  }
  parent <- env@parent
  if (!is.null(parent)) {
    return(env_get(parent, gval))
  }
  cli_abort("GraphValue not found in environment")
}

#' @title Lower a function to StableHLO
#' @description
#' Immediately lower a flattened function to a StableHLO Func object.
#' @param f (`function` | `Gr`)\cr
#'   Flattened function to lower.
#' @return (`list`) with elements:
#'   - `func`: The StableHLO `Func` object
#'   - `out_tree`: The output tree structure
#' @export
stablehlo <- function(graph, static = character(), constants_as_inputs = TRUE, env = NULL) {
  # Node -> FuncVariable
  env <- HloEnv(parent = env)
  func <- stablehlo::local_func(id = "main")
  inps <- if (constants_as_inputs) c(graph@constants, graph@inputs) else graph@inputs

  get2 <- function(gval) {
    fvar <- env_get(env, gval)
    if (!identical(fvar@func, func)) {
      FuncVariable(fvar@value_id, fvar@value_type, func)
    } else {
      fvar
    }
  }

  for (node in inps) {
    vt <- st2vt(node@aval)
    id <- stablehlo::ValueId()
    fi <- stablehlo::FuncInput(id, vt)
    func@inputs@items <- c(func@inputs@items, fi)
    fvar <- stablehlo::FuncVariable(id, vt, func)
    env_add(env, node, fvar)
  }

  do_call <- function(call) {
    prim <- call@primitive
    params <- call@params
    inputs <- lapply(call@inputs, \(x) get2(x))
    if (is_higher_order_primitive(prim)) {
      params <- c(params, list(.env = env))
    }
    fvars_out <- rlang::exec(prim[["stablehlo"]], !!!c(inputs, params))
    if (length(call@outputs) != length(fvars_out)) {
      cli_abort("Expected {length(call@outputs)} outputs, but got {length(fvars_out)}")
    }
    for (i in seq_along(fvars_out)) {
      env_add(env, call@outputs[[i]], fvars_out[[i]])
    }
  }

  for (call in graph@calls) {
    do_call(call)
  }

  outputs <- lapply(graph@outputs, \(x) get2(x))
  func <- do.call(stablehlo::hlo_return, outputs)

  constants <- graph@constants

  list(func, constants)
}
