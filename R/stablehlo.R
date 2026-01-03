# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

HloEnv <- S7::new_class(
  "HloEnv",
  properties = list(
    parent = NULL | new_S3_class("anvil::HloEnv"),
    gval_to_fval = new_property(class_hashtab, default = quote(hashtab()))
  )
)

env_add <- function(env, gval, fval) {
  env@gval_to_fval[[gval]] <- fval
  invisible(env)
}

env_get <- function(env, gval) {
  fval <- env@gval_to_fval[[gval]]
  if (!is.null(fval)) {
    return(fval)
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
#' @param graph (`Graph`)\cr
#'   The graph to lower.
#' @param constants_as_inputs (`logical(1)`)\cr
#'   Whether to add constants as inputs.
#' @param env (`HloEnv` | `NULL`)\cr
#'   The environment for storing graph value to func variable mappings.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type.
#' @return (`list`) with elements:
#'   - `func`: The StableHLO `Func` object
#'   - `constants`: The constants of the graph
#' @export
stablehlo <- function(graph, constants_as_inputs = TRUE, env = NULL, donate = character()) {
  # Node -> FuncValue
  env <- HloEnv(parent = env)
  func <- stablehlo::local_func(id = "main")
  inps <- if (constants_as_inputs) c(graph@constants, graph@inputs) else graph@inputs

  gnode_to_fval <- function(gnode) {
    fval <- env_get(env, gnode)
    if (!identical(fval@func, func)) {
      FuncValue(fval@value_id, fval@value_type, func)
    } else {
      fval
    }
  }

  # Compute which inputs are donated (only graph@inputs, not constants)
  donate_flat <- if (length(donate) > 0L && !is.null(graph@in_tree)) {
    # Constants are never donated, inputs may be
    c(
      rep(FALSE, length(graph@constants)),
      flat_mask_from_names(graph@in_tree, donate)
    )
  } else {
    rep(FALSE, length(inps))
  }

  # Get output types for aliasing
  out_types <- lapply(graph@outputs, function(out) {
    st2va(out@aval)
  })

  # Track which outputs have been aliased (0-based indices)
  aliased_outputs <- integer()

  for (i in seq_along(inps)) {
    node <- inps[[i]]
    vt <- st2va(node@aval)
    id <- stablehlo::ValueId()

    # Check if this input is donated and find a matching output
    alias <- NULL
    if (donate_flat[[i]]) {
      # Find an output with matching type that hasn't been aliased yet
      for (j in seq_along(out_types)) {
        if ((j - 1L) %in% aliased_outputs) {
          next
        }
        out_vt <- out_types[[j]]
        if (vt == out_vt) {
          alias <- j - 1L # 0-based index for stablehlo
          aliased_outputs <- c(aliased_outputs, alias)
          break
        }
      }
    }

    fi <- stablehlo::FuncInput(id, vt, alias = alias)
    func@inputs@items <- c(func@inputs@items, fi)
    fval <- stablehlo::FuncValue(id, vt, func)
    env_add(env, node, fval)
  }

  if (!constants_as_inputs) {
    for (const in graph@constants) {
      if (is.null(env_get(env, const))) {
        cli_abort("Internal error: constant not found in environment")
      }
    }
  }

  do_call <- function(call) {
    prim <- call@primitive
    params <- call@params
    inputs <- lapply(call@inputs, \(x) {
      if (is_graph_literal(x)) {
        # need to add a literal to the program
        fval <- hlo_tensor(value = x@aval@data, dtype = x@aval@dtype, shape = x@aval@shape@dims, func = func)
        env_add(env, x, fval)
        fval
      } else {
        gnode_to_fval(x)
      }
    })
    if (is_higher_order_primitive(prim)) {
      params <- c(params, list(.env = env))
    }
    fvals_out <- rlang::exec(prim[["stablehlo"]], !!!c(inputs, params))
    if (length(call@outputs) != length(fvals_out)) {
      cli_abort("Expected {length(call@outputs)} outputs, but got {length(fvals_out)}")
    }
    for (i in seq_along(fvals_out)) {
      env_add(env, call@outputs[[i]], fvals_out[[i]])
    }
  }

  for (call in graph@calls) {
    do_call(call)
  }

  outputs <- lapply(graph@outputs, \(x) {
    if (is_graph_literal(x)) {
      # this only happens when a literal is directly returned
      hlo_tensor(value = x@aval@data, dtype = x@aval@dtype, shape = x@aval@shape@dims, func = func)
    } else {
      gnode_to_fval(x)
    }
  })
  func <- do.call(stablehlo::hlo_return, outputs)

  constants <- graph@constants

  list(func, constants)
}
