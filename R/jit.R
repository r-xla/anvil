# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

#' @title Lower a function to StableHLO
#' @description
#' Immediately lower a flattened function to a StableHLO Func object.
#' @param f (`function` | `Gr`)\cr
#'   Flattened function to lower.
#' @param .avals (`list()`)\cr
#'   List of flattened abstract values (avals) for the function arguments.
#'   ShapedTensors will be traced (wrapped in HloBox), while other values
#'   will be passed through as-is (static).
#' @return (`list`) with elements:
#'   - `func`: The StableHLO `Func` object
#'   - `out_tree`: The output tree structure
#' @export
stablehlo <- function(f, static = character()) {
  StableHloTransformation(f, static)
}

LoweringTransformation <- new_class(
  parent = Transformation,
  "LoweringTransformation"
)

#' @include graph.R
StableHloTransformation <- new_class(
  "StableHloTransformation",
  parent = LoweringTransformation,
  properties = list(
    input = class_function | Graph | GraphTransformation,
    static = class_character
  )
)

method(apply_transform, StableHloTransformation) <- function(gt, args) {
  func <- stablehlo::local_func(id = "main")

  fvars_in <- lapply(args, \(x) st2fv(x, func))
  # TODO: StableHloBox

  reducer <- function(prim, inputs, params) {
    rlang::exec(prim[["stablehlo"]], !!!c(inputs, params))
  }

  fvars_out <- graph_reduce(gt@input, reducer, fvars_in)
  func <- do.call(stablehlo::hlo_return, fvars_out)
  list(func, gt@input@out_tree)
}

#' @title JIT compile a function
#' @description
#' Convert a function to a JIT compiled function.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param static (`character()`)\cr
#'   Which parameters of `f` are static.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device to use for the compiled function.
#'   The default (`NULL`) uses the `PJRT_PLATFORM` environment variable or defaults to "cpu".
#' @param cache_size (`integer(1)`)\cr
#'   The size of the cache for the jit-compiled functions.
#' @return (`function`)
#' @export
jit <- function(f, static = character(), device = NULL, cache_size = 100L, backend) {
  device <- device %??% Sys.getenv("PJRT_PLATFORM", "cpu")
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  device_str <- as.character(device)

  tf <- stablehlo(f, static)

  f_jit <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    in_node <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_node$marked

    call_xla <- function(exec, out_node, args_flat, is_static_flat) {
      args_nonstatic <- args_flat[!is_static_flat]
      out_vals <- rlang::exec(
        pjrt::pjrt_execute,
        exec,
        !!!args_nonstatic,
        simplify = FALSE
      )
      out_vals <- lapply(out_vals, nv_tensor)
      unflatten(out_node, out_vals)
    }

    avals_in <- Map(function(x, is_static) {
      if (is_static) {
        x
      } else {
        if (platform(x) != device_str) {
          cli_abort("Expected device {device_str}, but buffer has device {platform(x)}")
        }
        raise_to_shaped(aval(x))
      }
    }, args_flat, is_static_flat)

    cache_hit <- cache$get(avals_in)
    if (!is.null(cache_hit)) {
      call_xla(cache_hit[[1]], cache_hit[[2]], args_flat, is_static_flat)
    }
    # TODO: Can we also pass the avals here already? I think it should be possible
    out <- apply_transform(tf, args_flat)
    src <- stablehlo::repr(out[[1L]])
    program <- pjrt_program(src = src, format = "mlir")
    exec <- pjrt_compile(program)
    cache$set(avals_in, list(exec, out[[2L]]))
    call_xla(exec, out[[2L]], args_flat, is_static_flat)
  }
  formals(f_jit) <- formals2(f)
  f_jit
}


jit2 <- function(f, static = character(), device = NULL, cache_size = 100L) {
  device <- device %??% Sys.getenv("PJRT_PLATFORM", "cpu")
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_subset(static, formalArgs2(f))
  device_str <- as.character(device)
  f_jit <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    in_node <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_node$marked
    avals_in <- Map(
      function(a, static) {
        if (static) {
          a
        } else {
          if (platform(a) != device_str) {
            cli_abort("Expected device {device_str}, but buffer has device {platform(a)}")
          }
        }
        if (static) a else raise_to_shaped(aval(a))
      },
      args_flat,
      is_static_flat
    )
    cache_hit <- cache$get(avals_in)
    # TODO: Factor this out into a function and call it at the end
    # instead of a recall, which does some work twice
    if (!is.null(cache_hit)) {
      # Only pass non-static arguments to pjrt_execute
      args_nonstatic <- args_flat[!is_static_flat]
      res <- rlang::exec(
        pjrt::pjrt_execute,
        cache_hit[[1L]],
        !!!args_nonstatic,
        simplify = FALSE
      )
      res <- lapply(res, nv_tensor)
      return(unflatten(cache_hit[[2]], res))
    }

    # Prepare flattened function and avals for lowering
    f_flat <- rlang::exec(flatten_fun, f, in_node = in_node)

    lowered <- stablehlo(f_flat, avals_in)
    program <- pjrt_program(src = repr(lowered$func), format = "mlir")
    exec <- pjrt_compile(program)
    cache$set(
      avals_in,
      list(
        exec,
        lowered$out_tree
      )
    )
    Recall()
  }
  formals(f_jit) <- formals2(f)
  return(f_jit)
}

jit2 <- function(.f, ...) {
  if (is_graph_graph)
  materialize(f, ...)
}

HloBox <- S7::new_class(
  "HloBox",
  parent = Box,
  properties = list(
    func_var = stablehlo::FuncVariable
  )
)

method(format, HloBox) <- function(x, ...) {
  sprintf("HloBox(%s)", format(aval(x)))
}

HloInterpreter <- S7::new_class(
  "HloInterpreter",
  parent = Interpreter,
  properties = list(
    func = stablehlo::Func
  )
  # TODO: Should also get a builder, once stablehlo has one
  # It definitely needs to keep track of the constants
)

method(process_primitive, HloInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  avals_in <- lapply(boxes, \(box) box@func_var)
  avals_out <- rlang::exec(prim[["jit"]], !!!c(avals_in, params))
  lapply(avals_out, \(aval) HloBox(func_var = aval, interpreter = interpreter))
}

#method(box, list(HloInterpreter, class_any)) <- function(interpreter, x) {
#  box(interpreter, aval(x))
#}

method(box, list(HloInterpreter, AnvilTensor)) <- function(interpreter, x) {
  box(interpreter, aval(x))
}

method(box, list(HloInterpreter, ShapedTensor)) <- function(interpreter, x) {
  HloBox(
    func_var = FuncVariable(
      value_id = stablehlo::ValueId(),
      value_type = st2vt(x),
      func = interpreter@main@global_data
    ),
    interpreter = interpreter
  )
}

method(box, list(HloInterpreter, ConcreteTensor)) <- function(interpreter, x) {
  func_var <- stablehlo::hlo_tensor(x@data, func = interpreter@main@global_data)
  return(HloBox(
    func_var = func_var,
    interpreter = interpreter
  ))
}

method(box, list(HloInterpreter, class_any)) <- function(interpreter, x) {
  box(interpreter, aval(x))
}


# Preserve value identity when something is already boxed
method(box, list(HloInterpreter, HloBox)) <- function(interpreter, x) {
  HloBox(
    func_var = FuncVariable(
      value_id = x@func_var@value_id,
      value_type = x@func_var@value_type,
      func = interpreter@main@global_data
    ),
    interpreter = interpreter
  )
}

method(aval, HloBox) <- function(x) {
  vt2st(x@func_var@value_type)
}
