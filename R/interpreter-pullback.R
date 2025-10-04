#' @include interpreter.R

PullbackNode <- S7::new_class(
  "PullbackNode",
  properties = list(
    pullback = class_function | NULL,
    # List of PullbackNode, but see: https://github.com/RConsortium/S7/issues/565
    parents = list_of(new_class("PullbackNode")),
    id = class_environment,
    required = class_logical
  ),
  constructor = function(pullback, parents, id, required = FALSE) {
    S7::new_object(
      S7::S7_object(),
      pullback = pullback,
      parents = parents,
      id = new.env(size = 0L),
      required = required
    )
  }
)

node_id <- function(node) {
  rlang::obj_address(node)
}

#' @title Gradient of a function
#' @description
#' Compute the gradient of a function using reverse mode automatic differentiation.
#' Output must be a 0-dimensional tensor.
#' @param f (`function`)
#' @param wrt (`character()`)
#'   Names of arguments to differentiate with respect to. If `NULL`,
#'   compute gradients w.r.t. all differentiable arguments.
#' @return (`function`)
#' @export
gradient <- function(f, wrt = NULL) {
  #    def gradfun(x, *xs):
  #    y, f_vjp = vjp(f, x, *xs)
  #    if np.shape(y) != (): raise TypeError
  #    x_bar, *_ = f_vjp(np.ones(np.shape(y), np.result_type(y)))
  #    return x_bar
  #  return gradfun

  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    if (!is.null(wrt)) {
      assert_subset(wrt, formalArgs2(f))
    }
    out <- rlang::exec(pullback2, f, !!!args, wrt = wrt)
    g <- out[[2L]]
    y <- out[[1L]]

    if (!identical(shape(y), integer())) {
      stop("Function must return a scalar")
    }
    # TODO: Assert float

    # TODO: check shape

    one <- nv_scalar(1.0, dtype = repr(dtype(y)))
    grad <- g(one)

    #if (length(grad) == 1L) {
    #  # single input
    #  grad <- grad[[1L]]
    #}

    return(grad)
  }
  # args() is needed for primitives, as they don't have formals
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}

pullback2 <- function(f, ..., wrt = NULL) {
  main <- local_main(PullbackInterpreter)
  interpreter <- PullbackInterpreter(main)

  if (!is.null(wrt)) {
    assert_subset(wrt, formalArgs2(f))
  } else {
    wrt <- formalArgs2(f)
  }

  args <- list(...)

  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  in_node <- build_tree(mark_some(args, wrt))
  is_required_flat <- if (is.null(wrt)) rep(TRUE, length(args_flat)) else in_node$marked
  f_flat <- rlang::exec(flatten_fun, f, in_node = in_node)

  # The inputs are root nodes, because they have no parents
  # TODO: When we allow for partial gradients, we need to set required = FALSE
  # for those where we don't want gradients.
  boxes_in <- Map(
    function(arg, req) {
      PullbackBox(interpreter, arg, PullbackNode(NULL, list(), required = req))
    },
    args_flat,
    is_required_flat
  )

  out_flat <- rlang::exec(f_flat, !!!boxes_in)

  box_out <- rlang::exec(unflatten, !!!out_flat)

  #if (length(boxes_out) != 1L) {
  #  stop("Function must return a single value")
  #}

  in_nodes <- lapply(boxes_in, \(box) box@node)
  out_node <- box_out@node

  # TODO: Crate
  f <- function(grad) {
    outs_flat <- backward_pass(in_nodes, out_node, grad)
    out <- rlang::exec(unflatten, in_tree, outs_flat)
    out <- out[names(args) %in% wrt]
    out
  }

  list(
    box_out@primal,
    f
  )
}

#' @title Pullback of a function
#' @description
#' Compute the pullback (transposed derivative) of a function.
#' @param f (`function`)\cr
#'   Function to compute the pullback of.
#' @param ... (`any`)\cr
#'   Example arguments to pass to the function.
#' @param wrt (`character()`)
#'   Names of arguments to differentiate with respect to. If `NULL`,
#'   compute gradients w.r.t. all differentiable arguments.
#' @return (`function`)
#' @export
pullback <- function(f, ..., wrt = NULL) {
  assert_subset(wrt, formalArgs2(f))
  pullback2(f, ..., wrt = wrt)[[2L]]
}

# Backward is essentially implemented like in pytorch or microjax

backward_pass <- function(in_nodes, out_node, gradient) {
  node_map <- hashtab()
  node_map[[node_id(out_node)]] <- gradient

  # 1. Reverse the topological sorting
  # 2. Prune it to compute only the backward ops that
  #    are needed for the required roots.
  #    This is important when we only want gradients of a subset of the inputs.

  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    nvl_add(grad1, grad2)
  }

  # Use full reverse toposort as requested
  topo_sorted <- reverse_toposort(out_node)
  # walk from leaf to root
  for (node in topo_sorted) {
    node_grad <- node_map[[node_id(node)]]
    closures <- node@pullback(node_grad)
    if (length(closures)) {
      for (i in seq_along(closures)) {
        cl <- closures[[i]]
        if (is.null(cl)) {
          next
        }
        parent_id <- node_id(node@parents[[i]])
        contrib <- cl()
        node_map[[parent_id]] <- add_or_init(
          node_map[[parent_id]],
          contrib
        )
      }
    }
  }
  outs <- lapply(seq_along(in_nodes), function(i) {
    node <- in_nodes[[i]]
    if (node@required) node_map[[node_id(node)]] else NULL
  })
  outs
}

reverse_toposort <- function(end_node) {
  .toposort <- function(seen, node) {
    result <- list()
    if (!(set_has(seen, node_id(node)))) {
      set_add(seen, node_id(node))
      for (parent in node@parents) {
        result <- c(result, .toposort(seen, parent))
      }
      result <- c(result, list(node))
    }
    return(result)
  }
  x <- .toposort(set(), end_node)
  # Remove root nodes, because they don't have a grad
  # function. There we want to collect the gradients
  rev(x[sapply(x, \(node) length(node@parents) > 0L)])
}

pruned_reverse_toposort <- function(end_node) {
  ordered_nodes <- reverse_toposort(end_node)
  memo <- hashtab() # nolint
  contributes <- function(node) {
    id <- node_id(node)
    if (!is.null(memo[[id]])) {
      return(memo[[id]])
    }
    v <- if (!length(node@parents)) {
      node@required
    } else {
      any(vapply(node@parents, contributes, logical(1)))
    }

    memo[[id]] <- v
    v
  }

  ordered_nodes[vapply(ordered_nodes, contributes, logical(1))]
}

PullbackInterpreter <- new_class(
  "PullbackInterpreter",
  parent = Interpreter
)

method(process_primitive, PullbackInterpreter) <- function(
  interpreter,
  prim,
  boxes,
  params
) {
  primals_in <- lapply(boxes, \(box) box@primal)
  nodes_in <- lapply(boxes, \(box) box@node)
  req_in <- vapply(nodes_in, function(n) n@required, logical(1))
  out <- rlang::exec(prim@pullback_rule, primals = primals_in, .required = req_in, !!!params)
  primals_out <- out[[1L]]
  pullback_out <- out[[2L]]
  node_out <- PullbackNode(pullback_out, nodes_in, required = any(req_in))
  # they all have the same node, because they are computed by the same op
  lapply(primals_out, \(primal) PullbackBox(interpreter, primal, node_out))
}

PullbackBox <- new_class(
  "PullbackBox",
  parent = Box,
  properties = list(
    # TODO: What type is this?
    primal = class_any,
    node = PullbackNode
  )
)

method(box, list(PullbackInterpreter, class_any)) <- function(interpreter, x) {
  PullbackBox(
    interpreter = interpreter,
    primal = x,
    node = PullbackNode(NULL, list(), required = FALSE)
  )
}

method(aval, PullbackBox) <- function(x) {
  x@primal
}
