#' @title Builder for IR
#' @description
#' This class is used to build IR expressions.
#'
#' There is a correspondence between boxes and variables, i.e. each box
#' is associated with exactly one variable.
#'
#' @noRd
Builder <- R6::R6Class(
  "Builder",
  public = list(
    #' @field (`IREquations`)
    equations = NULL,
    #' @field (`hashtab`)\cr
    #' Mapping: `string` -> `Variable`
    boxes_to_variables = NULL,
    #' @field (`hashtab`)\cr
    #' Mapping: `string` -> `Variable`
    const_boxes = NULL,
    #' @field (`hashtab`)\cr
    #' Mapping: `Variable` -> `class_any`
    #' I.e., stores the values of the constant variables.
    const_values = NULL,
    #' @field (`list`)
    boxes = NULL,
    #' @description
    #' Initialize the builder.
    initialize = function() {
      self$equations <- list()
      self$boxes_to_variables <- hashtab()
      self$const_boxes <- hashtab()
      self$const_values <- hashtab()
      self$boxes <- list()
    },
    #' @description
    #' Create a new box.
    #' Boxes are registered so
    #' @param interpreter (`IRInterpreter`)
    #' @param `aval` (`ShapedTensor`)
    #' @return `IRBox`
    new_box = function(interpreter, aval) {
      if (!inherits(aval, ShapedTensor)) {
        stop("aval must be a ShapedTensor")
      }
      box <- IRBox(
        interpreter = interpreter,
        aval = aval
      )
      # TODO: Not sure why we are storing them.
      self$boxes <- c(self$boxes, box)
      return(box)
    },
    add_equation = function(equation) {
      self$equations <- c(self$equations, equation)
      invisible(self)
    },
    add_variable = function(box) {
      if (ir_id(box) %in% hashkeys(self$boxes_to_variables)) {
        stop("box already added")
      }
      var <- IRVariable(box@aval)
      self$boxes_to_variables[[ir_id(box)]] <- var
      return(var)
    },
    get_variable = function(box) {
      var <- self$boxes_to_variables[[ir_id(box)]]
      if (is.null(var)) {
        stop("box not found")
      }
      var
    },
    add_constant = function(box, value) {
      var <- self$add_variable(box)
      self$const_boxes[[ir_id(value)]] <- box
      self$const_values[[var]] <- value
      return(var)
    },
    build = function(in_boxes, out_boxes) {
      const_variables <- hashkeys(self$const_values)
      const_values <- hashvalues(self$const_values)

      t2v <- function(x) {
        self$boxes_to_variables[[ir_id(x)]]
      }
      in_binders <- c(const_variables, lapply(in_boxes, t2v))
      out_vars <- lapply(out_boxes, t2v)
      ir <- IRExpr(
        IRVariables(in_binders),
        IREquations(self$equations),
        IRVariables(out_vars)
      )
      #typecheck_ir(ir)
      #res <- inline_literals(ir, const_values)
      list(ir, const_values)
    }
  )
)

# inline all Variables that are constant scalars
#inline_literals <- function(ir_expr, const_values) {
#  # Split in_binders into const_binders and other_binders
#  n_consts <- length(const_values)
#  const_binders <- ir_expr@in_binders[seq_len(n_consts)]
#  if (n_consts < length(ir_expr@in_binders)) {
#    other_binders <- ir_expr@in_binders[(n_consts + 1):length(ir_expr@in_binders)]
#  } else {
#    other_binders <- list()
#  }
#
#  # Determine which constants are scalars (no shape)
#  scalars <- sapply(seq_along(const_values), function(i) {
#    x <- const_values[[i]]
#    is_nv_type(x) && length(shape(x)) == 0
#  })
#
#  # Partition const_binders and const_values based on scalars
#  new_const_binders <- const_binders[!scalars]
#  lit_binders <- const_binders[scalars]
#  new_consts <- const_values[!scalars]
#  lit_vals <- const_values[scalars]
#
#  # Create literals dictionary
#  literals <- list()
#  for (i in seq_along(lit_binders)) {
#    literals[[ir_id(lit_binders[[i]])]] <- IRLiteral(lit_vals[[i]])
#  }
#
#  # Replace literals in equations
#  new_eqns <- lapply(seq_along(ir_expr@equations), function(i) {
#    eqn <- ir_expr@equations[[i]]
#    new_inputs <- lapply(seq_along(eqn@inputs), function(j) {
#      x <- eqn@inputs[[j]]
#      lit <- literals[[ir_id(x)]]
#      if (!is.null(lit)) lit else x
#    })
#    IREquation(eqn@primitive, new_inputs, eqn@params, eqn@out_binders)
#  })
#
#  # Replace literals in outputs
#  new_outs <- lapply(seq_along(ir_expr@outs), function(i) {
#    x <- ir_expr@outs[[i]]
#    lit <- literals[[ir_id(x)]]
#    if (!is.null(lit)) lit else x
#  })
#
#  # Create new IR expression
#  new_ir_expr <- IRExpr(c(new_const_binders, other_binders), new_eqns, new_outs)
#
#  return(list(ir_expr = new_ir_expr, const_values = new_consts))
#}
#
