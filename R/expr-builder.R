
#' Here we need reference semantic, because this will be a global variable.
#' The problem is that we can't have one builder always at the bottom of the stack
#' Because
#' @export
Builder <- R6::R6Class(
  "Builder",
  public = list(
    equations = NULL,
    boxes_to_variables = NULL,
    const_boxes = NULL,
    const_variables = NULL,
    boxes = NULL,
    initialize = function() {
      self$equations <- list()
      self$boxes_to_variables <- list()
      self$const_boxes <- list()
      self$const_variables <- list()
      self$boxes <- list()
    },
    add_equation = function(equation) {
      self$equations <- c(self$equations, equation)
      self
    },
    add_variable = function(box) {
      self$boxes_to_variables[[id(box)]] <- Variable(box@aval)
      self
    },
    get_variable = function(box) {
      var <- self$boxes_to_variables[[id(box)]]
      if (is.null(var)) {
        stop("box not found")
      }
      var
    },
    add_constant = function(box, value) {
      self$const_boxes[[id(box)]] <- box
      self$const_variables[[box]] <- value
      self
    },
    build = function(in_boxes, out_boxes) {
      const_variables <- hashkeys(self$const_variables)
      const_boxes <- hashkeys(self$const_boxes)
      t2v <- function(x) {
        self$boxes_to_variables[[id(x)]]
      }
      in_binders <- c(const_variables, lapply(in_boxes, t2v))
      out_vars <- lapply(out_boxes, t2v)
      expr <- Expr(in_binders, self$equations, out_vars)
      typecheck_expr(expr)
      expr
    },
    new_box = function(aval) {
      if (!inherits(aval, ShapedArray)) {
        stop("aval must be a ShapedArray")
      }
      # A ConcreteArray is also a ShapedArray, but we need to raise it to a ShapedArray
      aval <- ExprBox(raise_to_shaped(aval))
      self$boxes <- c(self$boxes, aval)
    }
  )
)
