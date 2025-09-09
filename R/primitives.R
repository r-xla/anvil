# Core Primitives

nvl_op <- S7::new_S3_class("nvl_op")

anvil_op <- function(name, dispatch_args, fun) {
  gen <- S7::new_generic(name, dispatch_args, fun)
  class(gen) <- c("anvil_op", class(gen))
  gen
}

AnvilOp <- S7::new_S3_class("anvil_op")

prim_add <- function(lhs, rhs) {
  interprete(
    op_add,
    list(lhs, rhs)
  )
}

op_add <- anvil_op("AnvilAdd", c("lhs", "rhs"), function(lhs, rhs) {
  S7::S7_dispatch()
})
