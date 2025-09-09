# Test the IR implementation
source("R/repr.R")
source("R/ir.R")

# Create a simple multiplication operation
mul_op <- op("mul")

# Create an equation: z = mul(x, y)
eqn <- equation("z", mul_op, list("x", "y"))

# Create an Expr function: f(x, y) = z where z = mul(x, y)
j <- expr(
  parameters = list("x", "y"),
  equations = list(eqn),
  return_val = "z"
)

# Print the Expr
cat("Expr representation:\n")
cat(repr(j), "\n")

# Test with numeric literals
add_op <- op("add")
eqn2 <- equation("result", add_op, list("x", 5.0))

j2 <- expr(
  parameters = list("x"),
  equations = list(eqn2),
  return_val = "result"
)

cat("\nExpr with numeric literal:\n")
cat(repr(j2), "\n")
