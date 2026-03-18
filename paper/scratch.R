# nimble
library(nimble)


# nimble supports autodiff:

calc_grad <- nimbleFunction(
  setup = function() {},
  methods = list(
    f = function(x = double(1)) {
      returnType(double(0))
      return(sum(x^2))
    },
    grad = function(x = double(1)) {
      returnType(ADNimbleList())
      out <- nimDerivs(f(x), order = c(0, 1, 2), wrt = 1:length(x))
      return(out)
    }
  ),
  buildDerivs = list(f = list())
)

calc_grad_inst <- calc_grad()
c_calc_grad <- compileNimble(calc_grad_inst)

x <- c(1.0, 2.0, 3.0)
result <- c_calc_grad$grad(x)

result$value     # f(x) = 14
result$jacobian  # [2, 4, 6]
result$hessian   # diag(2, 2, 2)

# jags (via rjags)
library(rjags)

model_string <- "model {
  for (i in 1:N) {
    y[i] ~ dnorm(mu, tau)
  }
  mu ~ dnorm(0, 0.001)
  tau ~ dgamma(0.001, 0.001)
  sigma <- 1 / sqrt(tau)
}"

data <- list(y = c(1.2, 1.8, 2.1, 1.5, 1.9), N = 5)

model <- jags.model(textConnection(model_string), data = data, n.chains = 2, quiet = TRUE)
update(model, 1000)
samples <- coda.samples(model, c("mu", "sigma"), n.iter = 5000)
summary(samples)

# rstanarm
library(rstanarm)

fit <- stan_glm(Sepal.Length ~ Petal.Length, data = iris,
  family = gaussian(), prior = normal(0, 3), prior_intercept = normal(0, 5),
  chains = 2, iter = 2000, refresh = 0
)
summary(fit)

# --- RTMB ---
# Reverse-mode AD via tape recording. Supports arbitrary-order derivatives
# by chaining jacfun().
library(RTMB)

# Record a tape: MakeTape(f, prototype_input)
f <- function(x) sum(x^2)
F <- MakeTape(f, numeric(3))
F  # R^3 -> R^1

# Evaluate at a new point
F(c(1, 2, 3))  # 14

# Gradient (Jacobian of scalar function = gradient as row vector)
F$jacobian(c(1, 2, 3))  # [2, 4, 6]

# Hessian via chaining jacfun (each call differentiates once more)
H <- F$jacfun()$jacfun()
H(c(1, 2, 3))  # diag(2, 2, 2)

# More interesting example
g <- function(x) exp(x[1] + 1.23 * x[2])
G <- MakeTape(g, numeric(2))
G$jacobian(c(3, 4))  # [exp(3+4.92), 1.23*exp(3+4.92)]

# Simplify the tape for performance
G$simplify("optimize")

# ---------------------------------------------------------------------------
# R packages for Automatic Differentiation
# All five below are FORWARD-MODE only (dual numbers / forward accumulation).
# None supports reverse-mode AD.
# ---------------------------------------------------------------------------

# --- nabla ---
# Forward mode via nested dual numbers. Supports arbitrary-order derivatives.
library(nabla)

f <- function(x) x^2 + sin(x)
D(f, 1.0)                # f'(1)
D(f, 1.0, order = 2)     # f''(1)

# Multivariate
g <- function(x) sum(x^2)
gradient(g, c(1, 2, 3))  # [2, 4, 6]
hessian(g, c(1, 2, 3))   # diag(2, 2, 2)

# --- salad ---
# Forward mode via S4 dual numbers.
library(salad)

f <- function(x) x^2 + sin(x)
x <- dual(1.0)
y <- f(x)
value(y)  # 1.841...
d(y)      # 2.540... = 2*1 + cos(1)

# Multivariate: create all variables in a single dual() call
x <- dual(c(1, 2, 3))
y <- sum(x^2)
d(y)  # [2, 4, 6]

# --- dual ---
# Forward mode, minimal API. Seed grad = 1 for univariate derivative.
library(dual)

x <- dual(f = 1.0, grad = 1)
y <- x^2 + sin(x)
y$f     # value
y$grad  # derivative

# Multivariate: use seed vectors for partial derivatives
x1 <- dual(f = 1.0, grad = c(1, 0))
x2 <- dual(f = 2.0, grad = c(0, 1))
y <- x1^2 + x2^2
y$grad  # [2, 4] = [df/dx1, df/dx2]

# --- madness ---
# Forward mode, focused on matrix-valued functions and the Delta method.
library(madness)

x <- madness(matrix(1.0), vtag = "y", xtag = "x")
y <- x^2 + sin(x)
val(y)    # value
dvdx(y)   # derivative

# Matrix example
X <- madness(matrix(rnorm(6), 3, 2))
Y <- X %*% t(X)
val(Y)
dvdx(Y)   # Jacobian of vec(Y) w.r.t. vec(X)

# --- ADtools ---
# Forward mode, matrix-calculus notation. Archived from CRAN (2021).
library(ADtools)

f <- function(X) X %*% t(X)
X <- matrix(rnorm(6), 3, 2)
result <- auto_diff(f, at = list(X = X), wrt = "X")
result@x   # function value
result@dx  # Jacobian
