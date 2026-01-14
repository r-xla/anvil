#' @title Make initial state
#' @name nv_seed2state
#' @description
#' converts a random seed into a initial state tensor
#' @param dtype output dtype either "ui32" or "ui64"
#' @param shape output shape
#' @param random_seed explicitly provide the random seed of a R session. auto-detects if not provided.
#' @param hash_algo hash algorithm to hash the random state with. Default is 'sha512'.
#' @export
nv_seed2state <- function(
  dtype = "ui64",
  shape = 2,
  random_seed = NULL,
  hash_algo = "sha512"
) {
  checkmate::assertChoice(dtype, c("ui32", "ui64"))
  n_states <- prod(shape)

  # auto detect random_seed
  if (is.null(random_seed)) {
    random_seed <- .Random.seed # nolint
  }

  # hash the seed
  hash_hex <- digest::digest(
    random_seed,
    algo = hash_algo,
    serialize = TRUE
  )

  # convert hex string to bytes
  hex_pairs <- substring(
    hash_hex,
    seq(1, nchar(hash_hex), 2),
    seq(2, nchar(hash_hex), 2)
  )
  hash_bytes <- strtoi(hex_pairs, base = 16)

  # calculate bytes needed/available
  bytes_per_value <- if (dtype == "ui32") 4 else 8
  bytes_needed <- n_states * bytes_per_value
  total_bytes_available <- length(hash_bytes)

  # throw error if not enough bytes available
  if (bytes_needed > total_bytes_available) {
    cli_abort(
      "Requested {n_states} {dtype} values (total {bytes_needed} bytes) but hash
      algorithm '{hash_algo}' only provides {total_bytes_available} bytes. "
    )
  }

  # tensor of type i8 of shape (shape, 4/8)
  raw8 <- nv_tensor(
    as.integer(hash_bytes[seq_len(bytes_needed)]),
    shape = c(shape, bytes_per_value),
    dtype = "i8"
  )

  # upcast raw8 to requested dtype, last dimension vanished
  nv_bitcast_convert(raw8, dtype = dtype) # nolint
}


test_that("seed2state", {
  # auto-detect state
  set.seed(42)
  f <- function() {
    nv_seed2state(shape = c(3, 2))
  }
  g <- jit(f)
  out1 <- g()

  # explicitly provide state
  set.seed(42)
  f <- function() {
    nv_seed2state(shape = c(3, 2), random_seed = .Random.seed)
  }
  g <- jit(f)
  out2 <- g()

  expect_true(identical(as_array(out1), as_array(out2)))
  expect_equal(shape(out2), c(3, 2))
  expect_true(inherits(dtype.AnvilTensor(out1), UnsignedType))
  expect_equal(dtype.AnvilTensor(out1)@value, 64L)

  # test ui32
  set.seed(1)
  f <- function() {
    nv_seed2state(shape = 2, dtype = "ui32")
  }
  g <- jit(f)
  out3 <- g()
  expect_equal(shape(out3), 2)
  expect_true(inherits(dtype.AnvilTensor(out3), UnsignedType))
  expect_equal(dtype.AnvilTensor(out3)@value, 32L)
})
