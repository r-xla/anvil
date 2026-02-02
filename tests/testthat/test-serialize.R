# the main tests are in {pjrt}, hence we keep them minimal here

test_that("nv_serialize and nv_unserialize work for single tensor", {
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  lst <- list(x = x)
  raw_data <- nv_serialize(lst)
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data)
  expect_equal(lst, reloaded)
})

test_that("nv_write and nv_read works for a single tensor", {
  x <- nv_tensor(array(rnorm(12), dim = c(3, 4)))
  lst <- list(x = x)
  tmp <- tempfile(fileext = ".safetensors")
  nv_write(lst, tmp)
  reloaded <- nv_read(tmp)
  expect_equal(lst, reloaded)
})

test_that("serialization preserves ambiguity", {
  # Create tensors with different ambiguity
  ambiguous_tensor <- nv_scalar(1.0, ambiguous = TRUE) # ambiguous
  non_ambiguous_tensor <- nv_tensor(1.0, dtype = "f32") # non-ambiguous

  lst <- list(
    ambiguous = ambiguous_tensor,
    non_ambiguous = non_ambiguous_tensor
  )

  # Test with nv_serialize/nv_unserialize
  raw_data <- nv_serialize(lst)
  reloaded <- nv_unserialize(raw_data)

  expect_true(ambiguous(reloaded$ambiguous))
  expect_false(ambiguous(reloaded$non_ambiguous))
  expect_equal(lst$ambiguous, reloaded$ambiguous)
  expect_equal(lst$non_ambiguous, reloaded$non_ambiguous)

  # Test with nv_write/nv_read
  tmp <- tempfile(fileext = ".safetensors")
  nv_write(lst, tmp)
  reloaded2 <- nv_read(tmp)

  expect_true(ambiguous(reloaded2$ambiguous))
  expect_false(ambiguous(reloaded2$non_ambiguous))
  expect_equal(lst$ambiguous, reloaded2$ambiguous)
  expect_equal(lst$non_ambiguous, reloaded2$non_ambiguous)
})
