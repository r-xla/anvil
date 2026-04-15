old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  anvil.default_backend = "xla"
)
# https://github.com/HenrikBengtsson/Wishlist-for-R/issues/88
old_opts <- lapply(old_opts, function(x) if (is.null(x)) FALSE else x)

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
