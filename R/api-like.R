# `nv_*_like()` variants of the array constructors. Each takes an existing
# arrayish `like` and fills in any `NULL` attribute (dtype, shape, device,
# ambiguous, backend) from it, before delegating to the underlying
# constructor.
#
# `like_defaults()` is the shared helper that does the filling.

# Resolve defaults for an `nv_*_like()` helper. Each named argument is returned
# unchanged if non-NULL, or filled in from the corresponding attribute of
# `like`. Only fields the caller asked about are returned, so the result can
# be spliced into a `do.call()` of the underlying constructor.
#
# `device` is only read from `like` when `like` is a concrete AnvilArray; for
# a GraphBox (during tracing) we leave `device` as NULL so downstream
# constructors pick the device up from the tracing context.
like_defaults <- function(like, ...) {
  args <- list(...)
  getters <- list(
    dtype = dtype,
    shape = shape,
    device = function(x) if (is_anvil_array(x)) device(x),
    ambiguous = ambiguous,
    backend = backend
  )
  for (name in names(args)) {
    if (is.null(args[[name]])) {
      args[[name]] <- getters[[name]](like)
    }
  }
  args
}

#' @rdname AnvilArray
#' @param like ([`AnvilArray`])\cr
#'   An existing array. Any of `dtype`, `device`, `shape`, `ambiguous`, and
#'   `backend` that are `NULL` (the default) are taken from `like`.
#' @export
nv_array_like <- function(like, data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL, backend = NULL) {
  do.call(
    nv_array,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device, shape = shape, ambiguous = ambiguous, backend = backend)
    )
  )
}

#' @rdname AnvilArray
#' @export
nv_scalar_like <- function(like, data, dtype = NULL, device = NULL, ambiguous = NULL, backend = NULL) {
  do.call(
    nv_scalar,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device, ambiguous = ambiguous, backend = backend)
    )
  )
}

#' @rdname AnvilArray
#' @export
nv_empty_like <- function(like, dtype = NULL, shape = NULL, device = NULL, ambiguous = NULL) {
  do.call(nv_empty, like_defaults(like, dtype = dtype, shape = shape, device = device, ambiguous = ambiguous))
}

#' @rdname nv_fill
#' @export
nv_fill_like <- function(like, value, shape = NULL, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_fill,
    c(
      list(value = value),
      like_defaults(like, shape = shape, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_iota
#' @export
nv_iota_like <- function(like, dim, shape = NULL, start = 1L, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_iota,
    c(
      list(dim = dim, start = start),
      like_defaults(like, shape = shape, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_seq
#' @export
nv_seq_like <- function(like, start, end, steps = NULL, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_seq,
    c(
      list(start = start, end = end, steps = steps),
      like_defaults(like, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_eye
#' @export
nv_eye_like <- function(like, n, dtype = NULL, device = NULL) {
  do.call(nv_eye, c(list(n = n), like_defaults(like, dtype = dtype, device = device)))
}
