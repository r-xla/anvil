hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})

method(hash, class_environment) <- function(x) {
  # TODO: Make this nicer
  format(x)
}

# set utils
set <- function() {
  hashtab()
}

set_has <- function(set, key) {
  !identical(gethash(set, key, NA), NA)
}

set_add <- function(set, key) {
  set[[key]] <- NULL
}

dtype_from_buffer <- function(x) {
  d <- as.character(dtype(x))
  as_dtype(d)
}

raise_to_shaped <- function(aval) {
  ShapedTensor(aval@dtype, aval@shape)
}


id <- S7::new_generic("id", "x", function(x) {
  S7::S7_dispatch()
})

method(id, class_environment) <- function(x) {
  rlang::addr_address(x)
}

hashkeys <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
    val[[idx]] <<- k
  })
  val
}

hashvalues <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
    val[[idx]] <<- v
  })
  val
}

is_nv_type <- function(x) {
  any(sapply(globals$nv_types, function(type, x) inherits(x, type), x))
}


transpose_list <- function(.l) {
  if (length(.l) == 0L) {
    return(list())
  }
  res <- .mapply(list, .l, list())
  if (length(res) == length(.l[[1L]])) {
    names(res) <- names(.l[[1L]])
  }
  res
}

seq_len0 <- function(n) {
  x <- seq_len(n)
  if (length(x)) x - 1L else integer()
}

seq_along0 <- function(x) {
  x <- seq_along(x)
  if (length(x)) x - 1L else integer()
}
