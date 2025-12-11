hash <- S7::new_generic("hash", "x", function(x) {
  S7::S7_dispatch()
})

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

# TODO: unify this with st() etc. and all those related functions
raise_to_shaped <- function(aval) {
  ShapedTensor(aval@dtype, aval@shape)
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

# these functions also work with primitives etc.
formalArgs2 <- function(f) {
  names(formals2(f))
}

formals2 <- function(f) {
  formals(args(f))
}


# We assume little endian
minmax_raw <- function(bits, signed = TRUE) {
  stopifnot(bits %% 8 == 0, bits >= 8)
  n <- bits %/% 8
  if (!signed) {
    return(list(
      min = as.raw(rep(0x00, n)),
      max = as.raw(rep(0xFF, n))
    ))
  }
  hi_min <- as.raw(0x80) # 1000 0000
  hi_max <- as.raw(0x7F) # 0111 1111
  zeros <- as.raw(rep(0x00, n - 1))
  ff <- as.raw(rep(0xFF, n - 1))
  list(min = c(zeros, hi_min), max = c(ff, hi_max))
}


nv_minval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(-Inf, dtype = dtype, device = device)
  } else if (dtype %in% c("i1", "pred")) {
    nv_scalar(FALSE, dtype = "pred", device = device)
  } else {
    nv_scalar(globals$ranges_raw[[dtype]]$min, dtype = dtype, device = device)
  }
}

nv_maxval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(Inf, dtype = dtype, device = device)
  } else if (dtype %in% c("i1", "pred")) {
    nv_scalar(TRUE, dtype = "pred", device = device)
  } else {
    nv_scalar(globals$ranges_raw[[dtype]]$max, dtype = dtype, device = device)
  }
}

without <- function(x, indices) {
  if (length(indices)) {
    x[-indices]
  } else {
    x
  }
}

zero_env <- function() {
  new.env(size = 0L, parent = emptyenv())
}
shape2string <- function(x) {
  paste0("(", x, ")", collapse = ",")
}
