* `inline_literals`
* `typecheck_ir`
* Currently, we don't do the partial evaluation tracer
  because we don't allow to only differentiate w.r.t. all arguments,
  but if we allow to take derivative w.r.t. a subset, we should
  implement this.
* Proper mechanism for always having the jit interpreter at the bottom of the stack
  look at TODO(IMPORTANT) in interpreter.R
* better constant handling.
  I.e. each constant should apepar exactly once.
  We need to handle this in stablehlo I think.
  currently, differentiating multiplication creates one
  constant "1" for each input, which is obviously undesireable.
* Check that all teh autodiff rules are correct
* Allow for scalars (1, 1L, TRUE, FALSE) and do automatic upcasting.


* make `dtype(x)` work for `Box`.
* Make `nvl_tensor(x, dtype(y))` work.
* Add dtype to `ShapedTensor` and shaped tensor hash
* Implement LRU cache
* Input checks. Sometimes, pjrt will execute even when providng a i32 tensor to a f32 function,
  giving wrong results.
