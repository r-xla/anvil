#include <Rcpp.h>

extern "C" void *anvil_get_qr_handler(void);
extern "C" void *anvil_get_qr_handler_cuda(void);

// [[Rcpp::export]]
SEXP get_qr_handler() {
  return R_MakeExternalPtr(anvil_get_qr_handler(), R_NilValue, R_NilValue);
}

// [[Rcpp::export]]
SEXP get_qr_handler_cuda() {
  void *ptr = anvil_get_qr_handler_cuda();
  if (!ptr)
    return R_NilValue;
  return R_MakeExternalPtr(ptr, R_NilValue, R_NilValue);
}
