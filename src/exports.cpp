#include <Rcpp.h>

extern "C" void* anvil_get_qr_handler(void);

// [[Rcpp::export]]
SEXP get_qr_handler() {
  return R_MakeExternalPtr(anvil_get_qr_handler(), R_NilValue, R_NilValue);
}
