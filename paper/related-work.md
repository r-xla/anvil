Compilation of R code:

* https://github.com/nimble-dev/nCompiler
* brms: Transpiles to Stan
* nimble:
  * internally creats C++ code
  * primarily for hierarchical statistical models
* JAGS: no real "transpilation", just write native JAGS
* BUGS: Seems pretty old and outdated by JAGS, nimble
* Stan: Write native Stan code, c.f. brms for transpilation
* quickr: R -> Fortran -> native
* compiler package: Bytecode
* r2c: https://github.com/brodieG/r2c proof of concept


GPU:
* gputools: archived
* gpuR: archived
* GPUmatrix: archived

Other interesting packages:
* [greta](https://github.com/greta-dev/greta), which
  uses tensorflow as a backend.


* Autodiff
* nabla
* salad
* dual
* madness
* ADtools
