# tensors and such

Zero-dependency row‑major N‑dimensional tensor primitives.

Goal to add CUDA and optimized operations.

### todo

- [X] Backend abstraction
- [X] Nicer syntax. macro time `tset!(tensor.view_mut(), v: 99, 1, 2)` and `tget!(tensor.view(), 1, 2)`
- [ ] Slicing with ranges
- [ ] Elementwise broadcasting
- [ ] Basic linear algebra helpers
- [ ] Accelerated backends (GPU / parallel)
- [ ] x86 SIMD paths
