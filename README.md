# tensors and such

Zero-dependency row‑major N‑dimensional tensor primitives.

Goal to add CUDA and optimized operations.

## todo

- [X] Backend abstraction
- [X] Nicer syntax. macro time `tset!(tensor.view_mut(), v: 99, 1, 2)` and `tget!(tensor.view(), 1, 2)`
- [X] Basic GPU backend
- [ ] Slicing with ranges
- [X] Elementwise broadcasting
- [ ] Basic linear algebra helpers
- [ ] Accelerated backends (GPU / parallel)
- [ ] x86 SIMD paths
- [ ] Multiple gpu devices allowed

## to optimize

- [ ] Elementwise ops require computing all offsets every time. perhaps cache them
- [ ] `view_to_owned` can probably be optimized to copy larger chunks at once
- [ ] Restrict tensor values to require basic operations
- [ ] perf sucks ass. take advantage of contiguous memory buffers when available
- [ ] multiple cuda backends alive?