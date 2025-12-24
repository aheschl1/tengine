
# Adding a new backend method

## Elementwise Operations

To add a new elementwise operation to the remote backend, inside of `mod.rs`, specify:

```rust
specify_trait_unary_cabal!{<name of operation> where T: <extra bounds on T>}
```

This will expand out to four methods:

```rust fn apply_<name of operation>```

```rust fn apply_<name of operation>_nd```

```rust fn try_apply_<name of operation>_1d_strided```

```rust fn try_apply_<name of operation>_contiguous```

Then, in the implmentations, you can do

```rust
impl_cpu_unary!{ <name of operation>, <inline function> where T: <extra bounds on T> }
```

## Broadcast Operations

To add a new broadcast operation to the remote backend, inside of `ops/mod.rs` add a new op variant to the `Ops` enum.

Add an implmentation of the op. Binary ops currently must be applicable to all `T`

## Remote

The remote backend is govered partially by proc macros to reduce boilerplate. To add a new method, you will need to:

1. Add an empty implmentation in `remote/client.rs`.
2. Crate a message variant matching the method signature in `remote/protocol.rs`
3. Follow the failures in server.rs. This will be simplified with another proc macro.

