test: .FORCE
	cargo test

clean: .FORCE
	cargo clean

gen: .FORCE
	dune exec gen/gen.exe
	rustfmt src/wrappers/tensor_fallible_generated.rs
	rustfmt src/wrappers/tensor_generated.rs
	rustfmt torch-sys/src/c_generated.rs

.FORCE:
