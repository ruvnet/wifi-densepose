# Changelog
This documents the main changes to the `tch` crate.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Changed

## v0.22.0
### Changed
- PyTorch v2.9 support

## v0.21.0
### Changed
- PyTorch v2.8 support

## v0.20.0
### Changed
- PyTorch v2.7 support

## v0.19.0
### Changed
- PyTorch v2.6 support

## v0.18.1
### Changed
- PyTorch v2.5.1 support

## v0.18.0
### Changed
- PyTorch v2.5 support

## v0.17.0
### Changed
- PyTorch v2.4 support

## v0.16.0
### Changed
- PyTorch v2.3 support

## v0.15.0
### Changed
- PyTorch v2.2 support

## v0.14.0
### Changed
- PyTorch v2.1 support
  [803](https://github.com/LaurentMazare/tch-rs/pull/803).
- Add a `pyo3-tch` crate for interacting with Python via PyO3
  [730](https://github.com/LaurentMazare/tch-rs/pull/730).
- Expose the cuda fuser enabled flag,
  [728](https://github.com/LaurentMazare/tch-rs/pull/728).
- Improved the safetensor error wrapping,
  [720](https://github.com/LaurentMazare/tch-rs/pull/720).

## v0.13.0 - 2023-05-18
### Added
- Support static linking in the build script,
  [712](https://github.com/LaurentMazare/tch-rs/pull/712).
- Make the libtorch download opt-in rather than a default behavior. The libtorch
  library download can still be triggered by enabling the `download-libtorch`
  feature, [707](https://github.com/LaurentMazare/tch-rs/pull/707).
- Rename the `of_...` conversion functions to `from_...` so as to be closer to
  the Rust best practices,
  [706](https://github.com/LaurentMazare/tch-rs/pull/706). This is a breaking
  change and will require modifying calls such as `of_slice` to be `from_slice`
  instead.
- Expose some functions so that Python extensions that operates on PyTorch
  tensors can be written with `tch`,
  [704](https://github.com/LaurentMazare/tch-rs/pull/704).
- Rework the torch-sys build script making it easier to leverage a Python
  PyTorch install as a source for libtorch,
  [703](https://github.com/LaurentMazare/tch-rs/pull/703).

## v0.12.0 - 2023-05-10
### Changed
- EfficientNet models have been reworked, pre-trained models used `safetensors`
  weight by default, [679](https://github.com/LaurentMazare/tch-rs/pull/679).
- None can be used for nullable scalar types,
  [680](https://github.com/LaurentMazare/tch-rs/pull/680).
- Automated conversion of list arguments: all the generated functions that take
  as input a slice of int or float can now be used directly with int values or
  fixed length arrays [682](https://github.com/LaurentMazare/tch-rs/pull/682).
- Replace the `From<Tensor>` traits with some `TryFrom` versions,
  [683](https://github.com/LaurentMazare/tch-rs/pull/683). This is a breaking
  change, note that also the old version would flatten the tensor if needed to
  reduce the number of dimensions, this has to be done explicitely with the new
  version.

## v0.11.0 - 2023-03-20
### Added
- Adapt to C++ PyTorch library (`libtorch`) version `v2.0.0`.
### Changed
- Update the `half` dependency to version `2`, [646](https://github.com/LaurentMazare/tch-rs/pull/646).

## v0.10.3 - 2023-02-23
### Added
- Add some helper functions in a utils module to check the available devices and versions.

## v0.10.2 - 2023-02-19
### Added
- Add the `eps` and `amsgrad` options to Adam, [600](https://github.com/LaurentMazare/tch-rs/pull/600).
### Changed
- Fix loading of `VarStore` when using `Mps` devices, [623](https://github.com/LaurentMazare/tch-rs/pull/623).
- Use `ureq` instead of `curl` to reduce compile times, [620](https://github.com/LaurentMazare/tch-rs/pull/620).
- Fix the handling of dicts in TorchScript, [597](https://github.com/LaurentMazare/tch-rs/issues/597).

## v0.10.1 - 2022-12-12
### Changed
- Default `vs.load` to use the Python weight format when the file extension is `.pt` or `.bin`.

## v0.10.0 - 2022-12-12
### Added
- Expose functions for setting manual seeds for CUDA devices, [#500](https://github.com/LaurentMazare/tch-rs/pull/500).
- Expose functions for triggering manual sync, [#500](https://github.com/LaurentMazare/tch-rs/pull/500).
- Add some functions to load Python weight files.
### Changed
- Extending the Kaiming initialization, [#573](https://github.com/LaurentMazare/tch-rs/pull/573).

## v0.9.0 - 2022-11-05
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.13.0`.

## v0.8.0 - 2022-07-04
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.12.0`.

## v0.7.2 - 2022-05-16
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.11.0`.

## v0.6.1 - 2021-10-25
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.10.0`.

## v0.5.0 - 2021-06-25
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.9.0`.

## v0.4.1 - 2021-05-08
### Changed
- Adapt to C++ PyTorch library (`libtorch`) version `v1.8.1`.
