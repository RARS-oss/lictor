# SPDX-License-Identifier: MIT
# lictor -- build entry points. Every target refuses to run unless CARGO_TARGET_DIR is set:
# drive C: is full, so builds go to an explicit target directory, e.g.
#   export CARGO_TARGET_DIR=/mnt/d/lictor/target
# (.cargo/config.toml deliberately sets no target-dir; harness/env.sh exports the variable.)

.PHONY: guard check build bench selftest ci demo nostd-check

guard:
	@$(if $(CARGO_TARGET_DIR),,$(error set CARGO_TARGET_DIR (C: is full), e.g. export CARGO_TARGET_DIR=/mnt/d/lictor/target))true

check: guard
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace

build: guard
	cargo build --release -p lictor-cli

bench: build
	$(CARGO_TARGET_DIR)/release/lictor bench --n 200000

selftest: build
	$(CARGO_TARGET_DIR)/release/lictor selftest

ci: check build
	bash bench/run_all.sh "$(CARGO_TARGET_DIR)/release/lictor"
	bash scripts/ci_python.sh

demo: guard
	bash scripts/demo.sh

# core/detect/fuse without std: detect and fuse depend on core with default-features = false, so this really exercises no_std.
nostd-check: guard
	cargo check -p lictor-core -p lictor-detect -p lictor-fuse --no-default-features
