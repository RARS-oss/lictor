// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! Ed25519 seed files: generation (getrandom), storage (0600 on unix), default directory.

use crate::sign::ReceiptError;

/// getrandom
pub fn keygen() -> [u8; 32] {
    todo!("WP-4")
}

pub fn pubkey_hex(_seed: &[u8; 32]) -> String {
    todo!("WP-4")
}

/// 64 hex chars
pub fn load_seed(_path: &std::path::Path) -> Result<[u8; 32], ReceiptError> {
    todo!("WP-4")
}

/// mode 0600 on unix (behind `#[cfg(unix)]` via std::os::unix::fs::PermissionsExt; a no-op with a stderr warning elsewhere).
/// Refuses (ReceiptError::Key) a path under `/mnt/[a-z]/` (WSL DrvFs does not enforce 0600) unless `allow_drvfs`; refuses to
/// overwrite an existing file unless `force`.
pub fn save_seed(
    _path: &std::path::Path,
    _seed: &[u8; 32],
    _force: bool,
    _allow_drvfs: bool,
) -> Result<(), ReceiptError> {
    todo!("WP-4")
}

/// `$LICTOR_KEYS` if set, else `$HOME/.lictor` (ext4 on this machine) -- NEVER the repo directory.
pub fn default_key_dir() -> std::path::PathBuf {
    todo!("WP-4")
}
