// SPDX-License-Identifier: MIT
//! Ed25519 seed files: generation (getrandom), storage (0600 on unix), default directory.
//!
//! A seed file is 64 lowercase hex characters on one line; lines starting with `#` are comments (the committed
//! TEST keys say so in a comment). Keys never live in the repo or under `/mnt/[a-z]/` (WSL DrvFs ignores 0600).

use std::io::Write;
use std::path::{Path, PathBuf};

use ed25519_dalek::SigningKey;

use crate::sign::ReceiptError;

/// getrandom
pub fn keygen() -> [u8; 32] {
    let mut seed = [0u8; 32];
    // The OS RNG failing is not a data error a caller can act on; it is an environment fault.
    getrandom::getrandom(&mut seed).expect("os rng");
    seed
}

pub fn pubkey_hex(seed: &[u8; 32]) -> String {
    hex::encode(SigningKey::from_bytes(seed).verifying_key().to_bytes())
}

/// Parse seed text: comment lines (`#`) and blank lines are ignored; exactly one 64-hex-char line must remain.
pub fn parse_seed(text: &str) -> Result<[u8; 32], ReceiptError> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    let line = lines.next().ok_or_else(|| ReceiptError::Key("seed file has no hex line".into()))?;
    if lines.next().is_some() {
        return Err(ReceiptError::Key("seed file has more than one non-comment line".into()));
    }
    if line.len() != 64 || !line.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(ReceiptError::Key("seed must be exactly 64 hex characters".into()));
    }
    let bytes = hex::decode(line).map_err(|e| ReceiptError::Key(e.to_string()))?;
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes);
    Ok(seed)
}

/// 64 hex chars
pub fn load_seed(path: &Path) -> Result<[u8; 32], ReceiptError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| ReceiptError::Io(format!("{}: {e}", path.display())))?;
    parse_seed(&text).map_err(|e| match e {
        ReceiptError::Key(m) => ReceiptError::Key(format!("{}: {m}", path.display())),
        other => other,
    })
}

/// True for a path under a WSL DrvFs mount (`/mnt/<letter>/...`), checked on the path as given and, for a
/// relative path, on its absolute form.
pub fn is_drvfs(path: &Path) -> bool {
    fn text_is_drvfs(s: &str) -> bool {
        let s = s.replace('\\', "/");
        let b = s.as_bytes();
        b.len() >= 6 && s.starts_with("/mnt/") && b[5].is_ascii_lowercase() && b[6..].first() == Some(&b'/')
    }
    if text_is_drvfs(&path.to_string_lossy()) {
        return true;
    }
    if path.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            return text_is_drvfs(&cwd.join(path).to_string_lossy());
        }
    }
    false
}

/// mode 0600 on unix (behind `#[cfg(unix)]` via std::os::unix::fs::PermissionsExt; a no-op with a stderr warning elsewhere).
/// Refuses (ReceiptError::Key) a path under `/mnt/[a-z]/` (WSL DrvFs does not enforce 0600) unless `allow_drvfs`; refuses to
/// overwrite an existing file unless `force`.
pub fn save_seed(path: &Path, seed: &[u8; 32], force: bool, allow_drvfs: bool) -> Result<(), ReceiptError> {
    if !allow_drvfs && is_drvfs(path) {
        return Err(ReceiptError::Key(format!(
            "{} is under /mnt/<drive>/ (DrvFs ignores 0600); use $LICTOR_KEYS on ext4 or pass --i-know",
            path.display()
        )));
    }
    if path.exists() && !force {
        return Err(ReceiptError::Key(format!("{} exists; pass --force to overwrite", path.display())));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = open_private(path)?;
    f.write_all(hex::encode(seed).as_bytes())?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    set_private(path)?;
    Ok(())
}

#[cfg(unix)]
fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)
}

#[cfg(not(unix))]
fn open_private(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::create(path)
}

#[cfg(unix)]
fn set_private(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_private(path: &Path) -> std::io::Result<()> {
    eprintln!("warning: {}: file mode 0600 is not enforced on this platform", path.display());
    Ok(())
}

/// `$LICTOR_KEYS` if set, else `$HOME/.lictor` (ext4 on this machine) -- NEVER the repo directory.
pub fn default_key_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("LICTOR_KEYS").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".lictor")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_comments_and_rejects_junk() {
        let seed =
            parse_seed("# test key\n000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\n")
                .unwrap();
        assert_eq!(seed[0..4], [0, 1, 2, 3]);
        assert!(parse_seed("").is_err());
        assert!(parse_seed("abc").is_err());
        assert!(parse_seed(&format!("{}\n{}", "0".repeat(64), "1".repeat(64))).is_err());
        assert!(parse_seed(&"g".repeat(64)).is_err());
    }

    #[test]
    fn drvfs_detection() {
        assert!(is_drvfs(Path::new("/mnt/c/Users/x/key.hex")));
        assert!(is_drvfs(Path::new("/mnt/d/lictor/key.hex")));
        assert!(!is_drvfs(Path::new("/mnt/wsl/key.hex")));
        assert!(!is_drvfs(Path::new("/home/u/.lictor/key.hex")));
        assert!(!is_drvfs(Path::new("/mnt/C/x")));
    }

    #[test]
    fn keygen_is_random_and_pubkey_is_hex64() {
        let a = keygen();
        let b = keygen();
        assert_ne!(a, b);
        assert_eq!(pubkey_hex(&a).len(), 64);
    }
}
