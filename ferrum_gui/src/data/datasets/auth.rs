//! Kaggle credential resolution and standard Base64 / Basic auth encoding.

use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

/// Minimal parser for `~/.kaggle/kaggle.json` (`{"username":"..","key":".."}`),
/// using `serde_json` (already a dependency).
pub fn parse_kaggle_json(s: &str) -> Option<KaggleCreds> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let username = v.get("username")?.as_str()?.to_string();
    let key = v.get("key")?.as_str()?.to_string();
    if username.is_empty() || key.is_empty() {
        return None;
    }
    Some(KaggleCreds { username, key })
}

/// Resolve Kaggle credentials from the environment, then from
/// `~/.kaggle/kaggle.json`.
pub fn kaggle_creds() -> Option<KaggleCreds> {
    kaggle_creds_from(|name| std::env::var(name).ok())
}

/// [`kaggle_creds`] over any environment lookup, so tests need not touch the real one.
pub(super) fn kaggle_creds_from(env: impl Fn(&str) -> Option<String>) -> Option<KaggleCreds> {
    if let (Some(username), Some(key)) = (env("KAGGLE_USERNAME"), env("KAGGLE_KEY")) {
        if !username.is_empty() && !key.is_empty() {
            return Some(KaggleCreds { username, key });
        }
    }
    let home = env("HOME").or_else(|| env("USERPROFILE"))?;
    let path = Path::new(&home).join(".kaggle").join("kaggle.json");
    let s = std::fs::read_to_string(path).ok()?;
    parse_kaggle_json(&s)
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub(super) fn basic_auth_header(user: &str, key: &str) -> String {
    format!(
        "Basic {}",
        base64_encode(format!("{user}:{key}").as_bytes())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_ds_auth_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn lookup<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn parse_kaggle_json_extracts_creds() {
        let c = parse_kaggle_json(r#"{"username":"alice","key":"deadbeef"}"#).unwrap();
        assert_eq!(c.username, "alice");
        assert_eq!(c.key, "deadbeef");
        assert!(parse_kaggle_json(r#"{"username":"alice"}"#).is_none());
        assert!(parse_kaggle_json(r#"{"username":"","key":"x"}"#).is_none());
        assert!(parse_kaggle_json("not json").is_none());
    }

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(basic_auth_header("user", "pass"), "Basic dXNlcjpwYXNz");
    }

    #[test]
    fn kaggle_credentials_come_from_the_environment_then_kaggle_json() {
        let creds = |u: &str, k: &str| {
            Some(KaggleCreds {
                username: u.into(),
                key: k.into(),
            })
        };
        let env = [("KAGGLE_USERNAME", "alice"), ("KAGGLE_KEY", "k1")];
        assert_eq!(kaggle_creds_from(lookup(&env)), creds("alice", "k1"));
        let d = scratch("kaggle_home");
        std::fs::create_dir_all(d.join(".kaggle")).unwrap();
        std::fs::write(
            d.join(".kaggle/kaggle.json"),
            r#"{"username":"bob","key":"k2"}"#,
        )
        .unwrap();
        let home = d.to_str().unwrap();
        assert_eq!(
            kaggle_creds_from(lookup(&[("HOME", home)])),
            creds("bob", "k2")
        );
        let empty = [
            ("KAGGLE_USERNAME", ""),
            ("KAGGLE_KEY", "k"),
            ("USERPROFILE", home),
        ];
        assert_eq!(kaggle_creds_from(lookup(&empty)), creds("bob", "k2"));
        assert_eq!(kaggle_creds_from(lookup(&[])), None);
        assert_eq!(
            kaggle_creds_from(lookup(&[("HOME", "/nonexistent-home")])),
            None
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
