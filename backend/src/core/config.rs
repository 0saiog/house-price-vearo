//! Settings, read from the environment with a `.env` file as the fallback.
//!
//! `dotenvy` does the file loading. It is a small crate, but `.env` has more
//! corners than it looks - quoting, escapes, `export` prefixes, multi-line
//! values - and a hand-rolled parser that silently mishandles one of them is a
//! configuration bug that only shows up in deployment.

use std::path::PathBuf;

/// Where a `.env` may live, in the order they are tried.
///
/// `cargo run -p backend` runs from the workspace root, so the file the README
/// describes is at `backend/.env` from there; running the binary from inside
/// `backend/` instead makes it `.env`. Checking both means the settings are
/// picked up either way rather than silently falling back to defaults.
const ENV_PATHS: [&str; 2] = ["backend/.env", ".env"];

/// Everything the service needs to start.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address to bind, e.g. `0.0.0.0:8000`.
    pub bind: String,
    /// Vearo checkpoint with the trained weights.
    pub model_path: PathBuf,
    /// Feature spec written by the training pipeline.
    pub preprocess_path: PathBuf,
    /// Browser origin allowed to call this API.
    pub allowed_origin: String,
}

impl Config {
    /// Loads settings, preferring real environment variables over the `.env` file.
    ///
    /// # Panics
    /// Panics if `PORT` is not a number - a typo there should stop the process,
    /// not silently bind somewhere unexpected.
    #[must_use]
    pub fn load() -> Self {
        // `from_filename` does not overwrite variables already in the
        // environment, so a real env var still wins over the file.
        for path in ENV_PATHS {
            if dotenvy::from_filename(path).is_ok() {
                tracing::debug!(path, "loaded .env");
                break;
            }
        }

        let get =
            |key: &str, fallback: &str| std::env::var(key).unwrap_or_else(|_| fallback.to_string());

        let host = get("HOST", "127.0.0.1");
        let port = get("PORT", "8000");
        assert!(
            port.parse::<u16>().is_ok(),
            "PORT must be a number, got {port:?}"
        );

        Self {
            bind: format!("{host}:{port}"),
            model_path: get("MODEL_PATH", "models/house_price.ve").into(),
            preprocess_path: get("PREPROCESS_PATH", "models/preprocess.json").into(),
            allowed_origin: get("ALLOWED_ORIGIN", "http://localhost:5173"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_locations_cover_both_ways_of_starting_the_service() {
        // A regression guard: reading only "." meant `cargo run -p backend` from
        // the workspace root never saw backend/.env and silently used defaults.
        assert!(ENV_PATHS.contains(&"backend/.env"));
        assert!(ENV_PATHS.contains(&".env"));
    }

    #[test]
    fn dotenvy_handles_comments_quotes_and_blanks() {
        let path = std::env::temp_dir().join("hp_vearo_env_test");
        std::fs::write(
            &path,
            "# a comment\n\nHP_TEST_PORT=9000\nHP_TEST_ORIGIN=\"http://x\"\n",
        )
        .unwrap();
        let values: std::collections::HashMap<String, String> = dotenvy::from_filename_iter(&path)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(values.get("HP_TEST_PORT").unwrap(), "9000");
        assert_eq!(values.get("HP_TEST_ORIGIN").unwrap(), "http://x");
        assert_eq!(values.len(), 2, "comments and blank lines are not entries");
        std::fs::remove_file(path).unwrap();
    }
}
