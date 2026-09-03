//! Tracing setup.

/// Installs the log subscriber, honouring `RUST_LOG`.
///
/// Safe to call more than once; a second call is a no-op rather than a panic,
/// which matters because the integration tests build the app repeatedly.
pub fn init() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("backend=info,tower_http=info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
