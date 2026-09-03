//! Service entry point.
//!
//!     cargo run --release -p backend
//!
//! Reads `.env` (see `.env.example`), loads the model once, and serves.

use backend::core::config::Config;
use backend::services::inference::Engine;

#[tokio::main]
async fn main() {
    backend::utils::logging::init();
    let config = Config::load();

    // Loading at startup, not per request: a missing model should stop the
    // process now with a clear message, not surface as a 500 on the first call.
    let engine = match Engine::load(&config.model_path, &config.preprocess_path) {
        Ok(engine) => engine,
        Err(e) => {
            tracing::error!("{e}");
            std::process::exit(1);
        }
    };
    tracing::info!(
        features = engine.spec.input_dim(),
        layers = ?engine.spec.layer_dims,
        locations = engine.locations().len(),
        "model loaded"
    );

    let app = backend::app(engine, &config.allowed_origin);
    let listener = tokio::net::TcpListener::bind(&config.bind)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {}: {e}", config.bind));
    tracing::info!("listening on http://{}", config.bind);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .expect("server error");
}
