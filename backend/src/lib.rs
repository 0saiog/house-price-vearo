//! House price inference service.
//!
//! The model and its feature spec are loaded once, at startup, and shared by
//! every request. Building the router is a library function so the integration
//! tests can exercise the real app rather than a stand-in.

pub mod api;
pub mod core;
pub mod schemas;
pub mod services;
pub mod utils;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::services::inference::Engine;

/// State shared by every handler.
#[derive(Clone)]
pub struct AppState {
    /// The loaded model and feature spec.
    pub engine: Engine,
}

/// Builds the router.
///
/// CORS is restricted to the configured origin rather than opened to `*`: the
/// only browser that needs to call this is the project's own frontend.
pub fn app(engine: Engine, allowed_origin: &str) -> Router {
    let cors = match allowed_origin.parse::<axum::http::HeaderValue>() {
        Ok(origin) => CorsLayer::new()
            .allow_origin(origin)
            .allow_methods(Any)
            .allow_headers(Any),
        Err(_) => {
            tracing::warn!(
                origin = allowed_origin,
                "unparseable ALLOWED_ORIGIN, allowing any"
            );
            CorsLayer::permissive()
        }
    };

    Router::new()
        .route("/health", get(api::routes::prediction::health))
        .route("/locations", get(api::routes::prediction::locations))
        .route("/predict", post(api::routes::prediction::predict))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(AppState { engine })
}
