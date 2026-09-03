//! `GET /health`, `GET /locations`, `POST /predict`.

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::AppState;
use crate::schemas::prediction::{
    ErrorResponse, HealthResponse, PredictionRequest, PredictionResponse, format_rupees,
};
use crate::services::preprocessing::to_listing;

/// An error the caller can act on.
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorResponse { detail: self.1 })).into_response()
    }
}

/// Liveness plus what the service has loaded.
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model_loaded: true,
        features: state.engine.spec.input_dim(),
        layers: state.engine.spec.layer_dims.clone(),
    })
}

/// The cities the model has a column for.
///
/// The frontend populates its dropdown from here rather than from a copy of
/// `locations.json`, so the options can never list a city the loaded model does
/// not actually know.
pub async fn locations(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(
        state
            .engine
            .locations()
            .into_iter()
            .map(ToString::to_string)
            .collect(),
    )
}

/// Prices one property.
///
/// A body that does not match the schema is rejected by axum with 422 before
/// this runs; a body that parses but makes no sense is rejected here, also 422.
pub async fn predict(
    State(state): State<AppState>,
    payload: Result<Json<PredictionRequest>, JsonRejection>,
) -> Result<Json<PredictionResponse>, ApiError> {
    let Json(request) =
        payload.map_err(|e| ApiError(StatusCode::UNPROCESSABLE_ENTITY, e.body_text()))?;
    request
        .validate()
        .map_err(|detail| ApiError(StatusCode::UNPROCESSABLE_ENTITY, detail))?;

    let listing = to_listing(&request);
    let location_known = state.engine.knows_location(&request.location);

    // The forward pass is CPU-bound and takes a lock, so it does not belong on
    // the async runtime's threads.
    let engine = state.engine.clone();
    let price = tokio::task::spawn_blocking(move || engine.predict(&listing))
        .await
        .map_err(|e| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inference failed: {e}"),
            )
        })?;

    tracing::info!(location = %request.location, area = request.area_sqft, price, "predicted");

    Ok(Json(PredictionResponse {
        predicted_price: price,
        predicted_price_formatted: format_rupees(price),
        currency: "INR",
        location_known,
    }))
}
