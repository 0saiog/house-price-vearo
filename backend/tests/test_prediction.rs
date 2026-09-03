//! Integration tests against the real router, with the real trained model.
//!
//! These need `models/` to exist, which is what `cargo run --release -p ml`
//! writes. That is deliberate: a test suite that stubbed the model out would
//! pass while the service returned nonsense.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use backend::services::inference::Engine;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Builds the app with the checkpoint from the repository root.
fn app() -> axum::Router {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let engine = Engine::load(
        &root.join("models/house_price.ve"),
        &root.join("models/preprocess.json"),
    )
    .expect("models/ is missing - run `cargo run --release -p ml` first");
    backend::app(engine, "http://localhost:5173")
}

/// Sends one request and returns the status and parsed body.
async fn send(request: Request<Body>) -> (StatusCode, Value) {
    let response = app().oneshot(request).await.expect("router responded");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn post(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/predict")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn valid_flat() -> Value {
    json!({
        "location": "mumbai",
        "area_sqft": 1000.0,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "bathroom": 2,
        "balcony": 1,
        "floor_num": 5,
        "total_floors": 12
    })
}

#[tokio::test]
async fn health_reports_a_loaded_model() {
    let (status, body) = send(
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model_loaded"], true);
    assert!(body["features"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn happy_path_returns_a_plausible_price() {
    let (status, body) = send(post(valid_flat())).await;
    assert_eq!(status, StatusCode::OK);

    let price = body["predicted_price"]
        .as_f64()
        .expect("predicted_price is a number");
    assert!(price > 0.0, "price must be positive, got {price}");
    // A 1000 sqft flat in Mumbai is somewhere between 10 Lac and 100 Cr. Wide on
    // purpose: this asserts the units are rupees, not that the model is good.
    assert!(
        (1e6..1e10).contains(&price),
        "price {price} is not in a sane rupee range"
    );
    assert!(
        body["predicted_price_formatted"]
            .as_str()
            .unwrap()
            .contains(['L', 'C'])
    );
    assert_eq!(body["currency"], "INR");
    assert_eq!(body["location_known"], true);
}

#[tokio::test]
async fn a_body_missing_a_required_field_is_rejected_with_422() {
    let mut body = valid_flat();
    body.as_object_mut().unwrap().remove("area_sqft");
    let (status, body) = send(post(body)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"].as_str().unwrap().contains("area_sqft"));
}

#[tokio::test]
async fn a_nonsense_area_is_rejected_with_422() {
    let mut body = valid_flat();
    body["area_sqft"] = json!(-50.0);
    let (status, body) = send(post(body)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["detail"], "area_sqft must be a positive number");
}

#[tokio::test]
async fn an_unseen_city_is_answered_and_flagged() {
    let mut body = valid_flat();
    body["location"] = json!("atlantis");
    let (status, body) = send(post(body)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "an unknown city must not be an error"
    );
    assert_eq!(body["location_known"], false, "but the caller must be told");
    assert!(body["predicted_price"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn a_bigger_flat_in_the_same_city_predicts_more() {
    let app = app();
    let mut small = valid_flat();
    small["area_sqft"] = json!(600.0);
    let mut large = valid_flat();
    large["area_sqft"] = json!(2400.0);

    let get_price = |app: axum::Router, body: Value| async move {
        let response = app.oneshot(post(body)).await.unwrap();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<Value>(&bytes).unwrap()["predicted_price"]
            .as_f64()
            .unwrap()
    };

    let small_price = get_price(app.clone(), small).await;
    let large_price = get_price(app, large).await;
    assert!(
        large_price > small_price,
        "4x the area predicted less: {large_price} vs {small_price}"
    );
}

#[tokio::test]
async fn locations_lists_only_cities_the_model_knows() {
    let (status, body) = send(
        Request::builder()
            .uri("/locations")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let cities = body.as_array().expect("an array of cities");
    assert!(!cities.is_empty());
    assert!(
        cities.iter().all(|c| c != "other"),
        "the catch-all bucket is not a choice"
    );
    assert!(cities.iter().any(|c| c == "mumbai"));
}
