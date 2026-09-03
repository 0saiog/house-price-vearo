//! Request and response bodies.
//!
//! The request mirrors the model's input features exactly. Everything the model
//! can impute is optional, so a caller who does not know the balcony count sends
//! nothing and gets the training median rather than being forced to invent a zero.

use serde::{Deserialize, Serialize};

/// A property to price.
#[derive(Debug, Clone, Deserialize)]
pub struct PredictionRequest {
    /// City, as listed by `GET /locations`.
    pub location: String,
    /// Floor area in square feet. Must be positive.
    pub area_sqft: f64,
    /// `Furnished`, `Semi-Furnished` or `Unfurnished`.
    pub furnishing: String,
    /// `Resale` or `New Property`.
    pub transaction: String,
    /// Whether `area_sqft` is carpet area (the default) or super area.
    #[serde(default = "default_true")]
    pub is_carpet_area: bool,
    /// Number of bathrooms.
    #[serde(default)]
    pub bathroom: Option<f64>,
    /// Number of balconies.
    #[serde(default)]
    pub balcony: Option<f64>,
    /// Number of car parking spaces.
    #[serde(default)]
    pub car_parking: Option<f64>,
    /// Whether the parking is covered.
    #[serde(default)]
    pub parking_covered: bool,
    /// Storey the flat is on; ground is 0.
    #[serde(default)]
    pub floor_num: Option<f64>,
    /// Storeys in the building.
    #[serde(default)]
    pub total_floors: Option<f64>,
    /// Ownership type, e.g. `Freehold`.
    #[serde(default)]
    pub ownership: Option<String>,
    /// Compass direction the property faces.
    #[serde(default)]
    pub facing: Option<String>,
    /// Overlooks a garden or park.
    #[serde(default)]
    pub overlooking_garden: bool,
    /// Overlooks a pool.
    #[serde(default)]
    pub overlooking_pool: bool,
    /// Overlooks a main road.
    #[serde(default)]
    pub overlooking_main_road: bool,
}

const fn default_true() -> bool {
    true
}

impl PredictionRequest {
    /// Rejects values the model cannot mean anything sensible for.
    ///
    /// # Errors
    /// Returns a human-readable message naming the offending field.
    pub fn validate(&self) -> Result<(), String> {
        if self.location.trim().is_empty() {
            return Err("location must not be empty".into());
        }
        if !self.area_sqft.is_finite() || self.area_sqft <= 0.0 {
            return Err("area_sqft must be a positive number".into());
        }
        if self.area_sqft > 1_000_000.0 {
            return Err("area_sqft is implausibly large (over 1,000,000 sqft)".into());
        }
        for (name, value) in [
            ("bathroom", self.bathroom),
            ("balcony", self.balcony),
            ("car_parking", self.car_parking),
            ("total_floors", self.total_floors),
        ] {
            if let Some(v) = value
                && (!v.is_finite() || v < 0.0)
            {
                return Err(format!("{name} must not be negative"));
            }
        }
        if self.floor_num.is_some_and(|f| !f.is_finite()) {
            return Err("floor_num must be a number".into());
        }
        Ok(())
    }
}

/// The predicted price.
#[derive(Debug, Clone, Serialize)]
pub struct PredictionResponse {
    /// Predicted sale price in rupees.
    pub predicted_price: f64,
    /// The same number written the way the listings write it, e.g. `42.50 Lac`.
    pub predicted_price_formatted: String,
    /// Currency of `predicted_price`.
    pub currency: &'static str,
    /// True when the city was one the model has a column for, false when it fell
    /// into the `other` bucket - the caller deserves to know the prediction is
    /// weaker in that case.
    pub location_known: bool,
}

/// Service liveness and what it has loaded.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    /// Always `ok` when the service can answer.
    pub status: &'static str,
    /// Whether the weights and the feature spec are loaded.
    pub model_loaded: bool,
    /// Encoded feature width the model expects.
    pub features: usize,
    /// Layer widths of the loaded network.
    pub layers: Vec<usize>,
}

/// An error, as JSON.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// What went wrong, in a sentence.
    pub detail: String,
}

/// Formats rupees in Indian listing shorthand.
#[must_use]
pub fn format_rupees(value: f64) -> String {
    if value >= 1e7 {
        format!("{:.2} Cr", value / 1e7)
    } else if value >= 1e5 {
        format!("{:.2} Lac", value / 1e5)
    } else {
        format!("{value:.0}")
    }
}
