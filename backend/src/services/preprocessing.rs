//! Turns an API request into the exact structure the trainer used.
//!
//! There is no encoding logic here on purpose. The request is mapped onto
//! `hp_core::Listing` and the shared `FeatureSpec::encode` does the rest, so the
//! service cannot one-hot a column in a different order than training did.

use hp_core::Listing;
use hp_core::clean::{normalize_category, normalize_facing};

use crate::schemas::prediction::PredictionRequest;

/// Maps a validated request onto a cleaned listing.
///
/// The same normalisation the training pipeline applied is applied here, which
/// is why `"Semi-Furnished"` from a form matches `semi-furnished` in the
/// vocabulary, and why an omitted `facing` becomes the `missing` category the
/// model was actually trained with rather than an unseen empty string.
#[must_use]
pub fn to_listing(req: &PredictionRequest) -> Listing {
    Listing {
        location: normalize_category(&req.location),
        area_sqft: Some(req.area_sqft),
        bedrooms: None,
        locality: "missing".into(),
        society: "missing".into(),
        property_type: "other".into(),
        is_carpet_area: req.is_carpet_area,
        bathroom: req.bathroom,
        balcony: req.balcony,
        car_parking: req.car_parking,
        parking_covered: req.parking_covered,
        floor_num: req.floor_num,
        total_floors: req.total_floors,
        furnishing: normalize_category(req.furnishing.as_str()),
        transaction: normalize_category(req.transaction.as_str()),
        ownership: normalize_category(req.ownership.as_deref().unwrap_or("")),
        facing: normalize_facing(req.facing.as_deref().unwrap_or("")),
        overlooking_garden: req.overlooking_garden,
        overlooking_pool: req.overlooking_pool,
        overlooking_main_road: req.overlooking_main_road,
    }
}
