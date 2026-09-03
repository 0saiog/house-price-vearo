//! How a listing turns into a feature vector.
//!
//! The trainer and the API both call [`FeatureSpec::encode`], so the column
//! order, the medians used for missing values, the scaling and the one-hot
//! vocabularies can't end up different between the two. The spec is fitted on
//! the training split and saved to `preprocess.json`.

use serde::{Deserialize, Serialize};

/// Category used for a value the model never saw while training.
pub const OTHER: &str = "other";

/// A listing in model terms: cleaned, but not yet scaled or encoded.
///
/// `None` means the source row did not have the value; the spec imputes it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Listing {
    /// City, lowercase, as it appears in `location`.
    pub location: String,
    /// Floor area in square feet.
    pub area_sqft: Option<f64>,
    /// Bedrooms parsed from the listing title or entered by the caller.
    pub bedrooms: Option<f64>,
    /// Neighbourhood within the city.
    pub locality: String,
    /// Named development or housing society, when known.
    pub society: String,
    /// `flat`, `apartment`, `house`, `villa`, `plot`, ...
    pub property_type: String,
    /// True when `area_sqft` came from `Carpet Area`, false when from `Super Area`.
    pub is_carpet_area: bool,
    /// Number of bathrooms.
    pub bathroom: Option<f64>,
    /// Number of balconies.
    pub balcony: Option<f64>,
    /// Number of car parking spaces.
    pub car_parking: Option<f64>,
    /// Whether any parking space is covered.
    pub parking_covered: bool,
    /// Storey the flat is on; ground is 0.
    pub floor_num: Option<f64>,
    /// Storeys in the building.
    pub total_floors: Option<f64>,
    /// `Furnished` / `Semi-Furnished` / `Unfurnished`, lowercase.
    pub furnishing: String,
    /// `Resale` / `New Property` / `Other`, lowercase.
    pub transaction: String,
    /// `Freehold` / `Leasehold` / ..., lowercase.
    pub ownership: String,
    /// Compass direction, punctuation stripped (`northeast`).
    pub facing: String,
    /// Listing mentions a garden or park.
    pub overlooking_garden: bool,
    /// Listing mentions a pool.
    pub overlooking_pool: bool,
    /// Listing mentions a main road.
    pub overlooking_main_road: bool,
}

/// Names of the numeric features, in the order they occupy in the vector.
pub const NUMERIC: [&str; 15] = [
    "log_area_sqft",
    "bedrooms",
    "bathroom",
    "log_area_per_bedroom",
    "bathrooms_per_bedroom",
    "balcony",
    "car_parking",
    "floor_num",
    "total_floors",
    "floor_ratio",
    "is_carpet_area",
    "parking_covered",
    "overlooking_garden",
    "overlooking_pool",
    "overlooking_main_road",
];

/// A one-hot encoded column and the vocabulary it was fitted with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Categorical {
    /// Feature name, e.g. `location`.
    pub name: String,
    /// Accepted values. Always ends with [`OTHER`].
    pub vocab: Vec<String>,
}

/// Encodes a column by what its values are worth instead of one-hot.
///
/// One-hot on a column with 500 values costs 500 input columns to say something
/// that mostly boils down to "this area is expensive". Using the average log
/// price per sqft for each value says it in two numbers, and rare values get a
/// lot less room to overfit.
///
/// The average is pulled toward the overall average so a locality that only
/// appears three times can't claim a confident price. The row count goes in as
/// a second feature so the model can figure out how much to trust it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetEncoding {
    /// The categorical column this encodes, e.g. `locality`.
    pub name: String,
    /// Level to `[smoothed mean ln(price per sqft), ln(1 + rows)]`.
    pub levels: std::collections::HashMap<String, [f32; 2]>,
    /// Used for a level the training split never saw.
    pub fallback: [f32; 2],
    /// Training-split mean of each encoded column, for scaling.
    pub mean: [f32; 2],
    /// Training-split standard deviation of each encoded column.
    pub std: [f32; 2],
}

impl TargetEncoding {
    /// The scaled pair of features for one listing.
    #[must_use]
    pub fn apply(&self, level: &str) -> [f32; 2] {
        let raw = self.levels.get(level).copied().unwrap_or(self.fallback);
        let mut out = [0.0f32; 2];
        for i in 0..2 {
            let std = if self.std[i] > 1e-8 { self.std[i] } else { 1.0 };
            out[i] = (raw[i] - self.mean[i]) / std;
        }
        out
    }
}

/// How the target was transformed before training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    /// Train on `ln(1 + price)`, predict with `exp(y) - 1`.
    Log1p,
    /// Train on `ln(price / area)`, then multiply the decoded rate by area.
    LogPricePerSqft,
    /// Train on rupees directly.
    Raw,
}

/// Hidden-layer non-linearity stored with the exported model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Rectified linear unit, kept as the backwards-compatible default.
    #[default]
    Relu,
    /// Gaussian error linear unit.
    Gelu,
}

impl Target {
    /// Maps a price in rupees into the space the model is trained in.
    #[must_use]
    pub fn forward(self, price: f64, area_sqft: f64) -> f64 {
        match self {
            Self::Log1p => price.ln_1p(),
            Self::LogPricePerSqft => (price / area_sqft).ln(),
            Self::Raw => price,
        }
    }

    /// Maps a model output back to rupees.
    #[must_use]
    pub fn inverse(self, y: f64, area_sqft: f64) -> f64 {
        match self {
            Self::Log1p => y.exp_m1(),
            Self::LogPricePerSqft => y.exp() * area_sqft,
            Self::Raw => y,
        }
    }
}

/// Everything needed to turn a [`Listing`] into the exact vector the model was
/// trained on. Serialised to `models/preprocess.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Numeric feature names, matching [`NUMERIC`].
    pub numeric: Vec<String>,
    /// Value substituted for a missing numeric feature (training-split median).
    pub numeric_median: Vec<f32>,
    /// Training-split mean, subtracted when scaling.
    pub numeric_mean: Vec<f32>,
    /// Training-split standard deviation, divided out when scaling.
    pub numeric_std: Vec<f32>,
    /// One-hot columns, in vector order, after the numeric block.
    pub categorical: Vec<Categorical>,
    /// Target transform used during training.
    pub target: Target,
    /// Mean of the transformed target on the training split.
    pub target_mean: f32,
    /// Standard deviation of the transformed target on the training split.
    pub target_std: f32,
    /// Layer widths of the trained network, input first and output last.
    pub layer_dims: Vec<usize>,
    /// Hidden-layer activation used by the trained network.
    #[serde(default)]
    pub activation: Activation,
    /// Hidden activation dropout probability used during training.
    #[serde(default)]
    pub dropout: f32,
    /// Whether hidden layers include trainable layer normalisation.
    #[serde(default)]
    pub layer_norm: bool,
    /// High-cardinality columns encoded by their target statistics.
    #[serde(default)]
    pub target_encodings: Vec<TargetEncoding>,
}

impl FeatureSpec {
    /// Maps a price in rupees to the number the network is trained to output.
    ///
    /// The transform is followed by standardisation so that both target choices
    /// reach the optimiser on the same scale; without it, training on raw rupees
    /// squares values around 1e7 into an f32 loss near 1e14 and the comparison
    /// would measure float overflow rather than the modelling choice.
    #[must_use]
    pub fn encode_target(&self, price: f64, area_sqft: f64) -> f32 {
        ((self.target.forward(price, area_sqft) - f64::from(self.target_mean))
            / f64::from(self.target_std.max(1e-8))) as f32
    }

    /// Maps a network output back to rupees.
    #[must_use]
    pub fn decode_target(&self, y: f64, area_sqft: f64) -> f64 {
        self.target.inverse(
            y * f64::from(self.target_std.max(1e-8)) + f64::from(self.target_mean),
            area_sqft,
        )
    }

    /// Length of the encoded vector.
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.numeric.len()
            + 2 * self.target_encodings.len()
            + self
                .categorical
                .iter()
                .map(|c| c.vocab.len())
                .sum::<usize>()
    }

    /// The raw (unscaled, unimputed) numeric features of a listing, in
    /// [`NUMERIC`] order. `NaN` marks a missing value.
    #[must_use]
    pub fn raw_numeric(listing: &Listing) -> [f32; NUMERIC.len()] {
        let opt = |v: Option<f64>| v.map_or(f32::NAN, |x| x as f32);
        let floor_ratio = match (listing.floor_num, listing.total_floors) {
            (Some(n), Some(t)) if t > 0.0 => Some(n / t),
            _ => None,
        };
        let area_per_bedroom = match (listing.area_sqft, listing.bedrooms) {
            (Some(a), Some(b)) if b > 0.0 => Some((a / b).ln_1p()),
            _ => None,
        };
        let bathrooms_per_bedroom = match (listing.bathroom, listing.bedrooms) {
            (Some(a), Some(b)) if b > 0.0 => Some(a / b),
            _ => None,
        };
        [
            listing.area_sqft.map_or(f32::NAN, |a| a.ln_1p() as f32),
            opt(listing.bedrooms),
            opt(listing.bathroom),
            opt(area_per_bedroom),
            opt(bathrooms_per_bedroom),
            opt(listing.balcony),
            opt(listing.car_parking),
            opt(listing.floor_num),
            opt(listing.total_floors),
            opt(floor_ratio),
            f32::from(listing.is_carpet_area),
            f32::from(listing.parking_covered),
            f32::from(listing.overlooking_garden),
            f32::from(listing.overlooking_pool),
            f32::from(listing.overlooking_main_road),
        ]
    }

    /// Returns one numeric input by its serialised feature name.
    ///
    /// Matching by name keeps older feature specs loadable while new versions
    /// add columns: an old checkpoint simply never asks for the new values.
    #[must_use]
    pub fn numeric_value(name: &str, listing: &Listing) -> f32 {
        let values = Self::raw_numeric(listing);
        NUMERIC
            .iter()
            .position(|candidate| *candidate == name)
            .map_or(f32::NAN, |i| values[i])
    }

    /// Returns one categorical input by its serialised feature name.
    #[must_use]
    pub fn categorical_value<'a>(name: &str, listing: &'a Listing) -> &'a str {
        match name {
            "location" => listing.location.as_str(),
            "locality" => listing.locality.as_str(),
            "society" => listing.society.as_str(),
            "property_type" => listing.property_type.as_str(),
            "furnishing" => listing.furnishing.as_str(),
            "transaction" => listing.transaction.as_str(),
            "ownership" => listing.ownership.as_str(),
            "facing" => listing.facing.as_str(),
            _ => OTHER,
        }
    }

    /// Encodes a listing: impute, scale, then append the one-hot blocks.
    ///
    /// Anything not in the vocabulary goes to [`OTHER`]. That's what lets the
    /// API take a city the model has never seen without either failing or,
    /// worse, quietly shifting every column after it.
    #[must_use]
    pub fn encode(&self, listing: &Listing) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.input_dim());

        for (i, name) in self.numeric.iter().enumerate() {
            let value = Self::numeric_value(name, listing);
            let imputed = if value.is_finite() {
                value
            } else {
                self.numeric_median[i]
            };
            let std = if self.numeric_std[i] > 1e-8 {
                self.numeric_std[i]
            } else {
                1.0
            };
            out.push((imputed - self.numeric_mean[i]) / std);
        }

        for encoding in &self.target_encodings {
            let level = Self::categorical_value(&encoding.name, listing);
            out.extend(encoding.apply(level.trim()));
        }

        for spec in &self.categorical {
            let value = Self::categorical_value(&spec.name, listing);
            let value = value.trim().to_lowercase();
            let hit = spec.vocab.iter().position(|v| *v == value);
            let hit = hit.unwrap_or_else(|| spec.vocab.len() - 1); // trailing OTHER
            out.extend(std::iter::repeat_n(0.0, spec.vocab.len()));
            let base = out.len() - spec.vocab.len();
            out[base + hit] = 1.0;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> FeatureSpec {
        FeatureSpec {
            numeric: NUMERIC.iter().map(ToString::to_string).collect(),
            numeric_median: vec![7.0; NUMERIC.len()],
            numeric_mean: vec![0.0; NUMERIC.len()],
            numeric_std: vec![1.0; NUMERIC.len()],
            categorical: vec![Categorical {
                name: "location".into(),
                vocab: vec!["mumbai".into(), OTHER.into()],
            }],
            target: Target::Log1p,
            target_mean: 0.0,
            target_std: 1.0,
            layer_dims: vec![NUMERIC.len() + 2, 8, 1],
            activation: Activation::Relu,
            dropout: 0.0,
            layer_norm: false,
            target_encodings: Vec::new(),
        }
    }

    #[test]
    fn encodes_to_a_fixed_width_vector() {
        let s = spec();
        let listing = Listing {
            location: "mumbai".into(),
            area_sqft: Some(1000.0),
            ..Listing::default()
        };
        let v = s.encode(&listing);
        assert_eq!(v.len(), s.input_dim());
        assert_eq!(
            &v[NUMERIC.len()..],
            &[1.0, 0.0],
            "known city one-hots in place"
        );
    }

    #[test]
    fn unseen_category_falls_into_other() {
        let s = spec();
        let listing = Listing {
            location: "Atlantis".into(),
            ..Listing::default()
        };
        let v = s.encode(&listing);
        assert_eq!(&v[NUMERIC.len()..], &[0.0, 1.0]);
        assert_eq!(v.len(), s.input_dim(), "width is stable whatever the input");
    }

    #[test]
    fn missing_numerics_take_the_median() {
        let s = spec();
        let v = s.encode(&Listing {
            location: "mumbai".into(),
            ..Listing::default()
        });
        assert!(
            (v[1] - 7.0).abs() < 1e-6,
            "absent bathroom count imputed, not zeroed"
        );
    }

    #[test]
    fn target_round_trips_through_transform_and_scaling() {
        let mut s = spec();
        s.target_mean = 15.2;
        s.target_std = 0.8;
        let price = 4_200_000.0;
        let area = 1_200.0;
        let back = s.decode_target(f64::from(s.encode_target(price, area)), area);
        assert!((back - price).abs() / price < 1e-3, "got {back}");
    }
}
