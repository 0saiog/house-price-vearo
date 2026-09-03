//! Reading the CSV and cleaning it into rows we can train on.
//!
//! The `csv` crate handles the parsing. The `Description` column is a paragraph
//! of prose full of commas and quotes, so splitting on commas is not an option
//! and writing an RFC 4180 parser by hand isn't worth it. It also deserialises
//! into a struct, so a renamed column fails loudly instead of quietly becoming
//! an empty feature.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use hp_core::clean;
use hp_core::features::{
    Activation, Categorical, FeatureSpec, Listing, NUMERIC, OTHER, Target, TargetEncoding,
};
use serde::Deserialize;
use vearo::nn::SimpleRng;

/// One row of the source file, named by its CSV header.
///
/// Serde does the column mapping, so `Amount(in rupees)` is bound once here
/// rather than looked up by string on every row.
#[derive(Debug, Deserialize)]
pub struct RawListing {
    #[serde(rename = "Amount(in rupees)")]
    pub amount: String,
    #[serde(rename = "Carpet Area")]
    pub carpet_area: String,
    #[serde(rename = "Super Area")]
    pub super_area: String,
    #[serde(rename = "location")]
    pub location: String,
    #[serde(rename = "Floor")]
    pub floor: String,
    #[serde(rename = "Bathroom")]
    pub bathroom: String,
    #[serde(rename = "Balcony")]
    pub balcony: String,
    #[serde(rename = "Car Parking")]
    pub car_parking: String,
    #[serde(rename = "Furnishing")]
    pub furnishing: String,
    #[serde(rename = "Transaction")]
    pub transaction: String,
    #[serde(rename = "Ownership")]
    pub ownership: String,
    #[serde(rename = "facing")]
    pub facing: String,
    #[serde(rename = "overlooking")]
    pub overlooking: String,
    #[serde(rename = "Society")]
    pub society: String,
    #[serde(rename = "Title")]
    pub title: String,
    #[serde(rename = "Status")]
    pub status: String,
    #[serde(rename = "Price (in rupees)")]
    pub rate: String,
    #[serde(rename = "Dimensions")]
    pub dimensions: String,
    #[serde(rename = "Plot Area")]
    pub plot_area: String,
    #[serde(rename = "Description")]
    pub description: String,
}

/// Reads the whole file.
///
/// # Errors
/// Returns an error if the file cannot be opened, or if a row does not match
/// the expected header.
pub fn read(path: impl AsRef<Path>) -> Result<Vec<RawListing>> {
    let path = path.as_ref();
    let mut reader = csv::Reader::from_path(path).with_context(|| {
        format!(
            "cannot read {}. See the README for how to fetch the dataset",
            path.display()
        )
    })?;
    reader
        .deserialize()
        .collect::<Result<Vec<RawListing>, _>>()
        .with_context(|| format!("malformed row in {}", path.display()))
}

/// A usable training row: the cleaned listing and its price in rupees.
#[derive(Clone)]
pub struct Row {
    /// Cleaned features.
    pub listing: Listing,
    /// Target, in rupees.
    pub price: f64,
    /// Derived rate, for outlier trimming and the report.
    pub price_per_sqft: f64,
}

/// What cleaning removed, for the report and the console log.
#[derive(Default)]
pub struct CleaningLog {
    /// Rows in the source file.
    pub total: usize,
    /// Dropped because the price was "Call for Price" or unparseable.
    pub no_price: usize,
    /// Dropped because neither area column had a usable value.
    pub no_area: usize,
    /// Dropped as exact repeats of a listing already kept.
    pub duplicates: usize,
    /// Dropped by the price-per-sqft outlier trim.
    pub outliers: usize,
    /// Lower price-per-sqft bound kept.
    pub ppsf_low: f64,
    /// Upper price-per-sqft bound kept.
    pub ppsf_high: f64,
    /// Rows whose area came from `Carpet Area` rather than `Super Area`.
    pub from_carpet: usize,
}

/// The value at a percentile of `sorted`, which must be sorted ascending.
#[must_use]
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Cleans every row, optionally de-duplicates, and trims price-per-sqft outliers.
///
/// `dedup` should basically always be on. About three fifths of the rows that
/// get this far are duplicate listings, and one showing up in both splits turns
/// the test set into a memory test. It's a parameter so the report can show what
/// leaving them in would have looked like.
#[must_use]
pub fn build(raw: &[RawListing], dedup: bool) -> (Vec<Row>, CleaningLog) {
    let mut log = CleaningLog {
        total: raw.len(),
        ..CleaningLog::default()
    };
    let mut rows = Vec::with_capacity(raw.len());

    for r in raw {
        let Some(price) = clean::parse_amount(&r.amount) else {
            log.no_price += 1;
            continue;
        };
        // Carpet area is the honest number; super area includes shared space, so
        // prefer carpet and flag which one this row used.
        let carpet = clean::parse_area_sqft(&r.carpet_area);
        let is_carpet_area = carpet.is_some();
        let Some(area) = carpet.or_else(|| clean::parse_area_sqft(&r.super_area)) else {
            log.no_area += 1;
            continue;
        };
        if is_carpet_area {
            log.from_carpet += 1;
        }

        let floor = clean::parse_floor(&r.floor);
        let parking = clean::parse_parking(&r.car_parking);
        let [garden, pool, main_road] = clean::parse_overlooking(&r.overlooking);
        let bedrooms = clean::parse_bedrooms(&r.title);
        let society = clean::normalize_text_key(&r.society);
        let property_type = clean::property_type(&r.title);
        let locality = clean::extract_locality(&r.title, &r.society, &r.location);

        rows.push(Row {
            price,
            price_per_sqft: price / area,
            listing: Listing {
                location: clean::normalize_category(&r.location),
                area_sqft: Some(area),
                bedrooms,
                locality,
                society,
                property_type,
                is_carpet_area,
                bathroom: clean::parse_count(&r.bathroom),
                balcony: clean::parse_count(&r.balcony),
                car_parking: parking.count,
                parking_covered: parking.covered,
                floor_num: floor.number.map(f64::from),
                total_floors: floor.total.map(f64::from),
                furnishing: clean::normalize_category(&r.furnishing),
                transaction: clean::normalize_category(&r.transaction),
                ownership: clean::normalize_category(&r.ownership),
                facing: clean::normalize_facing(&r.facing),
                overlooking_garden: garden > 0.0,
                overlooking_pool: pool > 0.0,
                overlooking_main_road: main_road > 0.0,
            },
        });
    }

    if dedup {
        let mut seen = HashSet::new();
        let before = rows.len();
        rows.retain(|r| seen.insert(dedup_key(r)));
        log.duplicates = before - rows.len();
    }

    // Chop off both tails. A 1,000 rupee/sqft flat in Mumbai and a 900,000
    // rupee/sqft one are both typos, not signal.
    let mut rates: Vec<f64> = rows.iter().map(|r| r.price_per_sqft).collect();
    rates.sort_by(f64::total_cmp);
    log.ppsf_low = percentile(&rates, 1.0);
    log.ppsf_high = percentile(&rates, 99.0);
    let before = rows.len();
    rows.retain(|r| r.price_per_sqft >= log.ppsf_low && r.price_per_sqft <= log.ppsf_high);
    log.outliers = before - rows.len();

    (rows, log)
}

/// Identity of a listing, for de-duplication.
///
/// Two rows are the same listing if every feature the model sees and the price
/// match. `Index` and the free text columns are left out on purpose, since they
/// differ between copies of what is obviously one advert.
fn dedup_key(r: &Row) -> String {
    const SEP: char = '\u{1}';
    let l = &r.listing;
    let n = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.3}"));
    [
        format!("{:.2}", r.price),
        l.location.clone(),
        l.locality.clone(),
        l.society.clone(),
        l.property_type.clone(),
        n(l.area_sqft),
        n(l.bedrooms),
        l.is_carpet_area.to_string(),
        n(l.bathroom),
        n(l.balcony),
        n(l.car_parking),
        l.parking_covered.to_string(),
        n(l.floor_num),
        n(l.total_floors),
        l.furnishing.clone(),
        l.transaction.clone(),
        l.ownership.clone(),
        l.facing.clone(),
        l.overlooking_garden.to_string(),
        l.overlooking_pool.to_string(),
        l.overlooking_main_road.to_string(),
    ]
    .join(&SEP.to_string())
}

/// Shuffles with a fixed seed and splits off a test fraction.
///
/// The shuffle matters: the file is ordered by city, so an unshuffled tail split
/// would test on cities the model never trained on.
pub fn train_test_split(mut rows: Vec<Row>, test_fraction: f64, seed: u64) -> (Vec<Row>, Vec<Row>) {
    let mut rng = SimpleRng::new(seed);
    for i in (1..rows.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        rows.swap(i, j);
    }
    let n_test = (rows.len() as f64 * test_fraction).round() as usize;
    let test = rows.split_off(rows.len() - n_test);
    (rows, test)
}

/// Fits the feature spec on the training split only.
///
/// Fitting on everything would leak test-set statistics into the scaler and the
/// vocabularies, and quietly inflate every metric reported later.
#[must_use]
pub fn fit_spec(
    train: &[Row],
    target: Target,
    max_locations: usize,
    max_localities: usize,
    max_societies: usize,
    hidden: &[usize],
) -> FeatureSpec {
    let mut median = vec![0.0f32; NUMERIC.len()];
    let mut mean = vec![0.0f32; NUMERIC.len()];
    let mut std = vec![1.0f32; NUMERIC.len()];

    for i in 0..NUMERIC.len() {
        let mut seen: Vec<f64> = train
            .iter()
            .map(|r| f64::from(FeatureSpec::raw_numeric(&r.listing)[i]))
            .filter(|v| v.is_finite())
            .collect();
        seen.sort_by(f64::total_cmp);
        median[i] = percentile(&seen, 50.0) as f32;

        // Scale on the imputed column, since that is what the model will see.
        let n = train.len() as f64;
        let imputed = |r: &Row| {
            let v = f64::from(FeatureSpec::raw_numeric(&r.listing)[i]);
            if v.is_finite() {
                v
            } else {
                f64::from(median[i])
            }
        };
        let m = train.iter().map(imputed).sum::<f64>() / n;
        let var = train.iter().map(|r| (imputed(r) - m).powi(2)).sum::<f64>() / n;
        mean[i] = m as f32;
        std[i] = var.sqrt().max(1e-8) as f32;
    }

    let categorical = vec![
        Categorical {
            name: "location".into(),
            vocab: top_vocab(train, "location", max_locations),
        },
        Categorical {
            name: "locality".into(),
            vocab: top_vocab(train, "locality", max_localities),
        },
        Categorical {
            name: "society".into(),
            vocab: top_vocab(train, "society", max_societies),
        },
        Categorical {
            name: "property_type".into(),
            vocab: top_vocab(train, "property_type", usize::MAX),
        },
        Categorical {
            name: "furnishing".into(),
            vocab: top_vocab(train, "furnishing", usize::MAX),
        },
        Categorical {
            name: "transaction".into(),
            vocab: top_vocab(train, "transaction", usize::MAX),
        },
        Categorical {
            name: "ownership".into(),
            vocab: top_vocab(train, "ownership", usize::MAX),
        },
        Categorical {
            name: "facing".into(),
            vocab: top_vocab(train, "facing", usize::MAX),
        },
    ];

    let mut layer_dims = vec![0usize];
    layer_dims.extend_from_slice(hidden);
    layer_dims.push(1);

    let mut spec = FeatureSpec {
        numeric: NUMERIC.iter().map(ToString::to_string).collect(),
        numeric_median: median,
        numeric_mean: mean,
        numeric_std: std,
        categorical,
        target,
        target_mean: 0.0,
        target_std: 1.0,
        layer_dims,
        activation: Activation::Relu,
        dropout: 0.0,
        layer_norm: false,
        target_encodings: Vec::new(),
    };
    spec.layer_dims[0] = spec.input_dim();
    spec
}

/// Fits target encodings for `columns` on the rows passed in.
///
/// One-hot on a 500 value column costs 500 input columns to say something that
/// is mostly "this area is expensive". The average `ln(price per sqft)` says it
/// in two numbers and gives rare values much less room to overfit.
///
/// Price per sqft rather than price because it works across sizes. It's what
/// lets "Bandra is expensive" combine with a floor area the model never saw in
/// Bandra.
///
/// Each value gets pulled toward the overall average by `smoothing` fake
/// observations, so a locality that appears three times can't claim a confident
/// price:
///
/// ```text
/// encoded = (sum_for_level + prior * k) / (rows_in_level + k)
/// ```
///
/// Only uses the rows passed in. The caller hands over the training split with
/// the validation rows already taken out, so a locality's own price can't
/// inflate the validation or test numbers.
#[must_use]
pub fn fit_target_encodings(rows: &[Row], columns: &[&str], smoothing: f64) -> Vec<TargetEncoding> {
    let prior = rows.iter().map(|r| r.price_per_sqft.ln()).sum::<f64>() / rows.len() as f64;

    columns
        .iter()
        .map(|name| {
            let mut totals: HashMap<&str, (f64, usize)> = HashMap::new();
            for r in rows {
                let level = FeatureSpec::categorical_value(name, &r.listing);
                let entry = totals.entry(level).or_insert((0.0, 0));
                entry.0 += r.price_per_sqft.ln();
                entry.1 += 1;
            }

            let levels: HashMap<String, [f32; 2]> = totals
                .iter()
                .map(|(level, (sum, n))| {
                    let smoothed = (sum + prior * smoothing) / (*n as f64 + smoothing);
                    (
                        (*level).to_string(),
                        [smoothed as f32, (*n as f64).ln_1p() as f32],
                    )
                })
                .collect();

            // A value we've never seen gets the overall average and a count of
            // zero, which is exactly the "no idea about this one" signal.
            let fallback = [prior as f32, 0.0f32];

            // Scale on the values the training rows actually produce.
            let observed: Vec<[f32; 2]> = rows
                .iter()
                .map(|r| {
                    let level = FeatureSpec::categorical_value(name, &r.listing);
                    levels.get(level).copied().unwrap_or(fallback)
                })
                .collect();
            let n = observed.len() as f64;
            let mut mean = [0.0f32; 2];
            let mut std = [1.0f32; 2];
            for i in 0..2 {
                let m = observed.iter().map(|v| f64::from(v[i])).sum::<f64>() / n;
                let var = observed
                    .iter()
                    .map(|v| (f64::from(v[i]) - m).powi(2))
                    .sum::<f64>()
                    / n;
                mean[i] = m as f32;
                std[i] = var.sqrt().max(1e-8) as f32;
            }

            TargetEncoding {
                name: (*name).to_string(),
                levels,
                fallback,
                mean,
                std,
            }
        })
        .collect()
}

/// The `max_keep` most frequent values of a categorical feature, plus [`OTHER`].
fn top_vocab(train: &[Row], name: &str, max_keep: usize) -> Vec<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for r in train {
        *counts
            .entry(FeatureSpec::categorical_value(name, &r.listing))
            .or_insert(0) += 1;
    }
    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    // Frequency first, then name, so the vocabulary is deterministic across runs.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let mut vocab: Vec<String> = ranked
        .into_iter()
        .take(max_keep)
        .map(|(v, _)| v.to_string())
        .filter(|v| v != OTHER)
        .collect();
    vocab.push(OTHER.into());
    vocab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_pick_the_expected_ends() {
        let v: Vec<f64> = (0..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.0), 0.0);
        assert_eq!(percentile(&v, 50.0), 50.0);
        assert_eq!(percentile(&v, 99.0), 99.0);
    }

    #[test]
    fn split_is_disjoint_and_correctly_sized() {
        let rows: Vec<Row> = (0..100)
            .map(|i| Row {
                listing: Listing {
                    location: i.to_string(),
                    ..Listing::default()
                },
                price: f64::from(i),
                price_per_sqft: 1.0,
            })
            .collect();
        let (train, test) = train_test_split(rows, 0.2, 42);
        assert_eq!((train.len(), test.len()), (80, 20));

        let mut seen: Vec<u64> = train.iter().chain(&test).map(|r| r.price as u64).collect();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..100).collect::<Vec<_>>(),
            "split loses or duplicates no row"
        );
    }
}
