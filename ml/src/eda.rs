//! The exploratory charts, and the numbers the written commentary quotes.
//!
//! Each function returns the facts behind its chart, so the report can state
//! what a chart shows without anyone reading it off the picture by eye.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::dataset::{Row, percentile};
use crate::plot::{self, Summary};

/// A group summarised for a bar chart or a box plot.
pub struct Group {
    /// Category label.
    pub label: String,
    /// Rows in the group.
    pub count: usize,
    /// Median price, in rupees.
    pub median: f64,
}

/// Numbers the report quotes back in prose and tables.
pub struct Facts {
    /// Price percentiles: 1, 25, 50, 75, 99.
    pub price_quantiles: [f64; 5],
    /// Mean price over median price - a one-number read on the skew.
    pub price_skew_ratio: f64,
    /// Median price per square foot.
    pub median_ppsf: f64,
    /// The 15 cities with the most listings.
    pub top_locations: Vec<Group>,
    /// Median price by furnishing status.
    pub by_furnishing: Vec<Group>,
    /// Median price by bathroom count.
    pub by_bathroom: Vec<Group>,
    /// Correlation between log area and log price.
    pub log_area_price_corr: f64,
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// Groups rows by a key, keeping the `top` largest groups.
fn group_by<'a>(
    rows: &'a [Row],
    key: impl Fn(&'a Row) -> String,
    top: usize,
) -> Vec<(String, Vec<f64>)> {
    let mut buckets: HashMap<String, Vec<f64>> = HashMap::new();
    for r in rows {
        buckets.entry(key(r)).or_default().push(r.price);
    }
    let mut groups: Vec<(String, Vec<f64>)> = buckets.into_iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    groups.truncate(top);
    groups
}

fn to_groups(pairs: &[(String, Vec<f64>)]) -> Vec<Group> {
    pairs
        .iter()
        .map(|(label, prices)| Group {
            label: label.clone(),
            count: prices.len(),
            median: percentile(&sorted(prices), 50.0),
        })
        .collect()
}

/// Pearson correlation, computed in parallel over the row set.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.par_iter().sum::<f64>() / n;
    let my = ys.par_iter().sum::<f64>() / n;
    let (cov, vx, vy) = xs
        .par_iter()
        .zip(ys.par_iter())
        .map(|(x, y)| ((x - mx) * (y - my), (x - mx).powi(2), (y - my).powi(2)))
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    cov / (vx.sqrt() * vy.sqrt())
}

/// Writes every EDA chart into `dir` and returns the facts behind them.
///
/// # Errors
/// Returns an error if a chart cannot be written.
pub fn write_all(rows: &[Row], dir: &Path) -> Result<Facts> {
    std::fs::create_dir_all(dir)?;

    // 1 - the target's distribution. On a linear axis this is an unreadable
    // spike against the y axis; on a log axis it is close to symmetric, which is
    // the whole argument for training on log price.
    let prices: Vec<f64> = rows.iter().map(|r| r.price).collect();
    plot::histogram_log(
        &dir.join("01_price_distribution.svg"),
        &prices,
        60,
        "Sale price (log scale)",
        "price",
        true,
    )?;

    // 2 - price against area, both logged. A straight line here means a power
    // law, and that log-log is the space the model should work in.
    let points: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| r.listing.area_sqft.map(|a| (a, r.price)))
        .collect();
    plot::scatter_log(
        &dir.join("02_price_vs_area.svg"),
        &points,
        "Price vs floor area",
        "area (sqft, log)",
        "price (log)",
        false,
        false,
    )?;

    // 3 - the 15 busiest cities. `location` turned out to be a city, not a
    // neighbourhood, which is why 81 categories cover the whole file.
    let by_location = group_by(rows, |r| r.listing.location.clone(), 15);
    let top_locations = to_groups(&by_location);
    plot::bars(
        &dir.join("03_price_by_location.svg"),
        &top_locations
            .iter()
            .map(|g| g.label.clone())
            .collect::<Vec<_>>(),
        &top_locations.iter().map(|g| g.median).collect::<Vec<_>>(),
        "Median price by city (15 busiest)",
        "median price",
    )?;

    // 4 - furnishing. Three ordered categories, so a box plot shows both the
    // shift and how much the ranges overlap.
    let furnishing = group_by(rows, |r| r.listing.furnishing.clone(), 8);
    plot::boxplot(
        &dir.join("04_price_by_furnishing.svg"),
        &furnishing
            .iter()
            .map(|(l, p)| Summary {
                label: l.clone(),
                quartiles: plot::quartiles(p),
            })
            .collect::<Vec<_>>(),
        "Price by furnishing status",
        "price (log)",
    )?;

    // 5 - bathrooms, the best cheap proxy for size and segment.
    let mut bathroom = group_by(
        rows,
        |r| {
            r.listing
                .bathroom
                .map_or_else(|| "unknown".into(), |b| format!("{b:.0}"))
        },
        7,
    );
    bathroom.sort_by(|a, b| a.0.cmp(&b.0));
    plot::boxplot(
        &dir.join("05_price_by_bathroom.svg"),
        &bathroom
            .iter()
            .map(|(l, p)| Summary {
                label: format!("{l} bath"),
                quartiles: plot::quartiles(p),
            })
            .collect::<Vec<_>>(),
        "Price by bathroom count",
        "price (log)",
    )?;

    // 6 - the rate the outlier trim is based on.
    let ppsf: Vec<f64> = rows.iter().map(|r| r.price_per_sqft).collect();
    plot::histogram_log(
        &dir.join("06_price_per_sqft.svg"),
        &ppsf,
        60,
        "Price per square foot (after trimming)",
        "rupees / sqft",
        false,
    )?;

    let prices_sorted = sorted(&prices);
    let price_quantiles = [
        percentile(&prices_sorted, 1.0),
        percentile(&prices_sorted, 25.0),
        percentile(&prices_sorted, 50.0),
        percentile(&prices_sorted, 75.0),
        percentile(&prices_sorted, 99.0),
    ];
    let mean = prices.par_iter().sum::<f64>() / prices.len() as f64;
    let (log_areas, log_prices): (Vec<f64>, Vec<f64>) =
        points.iter().map(|(a, p)| (a.ln(), p.ln())).unzip();

    Ok(Facts {
        price_quantiles,
        price_skew_ratio: mean / price_quantiles[2],
        median_ppsf: percentile(&sorted(&ppsf), 50.0),
        top_locations,
        by_furnishing: to_groups(&furnishing),
        by_bathroom: to_groups(&bathroom),
        log_area_price_corr: pearson(&log_areas, &log_prices),
    })
}
