//! Charts, drawn with `plotters`.
//!
//! Each function writes one SVG file and the report links to it by name. Using
//! a library instead of writing SVG by hand means axis scaling, tick placement
//! and log axes are somebody else's problem.

use std::path::Path;

use anyhow::{Context, Result};
use plotters::coord::Shift;
use plotters::prelude::*;

/// Liberation Sans, embedded.
///
/// The `ab_glyph` text backend has no font of its own and no access to the
/// system's, so without this every chart fails with `FontUnavailable`. Bundling
/// one means the charts render identically on a developer laptop and in a slim
/// container that ships no fonts at all. Licence in `ml/assets/`.
const FONT: &[u8] = include_bytes!("../assets/LiberationSans-Regular.ttf");

/// Registers the bundled font under the family names the charts ask for.
///
/// `register_font` is idempotent in effect - a second call simply replaces the
/// entry - so this is safe to call before every chart.
fn use_bundled_font() -> Result<()> {
    for family in [FontFamily::SansSerif, FontFamily::Name("sans-serif")] {
        plotters::style::register_font(family.as_str(), FontStyle::Normal, FONT)
            .map_err(|_| anyhow::anyhow!("the bundled font is not valid TTF"))?;
    }
    Ok(())
}

const W: u32 = 900;
const H: u32 = 520;
const INK: RGBColor = RGBColor(31, 41, 51);
const ACCENT: RGBColor = RGBColor(47, 111, 159);
const ACCENT_SOFT: RGBColor = RGBColor(143, 188, 217);
const MARK: RGBColor = RGBColor(193, 68, 60);

/// Opens an SVG drawing area with a white background.
///
/// Explicitly white rather than transparent so the charts stay readable when
/// GitHub renders the report in dark mode.
fn canvas(path: &Path) -> Result<DrawingArea<SVGBackend<'_>, Shift>> {
    use_bundled_font()?;
    let area = SVGBackend::new(path, (W, H)).into_drawing_area();
    area.fill(&WHITE).context("fill chart background")?;
    Ok(area)
}

/// Formats rupees for an axis label.
fn rupees(v: &f64) -> String {
    let v = *v;
    if v.abs() >= 1e7 {
        format!("{:.1}Cr", v / 1e7)
    } else if v.abs() >= 1e5 {
        format!("{:.0}L", v / 1e5)
    } else if v.abs() >= 1e3 {
        format!("{:.0}K", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

/// Ticks for a loss axis. Values sit between about 0.02 and 1, so they need
/// real decimals. Without this the log axis falls back to plotters' default
/// float printing and a tick like 0.30000000000000004 gets drawn in full and
/// clipped, which is where the "00000000000001" labels came from.
fn small(v: &f64) -> String {
    let v = *v;
    if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

fn plain(v: &f64) -> String {
    if v.abs() >= 1000.0 || v.fract().abs() < 1e-9 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// Bounds of a slice, widened slightly so points do not sit on the axis.
fn span(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for v in values.filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi { (0.0, 1.0) } else { (lo, hi) }
}

/// A histogram of `values` on a log-spaced x axis.
///
/// # Errors
/// Returns an error if the chart cannot be written.
pub fn histogram_log(
    path: &Path,
    values: &[f64],
    bins: usize,
    title: &str,
    x_label: &str,
    money: bool,
) -> Result<()> {
    let positive: Vec<f64> = values.iter().copied().filter(|v| *v > 0.0).collect();
    let (lo, hi) = span(positive.iter().copied());
    let (log_lo, log_hi) = (lo.log10(), hi.log10());
    let width = (log_hi - log_lo) / bins as f64;

    let mut counts = vec![0u32; bins];
    for v in &positive {
        let idx = (((v.log10() - log_lo) / width) as usize).min(bins - 1);
        counts[idx] += 1;
    }
    let peak = f64::from(*counts.iter().max().unwrap_or(&1)) * 1.05;

    let area = canvas(path)?;
    let mut chart = ChartBuilder::on(&area)
        .caption(title, ("sans-serif", 22).into_font().color(&INK))
        .margin(16)
        .x_label_area_size(52)
        .y_label_area_size(72)
        .build_cartesian_2d((lo..hi).log_scale(), 0.0..peak)
        .context("build histogram axes")?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("listings")
        .x_label_formatter(&(if money { rupees } else { plain }))
        .y_label_formatter(&plain)
        .label_style(("sans-serif", 13).into_font().color(&INK))
        .draw()
        .context("draw histogram mesh")?;

    chart
        .draw_series(counts.iter().enumerate().map(|(i, count)| {
            let left = 10f64.powf(log_lo + width * i as f64);
            let right = 10f64.powf(log_lo + width * (i + 1) as f64);
            Rectangle::new(
                [(left, 0.0), (right, f64::from(*count))],
                ACCENT.mix(0.85).filled(),
            )
        }))
        .context("draw histogram bars")?;

    area.present().context("write histogram")?;
    Ok(())
}

/// A scatter of `(x, y)` pairs on log-log axes, optionally with a y = x line.
///
/// Points are sub-sampled: drawing all 174k listings makes a multi-megabyte SVG
/// and no clearer a picture. Axis limits still come from the full data, so the
/// sample cannot silently crop the range the chart claims to show.
///
/// # Errors
/// Returns an error if the chart cannot be written.
pub fn scatter_log(
    path: &Path,
    points: &[(f64, f64)],
    title: &str,
    x_label: &str,
    y_label: &str,
    money_x: bool,
    diagonal: bool,
) -> Result<()> {
    const MAX_POINTS: usize = 12_000;
    let usable: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| *x > 0.0 && *y > 0.0)
        .collect();
    let (x_lo, x_hi) = span(usable.iter().map(|(x, _)| *x));
    let (y_lo, y_hi) = span(usable.iter().map(|(_, y)| *y));
    let stride = usable.len().div_ceil(MAX_POINTS).max(1);

    let area = canvas(path)?;
    let mut chart = ChartBuilder::on(&area)
        .caption(title, ("sans-serif", 22).into_font().color(&INK))
        .margin(16)
        .x_label_area_size(52)
        .y_label_area_size(78)
        .build_cartesian_2d((x_lo..x_hi).log_scale(), (y_lo..y_hi).log_scale())
        .context("build scatter axes")?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_formatter(&(if money_x { rupees } else { plain }))
        .y_label_formatter(&rupees)
        .label_style(("sans-serif", 13).into_font().color(&INK))
        .draw()
        .context("draw scatter mesh")?;

    chart
        .draw_series(
            usable
                .iter()
                .step_by(stride)
                .map(|(x, y)| Circle::new((*x, *y), 1.6, ACCENT.mix(0.18).filled())),
        )
        .context("draw scatter points")?;

    if diagonal {
        let lo = x_lo.max(y_lo);
        let hi = x_hi.min(y_hi);
        chart
            .draw_series(LineSeries::new([(lo, lo), (hi, hi)], MARK.stroke_width(2)))
            .context("draw diagonal")?;
    }

    area.present().context("write scatter")?;
    Ok(())
}

/// A vertical bar chart over labelled categories.
///
/// # Errors
/// Returns an error if the chart cannot be written.
pub fn bars(
    path: &Path,
    labels: &[String],
    values: &[f64],
    title: &str,
    y_label: &str,
) -> Result<()> {
    let peak = values.iter().copied().fold(0.0f64, f64::max) * 1.08;

    let area = canvas(path)?;
    let mut chart = ChartBuilder::on(&area)
        .caption(title, ("sans-serif", 22).into_font().color(&INK))
        .margin(16)
        .x_label_area_size(96)
        .y_label_area_size(78)
        .build_cartesian_2d((0..labels.len()).into_segmented(), 0.0..peak)
        .context("build bar axes")?;
    chart
        .configure_mesh()
        .y_desc(y_label)
        .x_label_formatter(&|value| {
            let SegmentValue::CenterOf(i) = value else {
                return String::new();
            };
            labels.get(*i).cloned().unwrap_or_default()
        })
        .y_label_formatter(&rupees)
        .label_style(("sans-serif", 12).into_font().color(&INK))
        .x_label_style(
            ("sans-serif", 12)
                .into_font()
                .color(&INK)
                .transform(FontTransform::Rotate270),
        )
        .draw()
        .context("draw bar mesh")?;

    chart
        .draw_series(values.iter().enumerate().map(|(i, v)| {
            let mut bar = Rectangle::new(
                [
                    (SegmentValue::Exact(i), 0.0),
                    (SegmentValue::Exact(i + 1), *v),
                ],
                ACCENT.mix(0.85).filled(),
            );
            bar.set_margin(0, 0, 6, 6);
            bar
        }))
        .context("draw bars")?;

    area.present().context("write bars")?;
    Ok(())
}

/// Five-number summary of a group, for [`boxplot`].
pub struct Summary {
    /// Category label.
    pub label: String,
    /// The quartiles, as plotters wants them.
    pub quartiles: Quartiles,
}

/// A box plot, one box per group, on a log-price y axis.
///
/// # Errors
/// Returns an error if the chart cannot be written.
pub fn boxplot(path: &Path, groups: &[Summary], title: &str, y_label: &str) -> Result<()> {
    // The quartiles already contain log10(price). Plotters' Boxplot calculates
    // Tukey fences before mapping an element onto the chart. Feeding it raw
    // prices with a logarithmic coordinate can produce a negative lower fence,
    // which has no log coordinate and collapses the entire box to the baseline.
    // Transforming first keeps every element in a valid, linear coordinate
    // space while the labels still display rupees.
    let (lo, hi) = span(
        groups
            .iter()
            .flat_map(|g| g.quartiles.values().into_iter().map(f64::from)),
    );
    let padding = ((hi - lo) * 0.08).max(0.1);
    let (lo, hi) = ((lo - padding) as f32, (hi + padding) as f32);
    let labels: Vec<&str> = groups.iter().map(|g| g.label.as_str()).collect();

    let area = canvas(path)?;
    let mut chart = ChartBuilder::on(&area)
        .caption(title, ("sans-serif", 22).into_font().color(&INK))
        .margin(16)
        .x_label_area_size(80)
        .y_label_area_size(78)
        .build_cartesian_2d(labels.as_slice().into_segmented(), lo..hi)
        .context("build boxplot axes")?;
    chart
        .configure_mesh()
        .y_desc(y_label)
        .y_label_formatter(&|v: &f32| rupees(&10f64.powf(f64::from(*v))))
        .label_style(("sans-serif", 12).into_font().color(&INK))
        .draw()
        .context("draw boxplot mesh")?;

    // `labels` outlives the series, so the segment coordinates borrow from it
    // rather than from a temporary created inside the closure.
    chart
        .draw_series(groups.iter().zip(&labels).map(|(g, label)| {
            Boxplot::new_vertical(SegmentValue::CenterOf(label), &g.quartiles)
                .width(30)
                .style(ACCENT_SOFT.filled())
        }))
        .context("draw boxes")?;

    area.present().context("write boxplot")?;
    Ok(())
}

/// Labelled series against a shared x axis, for training curves.
///
/// # Errors
/// Returns an error if the chart cannot be written.
pub fn lines(
    path: &Path,
    series: &[(&str, Vec<f64>)],
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    let n = series
        .iter()
        .map(|(_, v)| v.len())
        .max()
        .unwrap_or(1)
        .max(2);
    let (lo, hi) = span(
        series
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .filter(|v| *v > 0.0),
    );

    let area = canvas(path)?;
    let mut chart = ChartBuilder::on(&area)
        .caption(title, ("sans-serif", 22).into_font().color(&INK))
        .margin(16)
        .x_label_area_size(52)
        .y_label_area_size(78)
        .build_cartesian_2d(0f64..(n - 1) as f64, (lo * 0.9..hi * 1.1).log_scale())
        .context("build line axes")?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_formatter(&plain)
        .y_label_formatter(&small)
        .label_style(("sans-serif", 13).into_font().color(&INK))
        .draw()
        .context("draw line mesh")?;

    for (i, (name, values)) in series.iter().enumerate() {
        let color = if i == 0 { ACCENT } else { MARK };
        chart
            .draw_series(LineSeries::new(
                values.iter().enumerate().map(|(x, y)| (x as f64, *y)),
                color.stroke_width(2),
            ))
            .context("draw line")?
            .label(*name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 18, y)], color.stroke_width(3)));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(INK.mix(0.3))
        .label_font(("sans-serif", 13).into_font().color(&INK))
        .draw()
        .context("draw legend")?;

    area.present().context("write lines")?;
    Ok(())
}

/// Builds Plotters quartiles in log10(price) space.
#[must_use]
pub fn quartiles(values: &[f64]) -> Quartiles {
    let as_f32: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .map(|v| v.log10() as f32)
        .collect();
    Quartiles::new(&as_f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxplot_quartiles_live_in_log_price_space() {
        let q = quartiles(&[1e6, 2e6, 4e6, 8e6, 16e6]);
        assert!(
            q.values().iter().all(|value| (5.0..8.0).contains(value)),
            "raw rupee quartiles collapse when drawn on the log-price chart"
        );
    }
}
