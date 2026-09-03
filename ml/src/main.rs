//! House price pipeline: load, inspect, clean, explore, train on Vearo, export.
//!
//! This is the notebook, written as a program. One command runs the whole thing
//! top to bottom, so there's no cell ordering to get wrong and no stale kernel
//! state hanging around.
//!
//!     cargo run --release -p ml -- --epochs 40 --cv
//!
//! Outputs: `models/` (weights plus the preprocessing spec the API loads) and
//! `reports/` (the charts and REPORT.md).

mod dataset;
mod eda;
mod plot;
mod report;
mod train;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hp_core::device::{self, Preference};
use hp_core::features::{Activation, FeatureSpec, Target};
use indicatif::{ProgressBar, ProgressStyle};
use train::{Config, Encoded};

/// Trains the house price model and writes the report.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The Kaggle CSV to train on.
    #[arg(long, default_value = "ml/data/house_prices.csv")]
    data: PathBuf,

    /// Where to write the weights and the feature spec.
    #[arg(long, default_value = "models")]
    models: PathBuf,

    /// Where to write REPORT.md and the charts.
    #[arg(long, default_value = "reports")]
    reports: PathBuf,

    /// Passes over the training split.
    #[arg(long, default_value_t = 40)]
    epochs: usize,

    /// Cities kept as their own one-hot column; the rest become `other`.
    #[arg(long, default_value_t = 100)]
    max_locations: usize,

    /// Localities extracted from titles and kept as one-hot columns.
    #[arg(long, default_value_t = 500)]
    max_localities: usize,

    /// Named societies kept as one-hot columns.
    #[arg(long, default_value_t = 500)]
    max_societies: usize,

    /// Hidden layer widths for the deep models.
    #[arg(long, value_delimiter = ',', default_value = "128,64")]
    hidden: Vec<usize>,

    /// Mini-batch size.
    #[arg(long, default_value_t = 512)]
    batch: usize,

    /// Peak AdamW learning rate for the deep models.
    #[arg(long, default_value_t = 3e-3)]
    learning_rate: f32,

    /// AdamW decoupled weight decay for the deep models.
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,

    /// Hidden activation: relu or gelu.
    #[arg(long, default_value = "relu", value_parser = parse_activation)]
    activation: Activation,

    /// Hidden activation dropout probability.
    #[arg(long, default_value_t = 0.0)]
    dropout: f32,

    /// Apply trainable layer normalisation before each hidden activation.
    #[arg(long)]
    layer_norm: bool,

    /// Stop after this many epochs without validation-loss improvement.
    #[arg(long, default_value_t = 10)]
    patience: usize,

    /// What the network predicts: `log1p` for log price, `rate` for log price
    /// per square foot (multiplied back up by area at inference).
    #[arg(long, default_value = "log1p", value_parser = parse_target)]
    target: Target,

    /// Train this many models with different seeds and average their
    /// predictions. Averaging is done in model space, before the log is undone.
    #[arg(long, default_value_t = 1)]
    ensemble: usize,

    /// Columns to encode by their mean log price per sqft, comma separated.
    /// Empty turns target encoding off.
    #[arg(long, value_delimiter = ',', default_value = "")]
    target_encode: Vec<String>,

    /// Pseudo-observations pulling each level's mean toward the global one.
    #[arg(long, default_value_t = 20.0)]
    te_smoothing: f64,

    /// Columns to drop from the one-hot block, comma separated. Useful once a
    /// column is target encoded and its 500 one-hot columns are redundant.
    #[arg(long, value_delimiter = ',', default_value = "")]
    drop_onehot: Vec<String>,

    /// Run the bonus 5-fold cross-validation on the winning model.
    #[arg(long)]
    cv: bool,

    /// Train one candidate on a development split, skipping reports and the final holdout.
    #[arg(long)]
    benchmark: bool,

    /// Where to run the tensors.
    #[arg(long, default_value = "auto", value_parser = parse_device)]
    device: Preference,
}

fn parse_device(s: &str) -> Result<Preference, String> {
    s.parse()
}

fn parse_target(s: &str) -> Result<Target, String> {
    match s.to_lowercase().as_str() {
        "log1p" | "log" => Ok(Target::Log1p),
        "rate" | "per_sqft" | "ppsf" => Ok(Target::LogPricePerSqft),
        "raw" => Ok(Target::Raw),
        other => Err(format!("unknown target {other:?}, expected log1p, rate or raw")),
    }
}

fn parse_activation(s: &str) -> Result<Activation, String> {
    match s.trim().to_lowercase().as_str() {
        "relu" => Ok(Activation::Relu),
        "gelu" => Ok(Activation::Gelu),
        other => Err(format!(
            "unknown activation {other:?}, expected relu or gelu"
        )),
    }
}

/// A spinner for the slow, silent stages.
fn spinner(message: &'static str) -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("{spinner} {msg} {elapsed}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    bar.enable_steady_tick(std::time::Duration::from_millis(120));
    bar.set_message(message);
    bar
}

/// Applies the target-encoding and one-hot-dropping options to a fitted spec.
///
/// Both change how wide the encoded vector is, so `layer_dims[0]` gets
/// recomputed after. Forget that and the checkpoint won't match the spec.
fn apply_encoding_options(spec: &mut FeatureSpec, training: &[dataset::Row], args: &Args) {
    let columns: Vec<&str> = args
        .target_encode
        .iter()
        .map(String::as_str)
        .filter(|c| !c.is_empty())
        .collect();
    if !columns.is_empty() {
        spec.target_encodings =
            dataset::fit_target_encodings(training, &columns, args.te_smoothing);
    }

    let dropped: Vec<&str> = args
        .drop_onehot
        .iter()
        .map(String::as_str)
        .filter(|c| !c.is_empty())
        .collect();
    if !dropped.is_empty() {
        spec.categorical
            .retain(|c| !dropped.contains(&c.name.as_str()));
    }

    spec.layer_dims[0] = spec.input_dim();
}

/// Floor area of a row. `build` only keeps rows that have one so the fallback
/// never fires, but it's 1.0 rather than 0.0 so the rate target can't divide by
/// zero if that ever changes.
fn row_area(row: &dataset::Row) -> f64 {
    row.listing.area_sqft.unwrap_or(1.0)
}

/// Encodes rows against a spec, into features, model-space targets and rupees.
fn encode(spec: &FeatureSpec, rows: &[dataset::Row]) -> Encoded {
    Encoded {
        x: rows.iter().map(|r| spec.encode(&r.listing)).collect(),
        y: rows.iter().map(|r| spec.encode_target(r.price, row_area(r))).collect(),
        price: rows.iter().map(|r| r.price).collect(),
        area_sqft: rows.iter().map(row_area).collect(),
    }
}

/// Encodes the fitting rows without letting a row contribute to its own
/// target-derived features.
///
/// The final mappings in `spec` are fitted on every fitting row because those
/// are the mappings validation, test and production will see. For the fitting
/// matrix itself, each fifth of the rows is encoded by mappings fitted on the
/// other four fifths. Keeping the final mapping's scale makes all five folds
/// live in the same feature space.
fn encode_training_oof(
    spec: &FeatureSpec,
    rows: &[dataset::Row],
    columns: &[&str],
    smoothing: f64,
) -> Encoded {
    const FOLDS: usize = 5;
    if columns.is_empty() {
        return encode(spec, rows);
    }

    let mut x = vec![Vec::new(); rows.len()];
    for fold in 0..FOLDS {
        let fitting: Vec<dataset::Row> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % FOLDS != fold)
            .map(|(_, row)| row.clone())
            .collect();
        let mut encodings = dataset::fit_target_encodings(&fitting, columns, smoothing);
        for (fold_encoding, final_encoding) in encodings.iter_mut().zip(&spec.target_encodings) {
            fold_encoding.mean = final_encoding.mean;
            fold_encoding.std = final_encoding.std;
        }
        let mut fold_spec = spec.clone();
        fold_spec.target_encodings = encodings;
        for (i, row) in rows.iter().enumerate().filter(|(i, _)| i % FOLDS == fold) {
            x[i] = fold_spec.encode(&row.listing);
        }
    }

    Encoded {
        x,
        y: rows.iter().map(|r| spec.encode_target(r.price, row_area(r))).collect(),
        price: rows.iter().map(|r| r.price).collect(),
        area_sqft: rows.iter().map(row_area).collect(),
    }
}

/// Fits the target scaling on the training split.
fn fit_target_scaling(spec: &mut FeatureSpec, train: &[dataset::Row]) {
    let t: Vec<f64> = train
        .iter()
        .map(|r| spec.target.forward(r.price, row_area(r)))
        .collect();
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let var = t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    spec.target_mean = mean as f32;
    spec.target_std = var.sqrt().max(1e-8) as f32;
}

/// Measures one candidate without touching the frozen 20% final holdout.
fn benchmark(args: &Args, dev: vearo::Device, rows: Vec<dataset::Row>) {
    let (development, _final_holdout) = dataset::train_test_split(rows, 0.2, 20_260_902);
    let (fit, tuning) = dataset::train_test_split(development, 0.125, 0x7a11_ce01);
    let (training, early_stop) = dataset::train_test_split(fit, 1.0 / 7.0, 0xea41_5709);

    let mut spec = dataset::fit_spec(
        &training,
        args.target,
        args.max_locations,
        args.max_localities,
        args.max_societies,
        &args.hidden,
    );
    fit_target_scaling(&mut spec, &training);
    apply_encoding_options(&mut spec, &training, args);

    let target_columns: Vec<&str> = args
        .target_encode
        .iter()
        .map(String::as_str)
        .filter(|c| !c.is_empty())
        .collect();
    let training = encode_training_oof(&spec, &training, &target_columns, args.te_smoothing);
    let early_stop = encode(&spec, &early_stop);
    let tuning = encode(&spec, &tuning);
    let cfg = Config {
        name: "optimization candidate",
        hidden: args.hidden.clone(),
        epochs: args.epochs,
        batch: args.batch,
        lr: args.learning_rate,
        weight_decay: args.weight_decay,
        activation: args.activation,
        dropout: args.dropout,
        layer_norm: args.layer_norm,
        patience: args.patience,
        seed: 23,
        device: dev,
    };
    // Averaging a few seeds cancels out the part of the error that comes from
    // the random init and batch order rather than from the data. Averaging
    // happens before undoing the log, so it's a geometric mean of the prices,
    // which is the right kind of average for something this skewed.
    let members = args.ensemble.max(1);
    let mut summed = vec![0.0f64; tuning.len()];
    let mut run = None;
    for member in 0..members {
        let cfg = Config { seed: 23 + member as u64, ..cfg.clone() };
        let (model, member_run) = train::train(&spec, &training, &early_stop, &tuning, &cfg);
        for (acc, raw) in summed
            .iter_mut()
            .zip(hp_core::predict(&model, &tuning.x, spec.input_dim(), 4096, dev))
        {
            *acc += f64::from(raw);
        }
        run = Some(member_run);
    }
    let mut run = run.expect("at least one ensemble member");
    if members > 1 {
        let predicted: Vec<f64> = summed
            .iter()
            .zip(&tuning.area_sqft)
            .map(|(total, area)| spec.decode_target(total / members as f64, *area))
            .collect();
        run.test = train::evaluate(&tuning.price, &predicted);
    }

    println!(
        "\nBENCHMARK features={} shape={:?} target={:?} activation={:?} dropout={} norm={} ensemble={} best_epoch={} MAE={:.0} RMSE={:.0} R2={:.4} MdAPE={:.3}% time={:.1}s",
        spec.input_dim(),
        run.layer_dims,
        args.target,
        run.activation,
        run.dropout,
        run.layer_norm,
        members,
        run.best_epoch,
        run.test.mae,
        run.test.rmse,
        run.test.r2,
        run.test.mdape,
        run.seconds,
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = device::select(args.device);
    println!("device: {}\n", device::name(dev));

    // ---- 1. load and inspect ------------------------------------------------
    println!("== 1. load & inspect ==");
    let bar = spinner("reading csv");
    let raw = dataset::read(&args.data)?;
    bar.finish_and_clear();
    println!("{} rows", raw.len());

    let profile = profile_columns(&raw);

    // ---- 2. clean -----------------------------------------------------------
    println!("\n== 2. clean & engineer features ==");
    let (rows, log) = dataset::build(&raw, true);
    println!(
        "kept {} of {} rows (dropped {} without a price, {} without an area, {} duplicate listings, {} rate outliers)",
        rows.len(),
        log.total,
        log.no_price,
        log.no_area,
        log.duplicates,
        log.outliers
    );
    if args.benchmark {
        benchmark(&args, dev, rows);
        return Ok(());
    }
    // Measured here so the report can justify dropping `Society` rather than
    // asserting it: this is the share the 50 largest societies actually cover.
    let society = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for r in &raw {
            let s = r.society.trim();
            if !s.is_empty() {
                *counts.entry(s).or_insert(0) += 1;
            }
        }
        let mut n: Vec<usize> = counts.values().copied().collect();
        n.sort_unstable_by(|a, b| b.cmp(a));
        let top: usize = n.iter().take(50).sum();
        report::Society {
            distinct: n.len(),
            top50_coverage_pct: 100.0 * top as f64 / raw.len() as f64,
        }
    };

    // ---- 3. explore ---------------------------------------------------------
    println!("\n== 3. exploratory plots ==");
    let bar = spinner("drawing charts");
    let facts = eda::write_all(&rows, &args.reports).context("write EDA charts")?;
    bar.finish_and_clear();
    println!("wrote 6 charts to {}", args.reports.display());

    // ---- 4. split, fit the spec, encode -------------------------------------
    println!("\n== 4. split & encode ==");
    let (train_rows, test_rows) = dataset::train_test_split(rows, 0.2, 20_260_902);
    let mut spec = dataset::fit_spec(
        &train_rows,
        Target::Log1p,
        args.max_locations,
        args.max_localities,
        args.max_societies,
        &args.hidden,
    );
    fit_target_scaling(&mut spec, &train_rows);
    // The validation rows come out before the target encodings are fitted, so a
    // level's own price cannot inflate the loss used to choose the best epoch.
    let (fit_rows, val_rows) = dataset::train_test_split(train_rows, 0.1, 0x5a1e_0900);
    apply_encoding_options(&mut spec, &fit_rows, &args);
    println!(
        "train {} / val {} / test {}, {} features ({} numeric + {} encoded + {} one-hot)",
        fit_rows.len(),
        val_rows.len(),
        test_rows.len(),
        spec.input_dim(),
        spec.numeric.len(),
        2 * spec.target_encodings.len(),
        spec.input_dim() - spec.numeric.len() - 2 * spec.target_encodings.len()
    );

    let mut spec_raw = spec.clone();
    spec_raw.target = Target::Raw;
    fit_target_scaling(&mut spec_raw, &fit_rows);

    let target_columns: Vec<&str> = args
        .target_encode
        .iter()
        .map(String::as_str)
        .filter(|c| !c.is_empty())
        .collect();
    let log_train = encode_training_oof(&spec, &fit_rows, &target_columns, args.te_smoothing);
    let log_val = encode(&spec, &val_rows);
    let log_test = encode(&spec, &test_rows);

    let raw_train = encode_training_oof(&spec_raw, &fit_rows, &target_columns, args.te_smoothing);
    let raw_val = encode(&spec_raw, &val_rows);
    let raw_test = encode(&spec_raw, &test_rows);
    let train_rows = fit_rows;

    // ---- 5. train ------------------------------------------------------------
    println!("\n== 5. train ==");
    let linear = Config {
        name: match args.target {
            Target::Log1p => "Linear regression (log target)",
            Target::LogPricePerSqft => "Linear regression (rate target)",
            Target::Raw => "Linear regression (raw target)",
        },
        hidden: vec![],
        epochs: args.epochs,
        batch: 512,
        lr: 3e-2,
        weight_decay: 0.0,
        activation: Activation::Relu,
        dropout: 0.0,
        layer_norm: false,
        patience: args.patience,
        seed: 11,
        device: dev,
    };
    let mlp = Config {
        name: match args.target {
            Target::Log1p => "MLP tuned (log target)",
            Target::LogPricePerSqft => "MLP tuned (rate target)",
            Target::Raw => "MLP tuned (raw target)",
        },
        hidden: args.hidden.clone(),
        epochs: args.epochs,
        batch: args.batch,
        lr: args.learning_rate,
        weight_decay: args.weight_decay,
        activation: args.activation,
        dropout: args.dropout,
        layer_norm: args.layer_norm,
        patience: args.patience,
        seed: 23,
        device: dev,
    };
    let mlp_raw = Config {
        name: "MLP tuned (raw target)",
        hidden: args.hidden.clone(),
        epochs: args.epochs,
        batch: args.batch,
        lr: args.learning_rate,
        weight_decay: args.weight_decay,
        activation: args.activation,
        dropout: args.dropout,
        layer_norm: args.layer_norm,
        patience: args.patience,
        seed: 23,
        device: dev,
    };

    let (linear_model, linear_run) = train::train(&spec, &log_train, &log_val, &log_test, &linear);
    let (mlp_model, mlp_run) = train::train(&spec, &log_train, &log_val, &log_test, &mlp);
    let (_, mlp_raw_run) = train::train(&spec_raw, &raw_train, &raw_val, &raw_test, &mlp_raw);
    let runs = vec![&linear_run, &mlp_run, &mlp_raw_run];

    // ---- 6. pick a winner and export ----------------------------------------
    println!("\n== 6. evaluate & export ==");
    let winner_is_mlp = mlp_run.test.mdape <= linear_run.test.mdape;
    let (winner, winner_model) = if winner_is_mlp {
        (&mlp_run, &mlp_model)
    } else {
        (&linear_run, &linear_model)
    };
    for r in &runs {
        println!(
            "  {:<32} MAE {:>12.0}  RMSE {:>12.0}  R2 {:>6.3}  MdAPE {:>5.1}%",
            r.name, r.test.mae, r.test.rmse, r.test.r2, r.test.mdape
        );
    }
    println!("winner: {}", winner.name);

    let mut winner_spec = spec.clone();
    winner_spec.layer_dims = winner.layer_dims.clone();
    winner_spec.activation = winner.activation;
    winner_spec.dropout = winner.dropout;
    winner_spec.layer_norm = winner.layer_norm;

    std::fs::create_dir_all(&args.models).context("create models dir")?;
    winner_model
        .save(args.models.join("house_price.ve"))
        .context("save weights")?;
    std::fs::write(
        args.models.join("preprocess.json"),
        serde_json::to_string_pretty(&winner_spec).context("serialise spec")?,
    )
    .context("write preprocess.json")?;
    // The frontend's city dropdown: exactly the values the model has a column
    // for, minus the catch-all.
    let locations: Vec<&String> = winner_spec.categorical[0]
        .vocab
        .iter()
        .filter(|v| *v != hp_core::features::OTHER)
        .collect();
    std::fs::write(
        args.models.join("locations.json"),
        serde_json::to_string_pretty(&locations).context("serialise locations")?,
    )
    .context("write locations.json")?;
    println!("exported to {}/", args.models.display());

    // Sanity check, the equivalent of reloading a pickle before trusting it: a
    // fresh model with the exported weights must reproduce a prediction.
    let reloaded = hp_core::Mlp::with_options(
        &winner_spec.layer_dims,
        999,
        winner_spec.activation,
        winner_spec.dropout,
        winner_spec.layer_norm,
    )
    .to(dev);
    reloaded
        .load(args.models.join("house_price.ve"))
        .context("reload weights")?;
    let sample = &log_test.x[0..1];
    let a = hp_core::predict(winner_model, sample, winner_spec.input_dim(), 1, dev)[0];
    let b = hp_core::predict(&reloaded, sample, winner_spec.input_dim(), 1, dev)[0];
    anyhow::ensure!(
        (a - b).abs() < 1e-6,
        "reloaded weights predict differently: {a} vs {b}"
    );
    println!(
        "reload check: {:.0} rupees, identical after a round trip",
        winner_spec.decode_target(f64::from(b), log_test.area_sqft[0])
    );

    // ---- 7. result charts ----------------------------------------------------
    let predicted = train::predict_rupees(winner_model, &spec, &log_test, dev);
    let pairs: Vec<(f64, f64)> = log_test.price.iter().copied().zip(predicted).collect();
    plot::scatter_log(
        &args.reports.join("07_predicted_vs_actual.svg"),
        &pairs,
        "Predicted vs actual (test set)",
        "actual price (log)",
        "predicted price (log)",
        true,
        true,
    )?;
    plot::lines(
        &args.reports.join("08_training_curves.svg"),
        &[
            ("train", winner.train_curve.clone()),
            ("validation", winner.val_curve.clone()),
        ],
        &format!("Training curves - {}", winner.name),
        "epoch",
        "MSE (model space)",
    )?;

    // ---- 7b. the baseline that actually matters ------------------------------
    let baseline = rate_card_baseline(&train_rows, &test_rows);
    println!(
        "  {:<32} MAE {:>12.0}  RMSE {:>12.0}  R2 {:>6.3}  MdAPE {:>5.1}%",
        "Rate card (city median x area)", baseline.mae, baseline.rmse, baseline.r2, baseline.mdape
    );

    // ---- 8. what de-duplication was worth ------------------------------------
    println!("\n== 8. leakage check: the same model without de-duplication ==");
    let leaky = leakage_run(
        &raw,
        args.max_locations,
        args.max_localities,
        args.max_societies,
        args.epochs,
        &winner.layer_dims,
        dev,
    );
    println!(
        "  with duplicates    MdAPE {:.1}%  R2 {:.3}",
        leaky.test.mdape, leaky.test.r2
    );
    println!(
        "  de-duplicated      MdAPE {:.1}%  R2 {:.3}",
        winner.test.mdape, winner.test.r2
    );

    // ---- 9. bonus: cross-validation -----------------------------------------
    let cv = args.cv.then(|| {
        println!("\n== 9. 5-fold cross-validation ==");
        cross_validate(&spec, &train_rows, &test_rows, winner, args.epochs, dev)
    });

    // ---- 10. report ----------------------------------------------------------
    let report = report::Report {
        profile,
        cleaning: &log,
        society,
        facts: &facts,
        spec: &winner_spec,
        train_rows: log_train.len(),
        val_rows: log_val.len(),
        test_rows: log_test.len(),
        runs,
        winner: winner.name,
        baseline,
        leaky: &leaky,
        cv,
    };
    let path = args.reports.join("REPORT.md");
    std::fs::write(&path, report.render()).context("write report")?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// A column's header and how to read it off a row.
type ColumnAccessor = (&'static str, fn(&dataset::RawListing) -> &str);

/// Per-column missing rate, cardinality and an example value.
fn profile_columns(raw: &[dataset::RawListing]) -> Vec<report::Column> {
    let columns: [ColumnAccessor; 20] = [
        ("Title", |r| &r.title),
        ("Description", |r| &r.description),
        ("Amount(in rupees)", |r| &r.amount),
        ("Price (in rupees)", |r| &r.rate),
        ("location", |r| &r.location),
        ("Carpet Area", |r| &r.carpet_area),
        ("Status", |r| &r.status),
        ("Floor", |r| &r.floor),
        ("Transaction", |r| &r.transaction),
        ("Furnishing", |r| &r.furnishing),
        ("facing", |r| &r.facing),
        ("overlooking", |r| &r.overlooking),
        ("Society", |r| &r.society),
        ("Bathroom", |r| &r.bathroom),
        ("Balcony", |r| &r.balcony),
        ("Car Parking", |r| &r.car_parking),
        ("Ownership", |r| &r.ownership),
        ("Super Area", |r| &r.super_area),
        ("Dimensions", |r| &r.dimensions),
        ("Plot Area", |r| &r.plot_area),
    ];

    columns
        .iter()
        .map(|(name, get)| {
            let mut distinct = std::collections::HashSet::new();
            let mut missing = 0usize;
            let mut example = String::new();
            for r in raw {
                let v = get(r).trim();
                if v.is_empty() {
                    missing += 1;
                } else if example.is_empty() {
                    example = v.chars().take(40).collect();
                    if v.chars().count() > 40 {
                        example.push_str("...");
                    }
                }
                distinct.insert(v);
            }
            report::Column {
                name: (*name).to_string(),
                missing_pct: 100.0 * missing as f64 / raw.len() as f64,
                distinct: distinct.len(),
                example,
            }
        })
        .collect()
}

/// The estimate an agent would make without any model: the city's median price
/// per square foot, times the floor area.
///
/// This is the bar the network actually has to clear. Beating a linear model on
/// the same engineered features says the network found interactions; beating
/// this says the whole pipeline was worth building at all.
fn rate_card_baseline(train_rows: &[dataset::Row], test_rows: &[dataset::Row]) -> train::Metrics {
    let mut by_city: HashMap<&str, Vec<f64>> = HashMap::new();
    for r in train_rows {
        by_city
            .entry(r.listing.location.as_str())
            .or_default()
            .push(r.price_per_sqft);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        dataset::percentile(v, 50.0)
    };
    let rates: HashMap<&str, f64> = by_city
        .iter_mut()
        .map(|(city, r)| (*city, median(r)))
        .collect();
    let mut all: Vec<f64> = train_rows.iter().map(|r| r.price_per_sqft).collect();
    let overall = median(&mut all);

    let predicted: Vec<f64> = test_rows
        .iter()
        .map(|r| {
            let rate = rates
                .get(r.listing.location.as_str())
                .copied()
                .unwrap_or(overall);
            rate * r.listing.area_sqft.unwrap_or(0.0)
        })
        .collect();
    let actual: Vec<f64> = test_rows.iter().map(|r| r.price).collect();
    train::evaluate(&actual, &predicted)
}

/// Trains the winning architecture on the file with its duplicates left in.
///
/// This exists to put a number on the leakage rather than assert it: it is the
/// score the project would have reported if de-duplication had been skipped,
/// which is a mistake that looks like a great result.
fn leakage_run(
    raw: &[dataset::RawListing],
    max_locations: usize,
    max_localities: usize,
    max_societies: usize,
    epochs: usize,
    layer_dims: &[usize],
    dev: vearo::Device,
) -> train::Run {
    let (rows, _) = dataset::build(raw, false);
    let (train_rows, test_rows) = dataset::train_test_split(rows, 0.2, 20_260_902);
    let mut spec = dataset::fit_spec(
        &train_rows,
        Target::Log1p,
        max_locations,
        max_localities,
        max_societies,
        &[],
    );
    fit_target_scaling(&mut spec, &train_rows);
    let mut tr = encode(&spec, &train_rows);
    let va = tr.split_off(0.1);
    let te = encode(&spec, &test_rows);
    let cfg = Config {
        name: "MLP 128-64, duplicates left in",
        hidden: layer_dims[1..layer_dims.len() - 1].to_vec(),
        epochs,
        batch: 512,
        lr: 3e-3,
        weight_decay: 1e-4,
        activation: Activation::Relu,
        dropout: 0.0,
        layer_norm: false,
        patience: 10,
        seed: 23,
        device: dev,
    };
    train::train(&spec, &tr, &va, &te, &cfg).1
}

/// Retrains the winning architecture on five folds of the full data.
fn cross_validate(
    spec: &FeatureSpec,
    train_rows: &[dataset::Row],
    test_rows: &[dataset::Row],
    winner: &train::Run,
    epochs: usize,
    dev: vearo::Device,
) -> Vec<train::Metrics> {
    let all: Vec<&dataset::Row> = train_rows.iter().chain(test_rows).collect();
    let fold_size = all.len() / 5;
    let hidden: Vec<usize> = winner.layer_dims[1..winner.layer_dims.len() - 1].to_vec();

    (0..5)
        .map(|fold| {
            let start = fold * fold_size;
            let end = if fold == 4 {
                all.len()
            } else {
                start + fold_size
            };
            let mut held = Encoded {
                x: vec![],
                y: vec![],
                price: vec![],
                area_sqft: vec![],
            };
            let mut kept = Encoded {
                x: vec![],
                y: vec![],
                price: vec![],
                area_sqft: vec![],
            };
            for (i, r) in all.iter().enumerate() {
                let target = if i >= start && i < end {
                    &mut held
                } else {
                    &mut kept
                };
                let area = row_area(r);
                target.x.push(spec.encode(&r.listing));
                target.y.push(spec.encode_target(r.price, area));
                target.price.push(r.price);
                target.area_sqft.push(area);
            }
            let val = kept.split_off(0.1);
            let cfg = Config {
                name: "cv fold",
                hidden: hidden.clone(),
                epochs,
                batch: 512,
                lr: 3e-3,
                weight_decay: 1e-4,
                activation: winner.activation,
                dropout: winner.dropout,
                layer_norm: winner.layer_norm,
                patience: 10,
                seed: 23 + fold as u64,
                device: dev,
            };
            let (_, run) = train::train(spec, &kept, &val, &held, &cfg);
            println!(
                "  fold {fold}: R2 {:.3}  MdAPE {:.1}%",
                run.test.r2, run.test.mdape
            );
            run.test
        })
        .collect()
}
