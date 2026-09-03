//! Mini-batch training on Vearo, and the metrics.
//!
//! Vearo has no slice op, so each batch gets gathered on the CPU and passed to
//! `Tensor::from_f32`. The autograd tape gets reset every step so memory
//! doesn't grow over a long run.

use hp_core::features::{Activation, FeatureSpec};
use hp_core::model::{Mlp, predict};
use vearo::nn::SimpleRng;
use vearo::optim::{AdamW, CosineSchedule};
use vearo::{Device, Tensor};

/// Encoded design matrix and its targets, ready for the optimiser.
pub struct Encoded {
    /// One encoded feature vector per row.
    pub x: Vec<Vec<f32>>,
    /// Target in model space, matching `x` row for row.
    pub y: Vec<f32>,
    /// Price in rupees, for reporting metrics in the units people care about.
    pub price: Vec<f64>,
    /// Floor area needed to turn a predicted price-per-square-foot back into rupees.
    pub area_sqft: Vec<f64>,
}

impl Encoded {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Splits off the last `fraction` of rows, for a validation set.
    #[must_use]
    pub fn split_off(&mut self, fraction: f64) -> Self {
        let keep = self.len() - (self.len() as f64 * fraction).round() as usize;
        Self {
            x: self.x.split_off(keep),
            y: self.y.split_off(keep),
            price: self.price.split_off(keep),
            area_sqft: self.area_sqft.split_off(keep),
        }
    }
}

/// One model configuration to train.
#[derive(Clone)]
pub struct Config {
    /// Name used in the comparison table.
    pub name: &'static str,
    /// Hidden widths; empty means a plain linear regression.
    pub hidden: Vec<usize>,
    /// Passes over the training split.
    pub epochs: usize,
    /// Rows per optimiser step.
    pub batch: usize,
    /// Peak learning rate.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Hidden-layer non-linearity.
    pub activation: Activation,
    /// Hidden activation dropout probability.
    pub dropout: f32,
    /// Whether to normalise each hidden layer before activation.
    pub layer_norm: bool,
    /// Stop after this many epochs without a validation-loss improvement.
    pub patience: usize,
    /// Seed for weight initialisation and batch shuffling.
    pub seed: u64,
    /// Where the tensors live.
    pub device: Device,
}

/// Test-set accuracy, in rupees.
#[derive(Clone, Copy)]
pub struct Metrics {
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Coefficient of determination.
    pub r2: f64,
    /// Median absolute percentage error - the honest headline for skewed prices.
    pub mdape: f64,
}

/// Scores predictions against actual prices, both in rupees.
#[must_use]
pub fn evaluate(actual: &[f64], predicted: &[f64]) -> Metrics {
    let n = actual.len() as f64;
    let mean = actual.iter().sum::<f64>() / n;
    let mut abs_errors = 0.0;
    let mut sq_errors = 0.0;
    let mut ss_tot = 0.0;
    let mut pct: Vec<f64> = Vec::with_capacity(actual.len());
    for (a, p) in actual.iter().zip(predicted) {
        abs_errors += (a - p).abs();
        sq_errors += (a - p).powi(2);
        ss_tot += (a - mean).powi(2);
        pct.push(((a - p) / a).abs() * 100.0);
    }
    pct.sort_by(f64::total_cmp);
    Metrics {
        mae: abs_errors / n,
        rmse: (sq_errors / n).sqrt(),
        r2: 1.0 - sq_errors / ss_tot,
        mdape: pct[pct.len() / 2],
    }
}

/// A finished training run.
pub struct Run {
    /// Model name.
    pub name: &'static str,
    /// Layer widths actually used.
    pub layer_dims: Vec<usize>,
    /// Trainable parameter count.
    pub params: usize,
    /// Per-epoch training loss, in model space.
    pub train_curve: Vec<f64>,
    /// Per-epoch validation loss, in model space.
    pub val_curve: Vec<f64>,
    /// Epoch whose validation loss was lowest.
    pub best_epoch: usize,
    /// Wall-clock training time.
    pub seconds: f64,
    /// Metrics on the held-out test split.
    pub test: Metrics,
    /// Device the run executed on.
    pub device: &'static str,
    /// Hidden-layer non-linearity.
    pub activation: Activation,
    /// Hidden activation dropout probability used while fitting.
    pub dropout: f32,
    /// Whether hidden layer normalisation was enabled.
    pub layer_norm: bool,
}

/// Predicts rupee prices for every row of `data`.
#[must_use]
pub fn predict_rupees(model: &Mlp, spec: &FeatureSpec, data: &Encoded, device: Device) -> Vec<f64> {
    predict(model, &data.x, spec.input_dim(), 4096, device)
        .into_iter()
        .zip(&data.area_sqft)
        .map(|(y, area)| spec.decode_target(f64::from(y), *area))
        .collect()
}

/// Trains one configuration and evaluates it on `test`.
///
/// Model selection uses `val`, never `test`: the test split is scored exactly
/// once, right at the end, so the number it gives is honest.
#[must_use]
pub fn train(
    spec: &FeatureSpec,
    train: &Encoded,
    val: &Encoded,
    test: &Encoded,
    cfg: &Config,
) -> (Mlp, Run) {
    let input_dim = spec.input_dim();
    let mut layer_dims = vec![input_dim];
    layer_dims.extend(cfg.hidden.iter().copied());
    layer_dims.push(1);

    let model = Mlp::with_options(
        &layer_dims,
        cfg.seed,
        cfg.activation,
        cfg.dropout,
        cfg.layer_norm,
    )
    .to(cfg.device);
    let params = model.parameters();
    let param_count: usize = params.iter().map(|p| p.shape().numel()).sum();
    let mut opt = AdamW::new(params, cfg.lr, 0.9, 0.999, 1e-8, cfg.weight_decay);

    let steps_per_epoch = train.len().div_ceil(cfg.batch);
    let total_steps = (steps_per_epoch * cfg.epochs) as u32;
    let schedule = CosineSchedule::new(cfg.lr, cfg.lr * 0.05, total_steps / 20, total_steps);

    let mut rng = SimpleRng::new(cfg.seed ^ 0x5eed);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut train_curve = Vec::with_capacity(cfg.epochs);
    let mut val_curve = Vec::with_capacity(cfg.epochs);

    // Keep the weights from the best epoch, not the last: with a cosine schedule
    // the final epoch is usually best, but "usually" is not "always".
    let best_file = tempfile::Builder::new()
        .prefix("hp_best_")
        .suffix(".ve")
        .tempfile()
        .expect("create unique best-checkpoint file");
    let best_path = best_file.path();
    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0;
    let mut stale_epochs = 0usize;

    let started = std::time::Instant::now();
    let mut step = 0u32;

    for epoch in 0..cfg.epochs {
        vearo::set_training(true);
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        for batch in order.chunks(cfg.batch) {
            opt.set_lr(schedule.lr_at(step));
            step += 1;

            let mut xs = Vec::with_capacity(batch.len() * input_dim);
            let mut ys = Vec::with_capacity(batch.len());
            for &i in batch {
                xs.extend_from_slice(&train.x[i]);
                ys.push(train.y[i]);
            }
            let x = Tensor::from_f32(&xs, [batch.len(), input_dim]).to(cfg.device);
            let y = Tensor::from_f32(&ys, [batch.len(), 1]).to(cfg.device);

            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let diff = model.forward(&x).sub(&y);
            let loss = diff.mul(&diff).mean(0, false);
            loss.backward();
            opt.step();

            epoch_loss += f64::from(loss.to_vec_f32()[0]) * batch.len() as f64;
        }

        let train_loss = epoch_loss / train.len() as f64;
        let val_loss = mse_in_model_space(&model, spec, val, cfg.device);
        train_curve.push(train_loss);
        val_curve.push(val_loss);

        if val_loss < best_val {
            best_val = val_loss;
            best_epoch = epoch;
            stale_epochs = 0;
            model.save(best_path).expect("write best checkpoint");
        } else {
            stale_epochs += 1;
        }
        if epoch % 5 == 0 || epoch == cfg.epochs - 1 {
            println!(
                "  [{}] epoch {epoch:>3}  train {train_loss:.5}  val {val_loss:.5}  lr {:.2e}",
                cfg.name,
                opt.lr()
            );
        }
        if stale_epochs >= cfg.patience.max(1) {
            println!(
                "  [{}] early stop at epoch {epoch}; best was {best_epoch}",
                cfg.name
            );
            break;
        }
    }

    model.load(best_path).expect("restore best checkpoint");

    let predicted = predict_rupees(&model, spec, test, cfg.device);
    let run = Run {
        name: cfg.name,
        layer_dims,
        params: param_count,
        train_curve,
        val_curve,
        best_epoch,
        seconds: started.elapsed().as_secs_f64(),
        test: evaluate(&test.price, &predicted),
        device: hp_core::device::name(cfg.device),
        activation: cfg.activation,
        dropout: cfg.dropout,
        layer_norm: cfg.layer_norm,
    };
    (model, run)
}

/// Mean squared error in model space, used for the validation curve.
fn mse_in_model_space(model: &Mlp, spec: &FeatureSpec, data: &Encoded, device: Device) -> f64 {
    let out = predict(model, &data.x, spec.input_dim(), 4096, device);
    let n = out.len() as f64;
    out.iter()
        .zip(&data.y)
        .map(|(p, t)| f64::from(p - t).powi(2))
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_perfect_on_perfect_predictions() {
        let actual = [1.0, 2.0, 3.0, 4.0];
        let m = evaluate(&actual, &actual);
        assert!(m.mae.abs() < 1e-12 && m.rmse.abs() < 1e-12 && m.mdape.abs() < 1e-12);
        assert!((m.r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predicting_the_mean_scores_zero_r2() {
        let actual = [1.0, 2.0, 3.0, 4.0];
        let m = evaluate(&actual, &[2.5; 4]);
        assert!(m.r2.abs() < 1e-12);
    }
}
