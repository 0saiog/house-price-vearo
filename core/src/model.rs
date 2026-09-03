//! The network, built on Vearo.
//!
//! Vearo doesn't have a `Sequential`, so the layers are just a `Vec<Linear>`
//! with an activation between them and nothing on the output. The trainer and
//! the API use this same type, so a checkpoint saved by one always fits the
//! other.

use vearo::nn::{Dropout, LayerNorm, Linear, Module};
use vearo::{Device, Tensor};

use crate::features::Activation;

/// A multilayer perceptron over `layer_dims`, input width first, output last.
///
/// `[d, 1]` is a plain linear regression; `[d, 128, 64, 1]` is the deep model.
pub struct Mlp {
    layers: Vec<Linear>,
    norms: Vec<LayerNorm>,
    dropouts: Vec<Dropout>,
    activation: Activation,
    layer_norm: bool,
}

impl Mlp {
    /// Builds the network. Each layer gets its own seed so two models with
    /// different shapes do not share initial weights by accident.
    ///
    /// # Panics
    /// Panics if `layer_dims` has fewer than two entries.
    #[must_use]
    pub fn new(layer_dims: &[usize], seed: u64) -> Self {
        Self::with_options(layer_dims, seed, Activation::Relu, 0.0, false)
    }

    /// Builds a configurable MLP whose full architecture can be reconstructed
    /// from the exported feature spec.
    #[must_use]
    pub fn with_options(
        layer_dims: &[usize],
        seed: u64,
        activation: Activation,
        dropout: f32,
        layer_norm: bool,
    ) -> Self {
        assert!(
            layer_dims.len() >= 2,
            "a network needs an input and an output width"
        );
        assert!((0.0..1.0).contains(&dropout), "dropout must be in [0, 1)");
        let layers = layer_dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| Linear::new(w[0], w[1], true, seed + i as u64))
            .collect();
        let norms = layer_dims[1..layer_dims.len() - 1]
            .iter()
            .map(|width| LayerNorm::new(*width, 1e-5))
            .collect();
        let dropouts = (0..layer_dims.len() - 2)
            .map(|i| Dropout::new(dropout, seed ^ (0xd0_0f_u64 + i as u64)))
            .collect();
        Self {
            layers,
            norms,
            dropouts,
            activation,
            layer_norm,
        }
    }

    /// Forward pass over a `[batch, input_dim]` tensor.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last = self.layers.len() - 1;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i != last {
                if self.layer_norm {
                    h = self.norms[i].forward(&h);
                }
                h = match self.activation {
                    Activation::Relu => h.relu(),
                    Activation::Gelu => h.gelu(),
                };
                h = self.dropouts[i].forward(&h);
            }
        }
        h
    }

    /// Every trainable tensor, in a stable order - this is the order
    /// `save_checkpoint` writes and `load_checkpoint` expects.
    #[must_use]
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut parameters: Vec<Tensor> = self.layers.iter().flat_map(Module::parameters).collect();
        if self.layer_norm {
            parameters.extend(self.norms.iter().flat_map(LayerNorm::parameters));
        }
        parameters
    }

    /// Moves the network onto `device`.
    #[must_use]
    pub fn to(&self, device: Device) -> Self {
        Self {
            layers: self.layers.iter().map(|l| l.to(device)).collect(),
            norms: self
                .norms
                .iter()
                .map(|norm| LayerNorm {
                    weight: norm.weight.to(device),
                    bias: norm.bias.to(device),
                    eps: norm.eps,
                })
                .collect(),
            dropouts: self.dropouts.iter().map(|d| d.to(device)).collect(),
            activation: self.activation,
            layer_norm: self.layer_norm,
        }
    }

    /// Restores weights from a `.ve` checkpoint written by [`vearo::checkpoint::save_checkpoint`].
    ///
    /// # Errors
    /// Returns an I/O error if the checkpoint cannot be read.
    ///
    /// # Panics
    /// Panics if the checkpoint's shapes do not match this network, which is the
    /// intended outcome: silently serving mismatched weights would be worse.
    pub fn load(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        vearo::checkpoint::load_checkpoint(&self.parameters(), path)
    }

    /// Saves weights to a `.ve` checkpoint.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be written.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        vearo::checkpoint::save_checkpoint(&self.parameters(), path)
    }
}

/// Runs the network over `rows` without building an autograd tape, in chunks, and
/// returns one output per row. `device` must match the device the model is on.
///
/// Prediction must not record onto the tape: the service would otherwise grow a
/// tape forever, and during training it would corrupt the step in progress.
#[must_use]
pub fn predict(
    model: &Mlp,
    rows: &[Vec<f32>],
    input_dim: usize,
    chunk: usize,
    device: Device,
) -> Vec<f32> {
    let was_training = vearo::is_training();
    let was_autograd = vearo::core::is_autograd_enabled();
    vearo::set_training(false);
    vearo::core::set_autograd_enabled(false);

    let mut out = Vec::with_capacity(rows.len());
    for group in rows.chunks(chunk.max(1)) {
        let flat: Vec<f32> = group.iter().flat_map(|r| r.iter().copied()).collect();
        let x = Tensor::from_f32(&flat, [group.len(), input_dim]).to(device);
        out.extend(model.forward(&x).to_vec_f32());
    }

    vearo::core::set_autograd_enabled(was_autograd);
    vearo::set_training(was_training);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trips_through_a_fresh_model() {
        vearo::backend_cpu::init();
        vearo::autograd::init();

        let dims = [4, 8, 1];
        let trained = Mlp::new(&dims, 1);
        let rows = vec![vec![0.5, -1.0, 2.0, 0.25]];
        let before = predict(&trained, &rows, 4, 32, Device::Cpu);

        let path = std::env::temp_dir().join("hp_core_model_test.ve");
        trained.save(&path).unwrap();

        // A differently seeded model starts with different weights, and must end
        // up predicting identically once the checkpoint is loaded.
        let restored = Mlp::new(&dims, 99);
        assert_ne!(predict(&restored, &rows, 4, 32, Device::Cpu), before);
        restored.load(&path).unwrap();
        assert_eq!(predict(&restored, &rows, 4, 32, Device::Cpu), before);

        std::fs::remove_file(path).unwrap();
    }
}
