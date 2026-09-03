//! Loading the model and running it.
//!
//! Vearo's `Tensor` has `Cell`s inside, so it's `Send` but not `Sync`, and the
//! autograd tape is per thread. So the model sits behind a mutex and predictions
//! run on a blocking thread. That's what you'd want anyway for a CPU-bound
//! forward pass inside an async server.

use std::sync::{Arc, Mutex};

use hp_core::device::{self, Preference};
use hp_core::features::OTHER;
use hp_core::{FeatureSpec, Listing, Mlp, predict};
use vearo::Device;

/// Why the service could not start.
#[derive(Debug)]
pub enum LoadError {
    /// A file was missing or unreadable.
    Io(std::path::PathBuf, std::io::Error),
    /// `preprocess.json` was not valid JSON for a `FeatureSpec`.
    Spec(serde_json::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(
                f,
                "cannot read {} ({e}). Train the model first: cargo run --release -p ml",
                path.display()
            ),
            Self::Spec(e) => write!(f, "preprocess.json is not a valid feature spec: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The loaded model and the spec that feeds it.
#[derive(Clone)]
pub struct Engine {
    /// Feature spec. Plain data, so it is shared without a lock.
    pub spec: Arc<FeatureSpec>,
    /// The network. Behind a mutex because Vearo tensors are not `Sync`.
    model: Arc<Mutex<Mlp>>,
    /// Where the weights live.
    device: Device,
}

impl Engine {
    /// Loads the checkpoint and the feature spec from disk.
    ///
    /// Called once at startup, never per request: reading and rebuilding the
    /// network on every call would dominate the latency and leak memory.
    ///
    /// # Errors
    /// Returns [`LoadError`] if either file is missing or malformed.
    pub fn load(
        model_path: &std::path::Path,
        preprocess_path: &std::path::Path,
    ) -> Result<Self, LoadError> {
        // A single request is one row through a 19,841-parameter network, so the
        // GPU would spend more time on the launch than the arithmetic. Serving
        // stays on the CPU even in a CUDA build; the GPU is for training.
        let device = device::select(Preference::Cpu);

        let json = std::fs::read_to_string(preprocess_path)
            .map_err(|e| LoadError::Io(preprocess_path.to_path_buf(), e))?;
        let spec: FeatureSpec = serde_json::from_str(&json).map_err(LoadError::Spec)?;

        let model = Mlp::with_options(
            &spec.layer_dims,
            0,
            spec.activation,
            spec.dropout,
            spec.layer_norm,
        )
        .to(device);
        model
            .load(model_path)
            .map_err(|e| LoadError::Io(model_path.to_path_buf(), e))?;

        Ok(Self {
            spec: Arc::new(spec),
            model: Arc::new(Mutex::new(model)),
            device,
        })
    }

    /// Predicts a price in rupees.
    ///
    /// # Panics
    /// Panics if the model mutex was poisoned by a previous panic mid-prediction.
    #[must_use]
    pub fn predict(&self, listing: &Listing) -> f64 {
        let row = vec![self.spec.encode(listing)];
        let model = self.model.lock().expect("model mutex poisoned");
        let raw = predict(&model, &row, self.spec.input_dim(), 1, self.device)[0];
        // Clamp at zero: the network is unconstrained, and a negative rupee price
        // is a nonsense answer to hand a caller even if the maths produced one.
        // A rate target predicts rupees per square foot, so undoing it needs the
        // area the caller gave. `validate` has already rejected a non-positive one.
        let area = listing.area_sqft.unwrap_or(1.0);
        self.spec.decode_target(f64::from(raw), area).max(0.0)
    }

    /// Whether the model has a dedicated column for this city.
    #[must_use]
    pub fn knows_location(&self, location: &str) -> bool {
        let needle = location.trim().to_lowercase();
        self.spec.categorical[0]
            .vocab
            .iter()
            .any(|v| *v == needle && v != OTHER)
    }

    /// The cities the model has columns for, for the frontend dropdown.
    #[must_use]
    pub fn locations(&self) -> Vec<&str> {
        self.spec.categorical[0]
            .vocab
            .iter()
            .filter(|v| *v != OTHER)
            .map(String::as_str)
            .collect()
    }
}
