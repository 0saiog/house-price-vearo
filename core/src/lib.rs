//! Domain code shared by the trainer and the inference service.
//!
//! Keeping the column parsers, the feature spec and the model in one crate is
//! what stops the service from encoding a listing differently than the trainer
//! did - the single most common way an ML web app silently returns nonsense.

pub mod clean;
pub mod device;
pub mod features;
pub mod model;

pub use device::{Preference, select};
pub use features::{Categorical, FeatureSpec, Listing, Target};
pub use model::{Mlp, predict};
