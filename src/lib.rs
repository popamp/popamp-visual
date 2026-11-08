//! Visualizer plug-ins on the GPU. Built only with the player's
//! `visualizers` feature, so a build without it carries no GPU code.

mod renderer;
pub mod scene;
mod window;

pub use renderer::{Frame, Gpu, Renderer, SPECTRUM_LEN, WAVE_LEN};
pub use scene::Scene;
pub use wgpu;
pub use window::{run, Control, Feed};
