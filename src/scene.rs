//! What a visualizer plug-in asks to draw. It comes from the plug-in as
//! JSON and is checked here, once: past [`Scene::parse`] the renderer can
//! trust the shape and the sizes, so a plug-in cannot ask the GPU for more
//! than these limits.

use serde::Deserialize;
use std::collections::HashSet;

pub const MAX_TARGETS: usize = 4;
pub const MAX_PASSES: usize = 16;
pub const MAX_SHADER_BYTES: usize = 256 * 1024;
pub const MAX_VERTICES_PER_FRAME: u64 = 2_000_000;
pub const MAX_DATA_BYTES: u32 = 1 << 20;
pub const MIN_SCALE: f32 = 0.1;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    #[serde(default)]
    pub targets: Vec<Target>,
    pub passes: Vec<Pass>,
    /// Bytes the plug-in hands over every frame, seen by shaders as `data`.
    #[serde(default)]
    pub data_bytes: u32,
}

/// A picture a pass draws into and later passes, or the next frame, read.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub name: String,
    /// Size as a share of the window.
    #[serde(default = "full_size")]
    pub scale: f32,
}

fn full_size() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pass {
    /// WGSL with the entry points `vs` and `fs`.
    pub shader: String,
    /// Where it draws: a target's name, or the window when absent.
    #[serde(default)]
    pub target: Option<String>,
    /// Targets it reads, bound in this order from binding 4 on.
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default = "fullscreen_triangle")]
    pub vertices: u32,
    #[serde(default = "one")]
    pub instances: u32,
    #[serde(default)]
    pub topology: Topology,
    #[serde(default)]
    pub blend: Blend,
}

fn fullscreen_triangle() -> u32 {
    3
}

fn one() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Topology {
    #[default]
    Triangles,
    Lines,
    Points,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Blend {
    #[default]
    None,
    Alpha,
    Add,
}

impl Scene {
    pub fn parse(json: &[u8]) -> Result<Scene, String> {
        let scene: Scene = serde_json::from_slice(json).map_err(|e| format!("scene: {e}"))?;
        scene.check()?;
        Ok(scene)
    }

    fn check(&self) -> Result<(), String> {
        if self.passes.is_empty() || self.passes.len() > MAX_PASSES {
            return Err(format!("a scene has 1 to {MAX_PASSES} passes, this one {}", self.passes.len()));
        }
        if self.targets.len() > MAX_TARGETS {
            return Err(format!("a scene has at most {MAX_TARGETS} targets, this one {}", self.targets.len()));
        }
        if self.data_bytes > MAX_DATA_BYTES {
            return Err(format!("data_bytes is at most {MAX_DATA_BYTES}, this one {}", self.data_bytes));
        }
        self.check_targets()?;
        self.check_passes()
    }

    fn check_targets(&self) -> Result<(), String> {
        let mut seen = HashSet::new();
        for t in &self.targets {
            if !seen.insert(t.name.as_str()) {
                return Err(format!("two targets are named {:?}", t.name));
            }
            if !(MIN_SCALE..=1.0).contains(&t.scale) {
                return Err(format!("target {:?}: scale is {MIN_SCALE} to 1, this one {}", t.name, t.scale));
            }
        }
        Ok(())
    }

    fn check_passes(&self) -> Result<(), String> {
        let mut written = HashSet::new();
        let mut vertices = 0u64;
        for (i, pass) in self.passes.iter().enumerate() {
            let at = format!("pass {}", i + 1);
            if pass.shader.len() > MAX_SHADER_BYTES {
                return Err(format!("{at}: a shader is at most {MAX_SHADER_BYTES} bytes"));
            }
            if pass.inputs.len() > MAX_TARGETS {
                return Err(format!("{at}: a pass reads at most {MAX_TARGETS} targets"));
            }
            for name in pass.target.iter().chain(&pass.inputs) {
                if self.target_index(name).is_none() {
                    return Err(format!("{at}: there is no target {name:?}"));
                }
            }
            // One writer per target is what makes "this frame or the last" a plain rule.
            if !written.insert(pass.target.as_deref()) {
                return Err(format!("{at}: another pass already draws there"));
            }
            vertices += u64::from(pass.vertices) * u64::from(pass.instances);
        }
        if vertices > MAX_VERTICES_PER_FRAME {
            return Err(format!("a frame draws at most {MAX_VERTICES_PER_FRAME} vertices, this one {vertices}"));
        }
        if !written.contains(&None) {
            return Err("no pass draws to the window".into());
        }
        Ok(())
    }

    pub fn target_index(&self, name: &str) -> Option<usize> {
        self.targets.iter().position(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(json: &str) -> String {
        Scene::parse(json.as_bytes()).unwrap_err()
    }

    #[test]
    fn a_pass_needs_only_its_shader() {
        let s = Scene::parse(br#"{"passes":[{"shader":"x"}]}"#).unwrap();
        assert_eq!((s.passes[0].vertices, s.passes[0].instances), (3, 1));
        assert_eq!((s.passes[0].topology, s.passes[0].blend), (Topology::Triangles, Blend::None));
        assert_eq!(s.passes[0].target, None);
    }

    #[test]
    fn a_target_may_read_itself_from_the_last_frame() {
        let s = Scene::parse(br#"{"targets":[{"name":"a","scale":0.5}],"passes":[{"shader":"x","target":"a","inputs":["a"]},{"shader":"y","inputs":["a"]}]}"#);
        assert_eq!(s.unwrap().targets[0].scale, 0.5);
    }

    #[test]
    fn what_is_too_much_or_makes_no_sense_is_refused() {
        let shader = "x".repeat(MAX_SHADER_BYTES + 1);
        for (json, wanted) in [
            (r#"{"passes":[]}"#.to_string(), "1 to 16 passes"),
            (format!(r#"{{"passes":[{{"shader":"{shader}"}}]}}"#), "a shader is at most"),
            (r#"{"passes":[{"shader":"x"}],"data_bytes":99999999}"#.to_string(), "data_bytes is at most"),
            (r#"{"passes":[{"shader":"x","vertices":3000000}]}"#.to_string(), "at most 2000000 vertices"),
            (r#"{"passes":[{"shader":"x","vertices":4294967295,"instances":4294967295}]}"#.to_string(), "at most 2000000 vertices"),
            (r#"{"passes":[{"shader":"x","target":"nope"}]}"#.to_string(), "no target \"nope\""),
            (r#"{"passes":[{"shader":"x","inputs":["nope"]}]}"#.to_string(), "no target \"nope\""),
            (r#"{"passes":[{"shader":"x"},{"shader":"y"}]}"#.to_string(), "already draws there"),
            (r#"{"targets":[{"name":"a"}],"passes":[{"shader":"x","target":"a"}]}"#.to_string(), "no pass draws to the window"),
            (r#"{"targets":[{"name":"a"},{"name":"a"}],"passes":[{"shader":"x"}]}"#.to_string(), "two targets"),
            (r#"{"targets":[{"name":"a","scale":2}],"passes":[{"shader":"x"}]}"#.to_string(), "scale is"),
            (
                r#"{"targets":[{"name":"a"},{"name":"b"},{"name":"c"},{"name":"d"},{"name":"e"}],"passes":[{"shader":"x"}]}"#.to_string(),
                "at most 4 targets",
            ),
            (r#"{"targets":[{"name":"a"}],"passes":[{"shader":"x","inputs":["a","a","a","a","a"]}]}"#.to_string(), "reads at most 4"),
            (r#"{"passes":[{"shader":"x","compute":true}]}"#.to_string(), "unknown field"),
        ] {
            let got = error(&json);
            assert!(got.contains(wanted), "{got:?} lacks {wanted:?}");
        }
    }
}
