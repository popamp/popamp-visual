//! Draws a checked [`Scene`]. The plug-in supplies shaders and bytes; every
//! GPU object and call is made here. Shaders go through wgpu's validation,
//! which also keeps their array and texture accesses inside what they were given. A compute entry
//! point in a shader stays unused: there is no way to start one.

use crate::scene::{Blend, Pass, Scene, Topology};
use parking_lot::Mutex;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub const WAVE_LEN: usize = 1024;
pub const SPECTRUM_LEN: usize = 512;
const FRAME_BYTES: u64 = 16;
const AUDIO_BYTES: u64 = ((WAVE_LEN + SPECTRUM_LEN) * 4) as u64;
const FIRST_INPUT_BINDING: u32 = 4;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// One frame as the player hands it over.
pub struct Frame<'a> {
    pub time: f32,
    pub dt: f32,
    pub wave: &'a [f32; WAVE_LEN],
    pub spectrum: &'a [f32; SPECTRUM_LEN],
    /// Exactly the scene's `data_bytes`.
    pub data: &'a [u8],
}

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The first device error or loss; once set, drawing stops.
    failure: Arc<Mutex<Option<String>>>,
}

impl Gpu {
    /// A device able to draw to `surface`, or to memory when there is none.
    pub fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Gpu, String> {
        let options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            apply_limit_buckets: false,
        };
        let adapter = pollster::block_on(instance.request_adapter(&options)).map_err(|e| format!("no usable GPU: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| format!("cannot open the GPU: {e}"))?;
        let failure = Arc::new(Mutex::new(None));
        let on_error = failure.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| record(&on_error, e.to_string())));
        let on_loss = failure.clone();
        device.set_device_lost_callback(move |_, message| record(&on_loss, format!("the GPU was lost: {message}")));
        Ok(Gpu { adapter, device, queue, failure })
    }

    fn check(&self) -> Result<(), String> {
        self.failure.lock().clone().map_or(Ok(()), Err)
    }
}

fn record(slot: &Mutex<Option<String>>, message: String) {
    slot.lock().get_or_insert(message);
}

struct CompiledPass {
    pipeline: wgpu::RenderPipeline,
    /// Indexed by the frame's flip: the same inputs, a frame apart.
    groups: [wgpu::BindGroup; 2],
    target: Option<usize>,
    vertices: u32,
    instances: u32,
}

struct Buffers {
    frame: wgpu::Buffer,
    audio: wgpu::Buffer,
    data: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

/// Two pictures per target, so a pass can read what it drew a frame ago.
type Targets = Vec<[wgpu::TextureView; 2]>;

pub struct Renderer {
    scene: Scene,
    output: wgpu::TextureFormat,
    size: (u32, u32),
    buffers: Buffers,
    layouts: Vec<wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::RenderPipeline>,
    targets: Targets,
    passes: Vec<CompiledPass>,
    flip: usize,
}

impl Renderer {
    pub fn new(gpu: &Gpu, scene: Scene, output: wgpu::TextureFormat, size: (u32, u32)) -> Result<Renderer, String> {
        let buffers = buffers(gpu, &scene);
        let layouts: Vec<_> = scene.passes.iter().map(|p| layout(gpu, p.inputs.len())).collect();
        let pipelines = compile(gpu, &scene, &layouts, output)?;
        let mut renderer =
            Renderer { scene, output, size: (0, 0), buffers, layouts, pipelines, targets: Vec::new(), passes: Vec::new(), flip: 0 };
        renderer.resize(gpu, size);
        gpu.check()?;
        Ok(renderer)
    }

    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output
    }

    /// Rebuilds what depends on the window size; the picture starts black.
    pub fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) {
        let most = gpu.device.limits().max_texture_dimension_2d;
        let size = (size.0.clamp(1, most), size.1.clamp(1, most));
        if size == self.size {
            return;
        }
        self.size = size;
        self.targets = self.scene.targets.iter().map(|t| [0, 1].map(|_| picture(gpu, size, t.scale))).collect();
        self.passes = self.scene.passes.iter().enumerate().map(|(i, p)| self.bind(gpu, i, p)).collect();
        self.flip = 0;
    }

    fn bind(&self, gpu: &Gpu, index: usize, pass: &Pass) -> CompiledPass {
        CompiledPass {
            pipeline: self.pipelines[index].clone(),
            groups: [0, 1].map(|flip| self.group(gpu, index, pass, flip)),
            target: pass.target.as_deref().and_then(|t| self.scene.target_index(t)),
            vertices: pass.vertices,
            instances: pass.instances,
        }
    }

    /// A target drawn by an earlier pass reads as this frame's, any other as the last frame's.
    fn input_view(&self, index: usize, name: &str, flip: usize) -> &wgpu::TextureView {
        let target = self.scene.target_index(name).expect("checked by Scene::parse");
        let drawn_already = self.scene.passes[..index].iter().any(|p| p.target.as_deref() == Some(name));
        &self.targets[target][if drawn_already { flip } else { 1 - flip }]
    }

    fn group(&self, gpu: &Gpu, index: usize, pass: &Pass, flip: usize) -> wgpu::BindGroup {
        let b = &self.buffers;
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: b.frame.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.audio.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&b.sampler) },
        ];
        for (n, name) in pass.inputs.iter().enumerate() {
            let view = self.input_view(index, name, flip);
            entries
                .push(wgpu::BindGroupEntry { binding: FIRST_INPUT_BINDING + n as u32, resource: wgpu::BindingResource::TextureView(view) });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.layouts[index], entries: &entries })
    }

    /// Draws one frame into `output`, a view of the format given to `new`.
    pub fn draw(&mut self, gpu: &Gpu, output: &wgpu::TextureView, frame: &Frame<'_>) -> Result<(), String> {
        gpu.check()?;
        self.upload(gpu, frame)?;
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for pass in &self.passes {
            let view = pass.target.map_or(output, |t| &self.targets[t][self.flip]);
            let attachment = wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            };
            let mut render =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor { color_attachments: &[Some(attachment)], ..Default::default() });
            render.set_pipeline(&pass.pipeline);
            render.set_bind_group(0, &pass.groups[self.flip], &[]);
            render.draw(0..pass.vertices, 0..pass.instances);
        }
        gpu.queue.submit([encoder.finish()]);
        self.flip = 1 - self.flip;
        gpu.check()
    }

    fn upload(&self, gpu: &Gpu, frame: &Frame<'_>) -> Result<(), String> {
        if frame.data.len() != self.scene.data_bytes as usize {
            return Err(format!("the plug-in gave {} bytes of data, its scene says {}", frame.data.len(), self.scene.data_bytes));
        }
        let (width, height) = self.size;
        let header = [frame.time, frame.dt, width as f32, height as f32];
        gpu.queue.write_buffer(&self.buffers.frame, 0, &floats(&header));
        gpu.queue.write_buffer(&self.buffers.audio, 0, &floats(frame.wave));
        gpu.queue.write_buffer(&self.buffers.audio, WAVE_LEN as u64 * 4, &floats(frame.spectrum));
        if !frame.data.is_empty() {
            gpu.queue.write_buffer(&self.buffers.data, 0, frame.data);
        }
        Ok(())
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn buffers(gpu: &Gpu, scene: &Scene) -> Buffers {
    let d = &gpu.device;
    let make = |size: u64, usage: wgpu::BufferUsages| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    // A shader may declare `data` even when the plug-in sends none, so it always has a size.
    let data_size = u64::from(scene.data_bytes.max(4)).next_multiple_of(4);
    Buffers {
        frame: make(FRAME_BYTES, wgpu::BufferUsages::UNIFORM),
        audio: make(AUDIO_BYTES, wgpu::BufferUsages::STORAGE),
        data: d.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &vec![0; data_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }),
        sampler: d.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
    }
}

fn layout(gpu: &Gpu, inputs: usize) -> wgpu::BindGroupLayout {
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty, count: None };
    let buffer = |ty| wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None };
    let storage = wgpu::BufferBindingType::Storage { read_only: true };
    let mut entries = vec![
        entry(0, buffer(wgpu::BufferBindingType::Uniform)),
        entry(1, buffer(storage)),
        entry(2, buffer(storage)),
        entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
    ];
    for n in 0..inputs as u32 {
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        entries.push(entry(FIRST_INPUT_BINDING + n, texture));
    }
    gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
}

fn compile(
    gpu: &Gpu,
    scene: &Scene,
    layouts: &[wgpu::BindGroupLayout],
    output: wgpu::TextureFormat,
) -> Result<Vec<wgpu::RenderPipeline>, String> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipelines = scene.passes.iter().zip(layouts).map(|(pass, layout)| pipeline(gpu, pass, layout, output)).collect();
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(format!("a shader was refused: {error}")),
        None => Ok(pipelines),
    }
}

fn pipeline(gpu: &Gpu, pass: &Pass, layout: &wgpu::BindGroupLayout, output: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let d = &gpu.device;
    let module =
        d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(pass.shader.as_str().into()) });
    let layout =
        d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    let format = if pass.target.is_some() { TARGET_FORMAT } else { output };
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: wgpu::PrimitiveState { topology: topology(pass.topology), ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: blend(pass.blend), write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn topology(t: Topology) -> wgpu::PrimitiveTopology {
    match t {
        Topology::Triangles => wgpu::PrimitiveTopology::TriangleList,
        Topology::Lines => wgpu::PrimitiveTopology::LineList,
        Topology::Points => wgpu::PrimitiveTopology::PointList,
    }
}

fn blend(b: Blend) -> Option<wgpu::BlendState> {
    match b {
        Blend::None => None,
        Blend::Alpha => Some(wgpu::BlendState::ALPHA_BLENDING),
        Blend::Add => {
            let add = wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            };
            Some(wgpu::BlendState { color: add, alpha: add })
        }
    }
}

fn picture(gpu: &Gpu, (width, height): (u32, u32), scale: f32) -> wgpu::TextureView {
    let side = |n: u32| ((n as f32 * scale) as u32).max(1);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: side(width), height: side(height), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: u32 = 4;
    const WAVE: [f32; WAVE_LEN] = [0.5; WAVE_LEN];
    const SPECTRUM: [f32; SPECTRUM_LEN] = [0.25; SPECTRUM_LEN];
    /// A triangle covering the window, and the bindings every shader sees.
    const PRELUDE: &str = "
        struct Frame { time: f32, dt: f32, width: f32, height: f32 }
        struct Audio { wave: array<f32, 1024>, spectrum: array<f32, 512> }
        @group(0) @binding(0) var<uniform> frame: Frame;
        @group(0) @binding(1) var<storage, read> audio: Audio;
        @group(0) @binding(2) var<storage, read> data: array<f32>;
        @group(0) @binding(3) var smooth_: sampler;
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            return vec4f(f32(i & 1) * 4.0 - 1.0, f32(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
        }
    ";

    /// No GPU at all (a bare build machine) skips the test with a note.
    fn gpu() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        match Gpu::new(&instance, None) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("skipped: {e}");
                None
            }
        }
    }

    fn scene(json: serde_json::Value) -> Scene {
        Scene::parse(json.to_string().as_bytes()).unwrap()
    }

    fn shader(fragment: &str) -> String {
        format!("{PRELUDE}\n{fragment}")
    }

    fn output(gpu: &Gpu) -> wgpu::Texture {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn first_pixel(gpu: &Gpu, texture: &wgpu::Texture) -> [u8; 4] {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * u64::from(SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None };
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout },
            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = buffer.slice(..).get_mapped_range().unwrap();
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    }

    fn draw(gpu: &Gpu, scene: Scene, frames: usize, data: &[u8]) -> [u8; 4] {
        let mut renderer = Renderer::new(gpu, scene, FORMAT, (SIZE, SIZE)).unwrap();
        let texture = output(gpu);
        let view = texture.create_view(&Default::default());
        for _ in 0..frames {
            let frame = Frame { time: 1.0, dt: 0.5, wave: &WAVE, spectrum: &SPECTRUM, data };
            renderer.draw(gpu, &view, &frame).unwrap();
        }
        first_pixel(gpu, &texture)
    }

    fn close(got: [u8; 4], wanted: [u8; 4]) {
        assert!(got.iter().zip(wanted).all(|(a, b)| a.abs_diff(b) <= 2), "{got:?} is not {wanted:?}");
    }

    #[test]
    fn a_shader_sees_the_time_the_size_and_the_sound() {
        let Some(gpu) = gpu() else { return };
        let fs = "@fragment fn fs() -> @location(0) vec4f {
            return vec4f(frame.dt, frame.width / 8.0, audio.spectrum[511], audio.wave[0]);
        }";
        close(draw(&gpu, scene(serde_json::json!({"passes":[{"shader":shader(fs)}]})), 1, &[]), [128, 128, 64, 128]);
    }

    #[test]
    fn a_shader_reads_the_plugins_bytes() {
        let Some(gpu) = gpu() else { return };
        let fs = "@fragment fn fs() -> @location(0) vec4f { return vec4f(data[1], 0.0, 0.0, 1.0); }";
        let s = scene(serde_json::json!({"data_bytes":8,"passes":[{"shader":shader(fs)}]}));
        let bytes: Vec<u8> = [0.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        close(draw(&gpu, s, 1, &bytes), [255, 0, 0, 255]);
    }

    #[test]
    fn a_shader_reading_past_its_data_stays_inside_the_buffer() {
        let Some(gpu) = gpu() else { return };
        let fs = "@fragment fn fs() -> @location(0) vec4f { return vec4f(data[1000000], 0.0, 0.0, 1.0); }";
        let s = scene(serde_json::json!({"data_bytes":4,"passes":[{"shader":shader(fs)}]}));
        // WGSL allows either the last element or zero there; anything else came from outside.
        let red = draw(&gpu, s, 1, &0.5f32.to_le_bytes())[0];
        assert!(red == 0 || red.abs_diff(128) <= 2, "{red}");
    }

    /// Each frame adds a quarter to what the target held a frame ago.
    fn feedback() -> Scene {
        let grow = "@group(0) @binding(4) var last: texture_2d<f32>;
            @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
                return textureSample(last, smooth_, p.xy / vec2f(frame.width, frame.height)) + vec4f(0.25);
            }";
        let show = "@group(0) @binding(4) var now: texture_2d<f32>;
            @fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
                return vec4f(textureSample(now, smooth_, p.xy / vec2f(frame.width, frame.height)).rgb, 1.0);
            }";
        scene(serde_json::json!({
            "targets":[{"name":"t"}],
            "passes":[{"shader":shader(grow),"target":"t","inputs":["t"]},{"shader":shader(show),"inputs":["t"]}]
        }))
    }

    #[test]
    fn a_pass_reads_its_own_last_frame_and_a_later_pass_this_frame() {
        let Some(gpu) = gpu() else { return };
        close(draw(&gpu, feedback(), 1, &[]), [64, 64, 64, 255]);
        close(draw(&gpu, feedback(), 3, &[]), [191, 191, 191, 255]);
    }

    #[test]
    fn broken_shaders_are_refused_with_a_reason() {
        let Some(gpu) = gpu() else { return };
        let no_entry = "@fragment fn wrong() -> @location(0) vec4f { return vec4f(1.0); }";
        let foreign_binding = "@group(0) @binding(9) var nothing: texture_2d<f32>;
            @fragment fn fs() -> @location(0) vec4f { return textureLoad(nothing, vec2i(0), 0); }";
        for bad in ["not wgsl at all".to_string(), shader(no_entry), shader(foreign_binding)] {
            let s = scene(serde_json::json!({"passes":[{"shader":bad}]}));
            let Err(e) = Renderer::new(&gpu, s, FORMAT, (SIZE, SIZE)) else { panic!("accepted") };
            assert!(e.starts_with("a shader was refused"), "{e}");
        }
    }

    #[test]
    fn data_of_the_wrong_size_stops_the_frame() {
        let Some(gpu) = gpu() else { return };
        let fs = "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }";
        let mut r =
            Renderer::new(&gpu, scene(serde_json::json!({"data_bytes":8,"passes":[{"shader":shader(fs)}]})), FORMAT, (SIZE, SIZE)).unwrap();
        let texture = output(&gpu);
        let frame = Frame { time: 0.0, dt: 0.0, wave: &WAVE, spectrum: &SPECTRUM, data: &[0; 4] };
        assert!(r.draw(&gpu, &texture.create_view(&Default::default()), &frame).unwrap_err().contains("gave 4 bytes"));
    }

    #[test]
    fn a_resize_keeps_drawing() {
        let Some(gpu) = gpu() else { return };
        let mut r = Renderer::new(&gpu, feedback(), FORMAT, (SIZE, SIZE)).unwrap();
        let texture = output(&gpu);
        let view = texture.create_view(&Default::default());
        let frame = Frame { time: 0.0, dt: 0.0, wave: &WAVE, spectrum: &SPECTRUM, data: &[] };
        r.draw(&gpu, &view, &frame).unwrap();
        r.resize(&gpu, (2, 2));
        r.resize(&gpu, (SIZE, SIZE));
        r.draw(&gpu, &view, &frame).unwrap();
        close(first_pixel(&gpu, &texture), [64, 64, 64, 255]);
    }
}
