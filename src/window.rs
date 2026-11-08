//! The drawing loop of a visualizer window. It runs on a thread of its
//! own, so that a heavy shader or a stuck GPU cannot hold up the player.

use crate::renderer::{Frame, Gpu, Renderer, SPECTRUM_LEN, WAVE_LEN};
use crate::scene::Scene;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A frame slower than this counts against the plug-in.
const SLOW_FRAME: Duration = Duration::from_secs(2);
/// Bad frames in a row that end the visualizer.
const STRIKES_TO_STOP: u32 = 3;
const IDLE: Duration = Duration::from_millis(100);

/// What the player supplies: the plug-in's answers and the sound.
pub trait Feed: Send {
    fn scene(&mut self) -> Result<Vec<u8>, String>;
    /// The bytes for the shaders, given the frame's numbers in the layout
    /// documented for the plug-in's `frame` call.
    fn data(&mut self, frame: &[u8]) -> Result<Vec<u8>, String>;
    fn sound(&mut self, wave: &mut [f32; WAVE_LEN], spectrum: &mut [f32; SPECTRUM_LEN]);
}

/// How the window's thread and the player's talk.
pub struct Control {
    pub stop: AtomicBool,
    pub size: Mutex<(u32, u32)>,
}

impl Control {
    pub fn new(size: (u32, u32)) -> Arc<Control> {
        Arc::new(Control { stop: AtomicBool::new(false), size: Mutex::new(size) })
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

/// Draws until told to stop (`Ok`) or until the plug-in or the GPU fails
/// (`Err`, with the reason to show).
pub fn run<W>(window: Arc<W>, mut feed: Box<dyn Feed>, control: Arc<Control>) -> Result<(), String>
where
    W: HasWindowHandle + HasDisplayHandle + Debug + Send + Sync + 'static,
{
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone())));
    let surface = instance.create_surface(window).map_err(|e| format!("no surface to draw on: {e}"))?;
    let gpu = Gpu::new(&instance, Some(&surface))?;
    let scene = Scene::parse(&feed.scene()?)?;
    let data_bytes = scene.data_bytes as usize;
    let mut config = surface_config(&gpu, &surface, *control.size.lock())?;
    let mut renderer = Renderer::new(&gpu, scene, config.format, (config.width, config.height))?;
    surface.configure(&gpu.device, &config);
    let mut state = State::new(data_bytes);
    while !control.stopped() {
        let size = *control.size.lock();
        if size.0 == 0 || size.1 == 0 {
            std::thread::sleep(IDLE);
            continue;
        }
        if size != (config.width, config.height) {
            (config.width, config.height) = size;
            surface.configure(&gpu.device, &config);
            renderer.resize(&gpu, size);
        }
        state.advance(feed.as_mut(), size)?;
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                std::thread::sleep(IDLE);
                continue;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&gpu.device, &config);
                continue;
            }
            wgpu::CurrentSurfaceTexture::Validation => return Err("the window cannot be drawn on".into()),
        };
        let view = frame.texture.create_view(&Default::default());
        renderer.draw(&gpu, &view, &state.frame())?;
        gpu.queue.present(frame);
        state.finished()?;
    }
    Ok(())
}

fn surface_config(gpu: &Gpu, surface: &wgpu::Surface<'_>, (width, height): (u32, u32)) -> Result<wgpu::SurfaceConfiguration, String> {
    let mut config = surface.get_default_config(&gpu.adapter, width.max(1), height.max(1)).ok_or("the GPU cannot draw on this window")?;
    // Colours as the plug-in wrote them, without a conversion it did not ask for.
    let caps = surface.get_capabilities(&gpu.adapter);
    config.format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(config.format);
    config.present_mode = wgpu::PresentMode::AutoVsync;
    Ok(config)
}

/// One frame's inputs, and the running count of bad frames.
struct State {
    wave: Box<[f32; WAVE_LEN]>,
    spectrum: Box<[f32; SPECTRUM_LEN]>,
    data: Vec<u8>,
    data_bytes: usize,
    started: Instant,
    last: Instant,
    time: f32,
    dt: f32,
    strikes: u32,
}

impl State {
    fn new(data_bytes: usize) -> State {
        let now = Instant::now();
        State {
            wave: Box::new([0.0; WAVE_LEN]),
            spectrum: Box::new([0.0; SPECTRUM_LEN]),
            data: vec![0; data_bytes],
            data_bytes,
            started: now,
            last: now,
            time: 0.0,
            dt: 0.0,
            strikes: 0,
        }
    }

    fn advance(&mut self, feed: &mut dyn Feed, size: (u32, u32)) -> Result<(), String> {
        let now = Instant::now();
        (self.time, self.dt) = (now.duration_since(self.started).as_secs_f32(), now.duration_since(self.last).as_secs_f32());
        self.last = now;
        feed.sound(&mut self.wave, &mut self.spectrum);
        if self.data_bytes == 0 {
            return Ok(());
        }
        // A frame that gets no fresh bytes draws with the last ones.
        match feed.data(&self.input(size)).and_then(|d| self.checked(d)) {
            Ok(data) => {
                self.data = data;
                Ok(())
            }
            Err(e) => self.strike(e),
        }
    }

    fn checked(&self, data: Vec<u8>) -> Result<Vec<u8>, String> {
        match data.len() == self.data_bytes {
            true => Ok(data),
            false => Err(format!("the plug-in gave {} bytes of data, its scene says {}", data.len(), self.data_bytes)),
        }
    }

    fn input(&self, (width, height): (u32, u32)) -> Vec<u8> {
        let header = [self.time, self.dt, width as f32, height as f32];
        let numbers = header.iter().chain(self.wave.iter()).chain(self.spectrum.iter());
        numbers.flat_map(|v| v.to_le_bytes()).collect()
    }

    fn frame(&self) -> Frame<'_> {
        Frame { time: self.time, dt: self.dt, wave: &self.wave, spectrum: &self.spectrum, data: &self.data }
    }

    fn finished(&mut self) -> Result<(), String> {
        let took = self.last.elapsed();
        if took > SLOW_FRAME {
            return self.strike(format!("a frame took {took:?}"));
        }
        self.strikes = 0;
        Ok(())
    }

    fn strike(&mut self, why: String) -> Result<(), String> {
        self.strikes += 1;
        log::warn!("visualizer: {why}");
        match self.strikes >= STRIKES_TO_STOP {
            true => Err(why),
            false => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(Result<Vec<u8>, String>);

    impl Feed for Fixed {
        fn scene(&mut self) -> Result<Vec<u8>, String> {
            unreachable!()
        }
        fn data(&mut self, frame: &[u8]) -> Result<Vec<u8>, String> {
            assert_eq!(frame.len(), 16 + (WAVE_LEN + SPECTRUM_LEN) * 4);
            self.0.clone()
        }
        fn sound(&mut self, wave: &mut [f32; WAVE_LEN], _: &mut [f32; SPECTRUM_LEN]) {
            wave[0] = 0.5;
        }
    }

    #[test]
    fn fresh_bytes_are_kept_and_the_sound_is_read() {
        let mut s = State::new(4);
        s.advance(&mut Fixed(Ok(vec![1, 2, 3, 4])), (10, 10)).unwrap();
        assert_eq!((s.frame().data, s.frame().wave[0]), (&[1, 2, 3, 4][..], 0.5));
    }

    #[test]
    fn a_bad_frame_draws_the_last_bytes_and_three_in_a_row_stop_it() {
        let mut s = State::new(4);
        let mut good = Fixed(Ok(vec![1, 2, 3, 4]));
        let mut short = Fixed(Ok(vec![9]));
        let mut failing = Fixed(Err("timeout".into()));
        s.advance(&mut good, (1, 1)).unwrap();
        s.advance(&mut short, (1, 1)).unwrap();
        s.advance(&mut failing, (1, 1)).unwrap();
        assert_eq!(s.frame().data, [1, 2, 3, 4]);
        assert_eq!(s.advance(&mut failing, (1, 1)).unwrap_err(), "timeout");
    }

    #[test]
    fn a_good_frame_clears_the_strikes() {
        let mut s = State::new(4);
        let mut failing = Fixed(Err("x".into()));
        let mut good = Fixed(Ok(vec![0; 4]));
        for _ in 0..4 {
            s.advance(&mut failing, (1, 1)).unwrap();
            s.advance(&mut failing, (1, 1)).unwrap();
            s.advance(&mut good, (1, 1)).unwrap();
            s.finished().unwrap();
        }
    }
}
