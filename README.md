# popamp-visual

Draws visualizer plug-ins for [Popamp](https://github.com/popamp/popamp).
A plug-in says what to draw, as a scene of WGSL passes; this crate checks
the scene and makes every GPU call, in a thread of its own so that a slow
shader cannot hold up the music.

It is compiled in only with the player's `visualizers` feature, so a build
without it carries no GPU code:

```sh
cargo build --release --features plus
```

How to write such a plug-in (the scene, the bindings shaders see, the
limits) is in Popamp's
[docs/plugins.md](https://github.com/popamp/popamp/blob/main/docs/plugins.md).
[plugin-spectrum](https://github.com/popamp/plugin-spectrum) is a working
example.

## License

MIT or Apache-2.0, at your option.
