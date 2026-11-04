//! Vibrations as waveforms for the pad's voice-coil actuators, at any rate: 48 kHz through its
//! USB audio device, 3 kHz in its Bluetooth haptics reports.
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::time::Duration;

/// The frequencies the heavy and light parts of a vibration play at.
pub const HEAVY_HZ: f32 = 80.0;
pub const LIGHT_HZ: f32 = 200.0;
/// Fades between steps, so a level change doesn't click.
const RAMP: f32 = 0.004;

/// The samples of a vibration: the heavy and light tones at each step's levels, with short
/// ramps between steps and to silence at the end.
pub fn waveform(steps: &[(f32, f32, Duration)], rate: u32) -> Vec<f32> {
    let ramp = ((RAMP * rate as f32) as usize).max(1);
    let mut out = Vec::new();
    let mut level = [0.0f32; 2];
    let mut t = 0usize;
    for &(heavy, light, length) in steps.iter().chain(&[(0.0, 0.0, Duration::ZERO)]) {
        let target = [heavy.clamp(0.0, 1.0), light.clamp(0.0, 1.0)];
        let from = level;
        let n = (length.as_secs_f32() * rate as f32) as usize;
        for i in 0..n.max(ramp) {
            let k = (i as f32 / ramp as f32).min(1.0);
            level = [0, 1].map(|j| from[j] + (target[j] - from[j]) * k);
            let time = t as f32 / rate as f32;
            let v =
                level[0] * (TAU * HEAVY_HZ * time).sin() + level[1] * (TAU * LIGHT_HZ * time).sin();
            out.push(v.clamp(-1.0, 1.0));
            t += 1;
        }
    }
    out
}

/// Under water: a slow, heavy swell of the current and bubbles rising past the hands, each a
/// short falling tone on one side.
pub struct Ambience {
    rate: f32,
    seed: u32,
    /// The level, eased toward the asked one so going under or out fades.
    level: f32,
    /// Low-passed noise: the swell.
    swell: f32,
    lfo: f32,
    /// Bubbles: tone (Hz), phase, envelope, side.
    bubble: Option<(f32, f32, f32, usize)>,
}

impl Ambience {
    pub fn new(rate: u32) -> Self {
        Ambience {
            rate: rate as f32,
            seed: 0x9e37_79b9,
            level: 0.0,
            swell: 0.0,
            lfo: 0.0,
            bubble: None,
        }
    }

    fn random(&mut self) -> f32 {
        // xorshift32
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        self.seed as f32 / u32::MAX as f32
    }

    /// The next frame (left, right actuator).
    pub fn next(&mut self, target: f32) -> [f32; 2] {
        // About a fifth of a second to fade.
        self.level += (target - self.level) * (5.0 / self.rate);
        if self.level < 1e-4 {
            self.bubble = None;
            return [0.0; 2];
        }
        let noise = self.random() * 2.0 - 1.0;
        // A one-pole low-pass around 30 Hz.
        self.swell += (noise - self.swell) * (TAU * 30.0 / self.rate);
        self.lfo = (self.lfo + 0.25 / self.rate).fract();
        let wave = 0.6 + 0.4 * (TAU * self.lfo).sin();
        let base = self.swell * 5.0 * wave;
        let mut out = [base, base];
        if self.bubble.is_none() && self.random() < 2.5 / self.rate {
            let tone = 140.0 + self.random() * 160.0;
            let side = usize::from(self.random() < 0.5);
            self.bubble = Some((tone, 0.0, 1.0, side));
        }
        if let Some((tone, phase, envelope, side)) = self.bubble.as_mut() {
            out[*side] += (TAU * *phase).sin() * *envelope * 0.35;
            // The tone rises as the bubble shrinks; it dies in about 60 ms.
            *tone *= 1.0 + 2.0 / self.rate;
            *phase = (*phase + *tone / self.rate).fract();
            *envelope *= 1.0 - 16.0 / self.rate;
            if *envelope < 0.01 {
                self.bubble = None;
            }
        }
        out.map(|v| v * self.level)
    }
}

/// What the actuators play: the queued effect over the ambience.
pub struct Voice {
    rate: u32,
    queue: VecDeque<f32>,
    ambience: Ambience,
    water: f32,
}

impl Voice {
    pub fn new(rate: u32) -> Self {
        Voice {
            rate,
            queue: VecDeque::new(),
            ambience: Ambience::new(rate),
            water: 0.0,
        }
    }

    /// Play a vibration, replacing the one playing.
    pub fn play(&mut self, steps: &[(f32, f32, Duration)]) {
        self.queue.clear();
        self.queue.extend(waveform(steps, self.rate));
    }

    pub fn set_water(&mut self, level: f32) {
        self.water = level.clamp(0.0, 1.0);
    }

    /// Whether anything is left to play (an effect, the ambience or its fade).
    pub fn active(&self) -> bool {
        !self.queue.is_empty() || self.water > 0.0 || self.ambience.level >= 1e-4
    }

    /// The next frame (left, right actuator).
    pub fn next(&mut self) -> [f32; 2] {
        let effect = self.queue.pop_front().unwrap_or_default();
        let [a, b] = self.ambience.next(self.water);
        [(effect + a).clamp(-1.0, 1.0), (effect + b).clamp(-1.0, 1.0)]
    }
}

/// The Bluetooth haptics report 0x32: 32 stereo frames of signed 8-bit PCM at 3 kHz, sent every
/// 32/3000 s. Its layout is egormanga's research (SAxense, https://apps.sdore.me/SAxense, MPL-2.0):
/// after the id and a sequence byte, packet 0x11 (7 bytes, the last a counter) and packet 0x12
/// (the 64 samples), each led by its id with bit 7 set and its length; the CRC-32 of the seed
/// byte 0xa2 and the first 138 bytes ends it.
pub const BT_HAPTICS_RATE: u32 = 3000;
pub const BT_HAPTICS_FRAMES: usize = 32;
pub const BT_HAPTICS_LEN: usize = 142;

pub fn bt_haptics_report(
    counter: u8,
    frames: &[[f32; 2]; BT_HAPTICS_FRAMES],
) -> [u8; BT_HAPTICS_LEN] {
    let mut r = [0; BT_HAPTICS_LEN];
    r[0] = 0x32;
    r[2] = 0x80 | 0x11;
    r[3] = 7;
    r[4..11].copy_from_slice(&[0xfe, 0, 0, 0, 0, 0xff, counter]);
    r[11] = 0x80 | 0x12;
    r[12] = 64;
    for (i, frame) in frames.iter().enumerate() {
        for (j, v) in frame.iter().enumerate() {
            r[13 + i * 2 + j] = ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8;
        }
    }
    let crc = crate::bt_crc(&r[..BT_HAPTICS_LEN - 4]);
    r[BT_HAPTICS_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambience_fades_in_and_out() {
        let rate = 48_000;
        let mut a = Ambience::new(rate);
        assert_eq!(a.next(0.0), [0.0; 2]);
        let under: Vec<[f32; 2]> = (0..rate).map(|_| a.next(1.0)).collect();
        let felt = under[rate as usize / 2..]
            .iter()
            .fold(0.0f32, |m, v| m.max(v[0].abs()).max(v[1].abs()));
        assert!(felt > 0.05 && felt <= 1.0, "{felt}");
        for _ in 0..rate * 2 {
            a.next(0.0);
        }
        assert_eq!(a.next(0.0), [0.0; 2]);
    }

    #[test]
    fn voice_ends_after_its_effect() {
        let mut v = Voice::new(BT_HAPTICS_RATE);
        assert!(!v.active());
        v.play(&[(1.0, 0.0, Duration::from_millis(10))]);
        let mut n = 0;
        while v.active() {
            v.next();
            n += 1;
        }
        // 10 ms at 3 kHz, then the ramp down.
        assert_eq!(n, 30 + 12);
    }

    #[test]
    fn bt_report_layout() {
        // Silence, counter 0: the bytes SAxense sends first, its CRC computed independently
        // (zlib's crc32 over b"\xa2" + the report's first 138 bytes).
        let r = bt_haptics_report(0, &[[0.0; 2]; BT_HAPTICS_FRAMES]);
        assert_eq!(
            r[..13],
            [0x32, 0, 0x91, 7, 0xfe, 0, 0, 0, 0, 0xff, 0, 0x92, 64]
        );
        assert_eq!(r[138..], 0x26e2_0c96u32.to_le_bytes());
        let full = bt_haptics_report(1, &[[1.0, -1.0]; BT_HAPTICS_FRAMES]);
        assert_eq!(full[13..15], [127, 0x81]);
        assert_eq!(full[10], 1);
    }
}
