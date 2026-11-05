//! The pad over hidapi, and the thread that writes to it: writes block (a Bluetooth report can
//! take milliseconds), so the game only hands over what the pad should do.
#[cfg(not(target_os = "ios"))]
use crate::{DEFAULT_LIGHTBAR, Output, PLAYER_1, Vibration, is_dualsense};
use crate::{TRIGGER_OFF, TriggerEffect};
#[cfg(not(target_os = "ios"))]
use hidapi::{BusType, HidApi, HidDevice, HidResult};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::mpsc::Sender;
#[cfg(not(target_os = "ios"))]
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;
#[cfg(not(target_os = "ios"))]
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Usb,
    Bluetooth,
}

/// An open DualSense.
#[cfg(not(target_os = "ios"))]
pub struct DualSense {
    device: HidDevice,
    transport: Transport,
    vibration: Vibration,
    /// The Bluetooth report's sequence number.
    seq: u8,
}

/// Whether a pad on Bluetooth may be written to. Any output report, and even reading a feature
/// report, switches it to its full input report (0x31) until it is turned off (SDL_hints.h,
/// `SDL_HINT_JOYSTICK_ENHANCED_REPORTS`; SDL_hidapi_ps5.c). Linux's `hid-playstation` has
/// already switched it and reads that report, but gilrs on macOS and Windows parses the simple
/// report 0x01 and would stop seeing the pad: there it waits for `HORIZON_DUALSENSE_BT=1`.
pub fn bluetooth_allowed() -> bool {
    cfg!(target_os = "linux") || std::env::var_os("HORIZON_DUALSENSE_BT").is_some_and(|v| v == "1")
}

#[cfg(not(target_os = "ios"))]
impl DualSense {
    /// The first DualSense hidapi lists, USB before Bluetooth (as SDL); `None` when there is
    /// none, an error when it can't be opened (Linux without access to its hidraw node).
    pub fn open(api: &HidApi) -> Option<HidResult<Self>> {
        static SKIPPED: AtomicBool = AtomicBool::new(false);
        let bluetooth = |d: &&hidapi::DeviceInfo| matches!(d.bus_type(), BusType::Bluetooth);
        let pads = api
            .device_list()
            .filter(|d| is_dualsense(d.vendor_id(), d.product_id()));
        let (wireless, wired): (Vec<_>, Vec<_>) = pads.partition(bluetooth);
        let info = match wired.first() {
            Some(info) => info,
            None if bluetooth_allowed() => wireless.first()?,
            None => {
                if !wireless.is_empty() && !SKIPPED.swap(true, Ordering::Relaxed) {
                    log::info!("Horizon: no DualSense output over Bluetooth on this system");
                }
                return None;
            }
        };
        let transport = match info.bus_type() {
            BusType::Bluetooth => Transport::Bluetooth,
            _ => Transport::Usb,
        };
        Some(info.open_device(api).map(|device| {
            let mut firmware = [0; 64];
            firmware[0] = 0x20;
            let version = match device.get_feature_report(&mut firmware) {
                Ok(n) if n >= 46 => Some(u16::from_le_bytes([firmware[44], firmware[45]])),
                _ => None,
            };
            let vibration = Vibration::of(info.product_id(), version);
            DualSense {
                device,
                transport,
                vibration,
                seq: 0,
            }
        }))
    }

    pub fn transport(&self) -> Transport {
        self.transport
    }

    pub fn write(&mut self, out: &Output) -> HidResult<()> {
        match self.transport {
            Transport::Usb => self.device.write(&out.usb(self.vibration))?,
            Transport::Bluetooth => {
                let report = out.bluetooth(self.vibration, self.seq);
                self.seq = (self.seq + 1) & 0x0f;
                self.device.write(&report)?
            }
        };
        Ok(())
    }

    /// A report as it is (the Bluetooth haptics' 0x32).
    pub fn write_raw(&self, report: &[u8]) -> HidResult<()> {
        self.device.write(report).map(|_| ())
    }

    /// An input report (USB 0x01, Bluetooth 0x31), waiting up to `millis` (-1: forever).
    pub fn read(&self, buf: &mut [u8], millis: i32) -> HidResult<usize> {
        self.device.read_timeout(buf, millis)
    }
}

/// What the pad should show while nothing else is asked: the triggers' effects (left, right),
/// the lightbar and the player LEDs (`None`: the pad's own).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub triggers: [TriggerEffect; 2],
    pub lightbar: Option<[u8; 3]>,
    pub player_leds: Option<u8>,
}

impl Default for State {
    fn default() -> Self {
        State {
            triggers: [TRIGGER_OFF; 2],
            lightbar: None,
            player_leds: None,
        }
    }
}

/// A vibration: the motors' levels (left, right), each for a while, in order.
pub type Rumble = Vec<([u8; 2], Duration)>;

// On iOS no thread reads the queue (the Link never opens); the variants stay so the
// Link's API is the same everywhere.
#[cfg_attr(target_os = "ios", allow(dead_code))]
enum Message {
    State(State),
    Rumble(Rumble),
    /// HD haptics over Bluetooth: a vibration's steps (heavy, light, length).
    Haptics(Vec<(f32, f32, Duration)>),
    /// HD haptics over Bluetooth: the water ambience's level.
    Water(f32),
    /// HD haptics over Bluetooth: the game's sounds to feel.
    Feed(Option<crate::synth::Feed>),
    Scan,
}

/// The output thread. Dropping it frees the triggers and stops the motors before it ends; on
/// iOS there is none (no hidapi backend) and the Link stays closed.
pub struct Link {
    tx: Option<Sender<Message>>,
    /// 0: no pad open, else 1 + the transport.
    open: Arc<AtomicU8>,
    /// The water level last sent, as f32 bits.
    water: AtomicU32,
    /// Whether a sound feed was sent.
    feeding: AtomicBool,
    thread: Option<JoinHandle<()>>,
}

impl Link {
    /// Start the thread and look for a DualSense.
    #[cfg(not(target_os = "ios"))]
    pub fn start() -> Link {
        let (tx, rx) = mpsc::channel();
        let open = Arc::new(AtomicU8::new(0));
        let shown = open.clone();
        let thread = std::thread::Builder::new()
            .name("dualsense".into())
            .spawn(move || Writer::new(shown).run(rx))
            .map_err(|e| log::warn!("Horizon: DualSense output: {e}"))
            .ok();
        Link {
            tx: Some(tx),
            open,
            water: AtomicU32::new(0),
            feeding: AtomicBool::new(false),
            thread,
        }
    }

    /// No output on iOS: hidapi has no backend there. The Link never opens.
    #[cfg(target_os = "ios")]
    pub fn start() -> Link {
        Link {
            tx: None,
            open: Arc::new(AtomicU8::new(0)),
            water: AtomicU32::new(0),
            feeding: AtomicBool::new(false),
            thread: None,
        }
    }

    /// The pad's transport while one is open and taking reports.
    pub fn open(&self) -> Option<Transport> {
        match self.open.load(Ordering::Relaxed) {
            1 => Some(Transport::Usb),
            2 => Some(Transport::Bluetooth),
            _ => None,
        }
    }

    /// Play a vibration through the actuators' HD haptics over Bluetooth, replacing the one
    /// playing: (heavy, light, length) steps.
    pub fn haptics(&self, steps: Vec<(f32, f32, Duration)>) {
        self.send(Message::Haptics(steps));
    }

    /// The HD haptics' water ambience over Bluetooth, 0..1; sent only when it changes.
    pub fn set_water(&self, level: f32) {
        let bits = level.clamp(0.0, 1.0).to_bits();
        if self.water.swap(bits, Ordering::Relaxed) != bits {
            self.send(Message::Water(level));
        }
    }

    /// Feel the game's sounds through the HD haptics over Bluetooth (None: no more). The
    /// thread wakes to read the feed every few milliseconds while it is set.
    pub fn set_feed(&self, feed: Option<crate::synth::Feed>) {
        if self.feeding.swap(feed.is_some(), Ordering::Relaxed) != feed.is_some() {
            self.send(Message::Feed(feed));
        }
    }

    /// Whether a sound feed is set.
    pub fn is_feeding(&self) -> bool {
        self.feeding.load(Ordering::Relaxed)
    }

    /// Look for the pad again (a controller connected).
    pub fn scan(&self) {
        self.send(Message::Scan);
    }

    /// What the pad shows from now on; sent again whenever it reconnects.
    pub fn set(&self, state: State) {
        self.send(Message::State(state));
    }

    /// Play a vibration, replacing the one playing.
    pub fn rumble(&self, rumble: Rumble) {
        self.send(Message::Rumble(rumble));
    }

    fn send(&self, message: Message) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(message);
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        // Closing the channel ends the thread, which releases the pad first.
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Looks for the pad this many more times, a second apart, after a scan finds nothing (its
/// hidraw node may appear after the controller does).
#[cfg(not(target_os = "ios"))]
const SCAN_TRIES: u32 = 5;

#[cfg(not(target_os = "ios"))]
struct Writer {
    api: Option<HidApi>,
    pad: Option<Box<dyn Handle>>,
    open: Arc<AtomicU8>,
    state: State,
    /// What the pad was last sent; `None` after (re)opening.
    sent: Option<(State, [u8; 2])>,
    /// Whether the lights may be set: over Bluetooth, not before the pad's connection
    /// animation is over, or it paints over them.
    lights: Lights,
    rumble: Rumble,
    /// When the vibration's current step ends.
    step_end: Option<Instant>,
    /// HD haptics over Bluetooth: what plays, the reports' counter, and when the next is due.
    voice: crate::synth::Voice,
    haptics_counter: u8,
    next_haptics: Option<Instant>,
    /// A sound feed is set: the thread polls it while the pad is on Bluetooth.
    feeding: bool,
    /// When the PCM was last handed back the actuators (see `stream_haptics`).
    reclaimed: Option<Instant>,
    scans_left: u32,
    next_scan: Instant,
}

#[cfg(not(target_os = "ios"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lights {
    /// Over Bluetooth, until the pad's clock says its animation is over (or this long after
    /// opening, when it sends no clock).
    Waiting(Instant),
    /// They may be set, after the reset report that hands them to the host.
    Ready,
    Reset,
}

/// The pad's sensor clock once its Bluetooth connection animation is over (SDL_hidapi_ps5.c's
/// `HIDAPI_DriverPS5_CheckPendingLEDReset`).
#[cfg(not(target_os = "ios"))]
const ANIMATION_OVER: u32 = 10_200_000;

/// The sensor clock of a Bluetooth input report 0x31: its state block starts at byte 2 and
/// holds the clock at 27..31, little-endian (SDL's `PS5StatePacketCommon_t`, the kernel's
/// `dualsense_input_report`).
#[cfg(not(target_os = "ios"))]
fn bluetooth_clock(report: &[u8]) -> Option<u32> {
    match report {
        [0x31, _, state @ ..] if state.len() >= 31 => {
            Some(u32::from_le_bytes(state[27..31].try_into().ok()?))
        }
        _ => None,
    }
}

/// What the writer needs from an open pad. It is a trait so the state machine can be tested
/// without hardware: hidapi's Windows back end can crash inside `hid_write` once the device is
/// gone, so every failure has to drop the handle.
#[cfg(not(target_os = "ios"))]
trait Handle: Send {
    fn transport(&self) -> Transport;
    fn write_raw(&self, report: &[u8]) -> HidResult<()>;
    fn write(&mut self, out: &Output) -> HidResult<()>;
    fn read(&self, buf: &mut [u8], millis: i32) -> HidResult<usize>;
}

#[cfg(not(target_os = "ios"))]
impl Handle for DualSense {
    fn transport(&self) -> Transport {
        DualSense::transport(self)
    }
    fn write_raw(&self, report: &[u8]) -> HidResult<()> {
        DualSense::write_raw(self, report)
    }
    fn write(&mut self, out: &Output) -> HidResult<()> {
        DualSense::write(self, out)
    }

    fn read(&self, buf: &mut [u8], millis: i32) -> HidResult<usize> {
        DualSense::read(self, buf, millis)
    }
}

#[cfg(not(target_os = "ios"))]
impl Writer {
    fn new(open: Arc<AtomicU8>) -> Self {
        Writer {
            api: None,
            pad: None,
            open,
            state: State::default(),
            sent: None,
            lights: Lights::Ready,
            rumble: Vec::new(),
            step_end: None,
            voice: crate::synth::Voice::new(crate::synth::BT_HAPTICS_RATE),
            haptics_counter: 0,
            next_haptics: None,
            feeding: false,
            reclaimed: None,
            scans_left: 1,
            next_scan: Instant::now(),
        }
    }

    fn run(mut self, rx: mpsc::Receiver<Message>) {
        loop {
            let now = Instant::now();
            if self.pad.is_none() && self.scans_left > 0 && now >= self.next_scan {
                self.scan();
            }
            let waiting = matches!(self.lights, Lights::Waiting(_)) && self.pad.is_some();
            let wake = [
                self.step_end,
                (self.pad.is_none() && self.scans_left > 0).then_some(self.next_scan),
                waiting.then(|| now + Duration::from_millis(250)),
                self.next_haptics,
                // The feed's sounds come any time: look every 5 ms.
                (self.feeding && self.next_haptics.is_none() && self.pad.is_some())
                    .then(|| now + Duration::from_millis(5)),
            ]
            .into_iter()
            .flatten()
            .min();
            let message = match wake {
                Some(at) => rx.recv_timeout(at.saturating_duration_since(now)),
                None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };
            match message {
                Ok(Message::State(state)) => self.state = state,
                Ok(Message::Rumble(rumble)) => {
                    self.rumble = rumble;
                    self.rumble.reverse();
                    self.step_end = None;
                }
                Ok(Message::Haptics(steps)) => self.voice.play(&steps),
                Ok(Message::Water(level)) => self.voice.set_water(level),
                Ok(Message::Feed(feed)) => {
                    self.feeding = feed.is_some();
                    self.voice.set_feed(feed);
                }
                Ok(Message::Scan) => {
                    // A pad that was unplugged while idle fails to read.
                    if self.pad.is_some() {
                        self.probe();
                    }
                    if self.pad.is_none() {
                        self.scans_left = SCAN_TRIES;
                        self.next_scan = Instant::now();
                    }
                    continue;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.watch_animation();
            self.advance_rumble();
            self.flush();
            self.stream_haptics();
        }
        if self.pad.is_some() && self.probe() {
            // The lights go back to a host's defaults, if they were changed.
            let release = Output {
                lightbar: self.state.lightbar.map(|_| DEFAULT_LIGHTBAR),
                player_leds: self.state.player_leds.map(|_| PLAYER_1),
                ..Output::release()
            };
            if let Some(pad) = self.pad.as_mut() {
                let _ = pad.write(&release);
            }
        }
    }

    fn scan(&mut self) {
        self.scans_left -= 1;
        self.next_scan = Instant::now() + Duration::from_secs(1);
        let api = match self.api.as_mut() {
            Some(api) => api.refresh_devices().map(|()| &*api),
            None => HidApi::new().map(|api| &*self.api.insert(api)),
        };
        let found = match api {
            Ok(api) => DualSense::open(api),
            Err(e) => {
                log::warn!("Horizon: DualSense output: {e}");
                None
            }
        };
        match found {
            Some(Ok(pad)) => {
                log::info!("DualSense output over {:?}", pad.transport());
                self.open
                    .store(1 + pad.transport() as u8, Ordering::Relaxed);
                self.lights = match pad.transport() {
                    Transport::Usb => Lights::Ready,
                    Transport::Bluetooth => {
                        Lights::Waiting(Instant::now() + Duration::from_secs(1))
                    }
                };
                self.pad = Some(Box::new(pad));
                self.sent = None;
                self.scans_left = 0;
                self.flush();
            }
            Some(Err(e)) => {
                // No access (Linux without a udev rule): vibration stays with gilrs.
                log::info!("Horizon: the DualSense can't be opened for output: {e}");
                self.scans_left = 0;
            }
            None => {}
        }
    }

    /// Over Bluetooth, read the pad's clock until its connection animation is over.
    fn watch_animation(&mut self) {
        let (Lights::Waiting(deadline), Some(pad)) = (self.lights, self.pad.as_ref()) else {
            return;
        };
        let mut buf = [0; 80];
        let mut clock = None;
        let mut failed = None;
        loop {
            match pad.read(&mut buf, 0) {
                Ok(0) => break,
                Ok(n) => clock = bluetooth_clock(&buf[..n]).or(clock),
                Err(e) => {
                    failed = Some(e.to_string());
                    break;
                }
            }
        }
        if let Some(why) = failed {
            self.lost(&why);
            return;
        }
        let over = match clock {
            Some(clock) => clock >= ANIMATION_OVER,
            // No full report to read the clock from: as SDL, take it as over.
            None => Instant::now() >= deadline,
        };
        if over {
            self.lights = Lights::Ready;
        }
    }

    /// Over Bluetooth, the HD haptics' reports while anything plays, one every 32 frames of
    /// 3 kHz; behind by more than a few, it starts again from now rather than catching up.
    fn stream_haptics(&mut self) {
        let bluetooth = self
            .pad
            .as_ref()
            .is_some_and(|p| p.transport() == Transport::Bluetooth);
        if !bluetooth || !self.voice.active() {
            self.next_haptics = None;
            return;
        }
        let period = Duration::from_secs_f64(
            crate::synth::BT_HAPTICS_FRAMES as f64 / f64::from(crate::synth::BT_HAPTICS_RATE),
        );
        // Again every 5 s while it plays: the kernel's rumble can take them back any time.
        let reclaim = self
            .reclaimed
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(5));
        if self.next_haptics.is_none() || reclaim {
            self.reclaimed = Some(Instant::now());
            // A report with the haptics-select flag (the kernel's rumble, another host's) leaves
            // the actuators to the motors' emulation, and the PCM unheard, until a report
            // without it: one with no flags at all changes nothing else.
            let Some(pad) = self.pad.as_mut() else {
                return;
            };
            if let Err(e) = pad.write(&Output::default()) {
                return self.lost(&e.to_string());
            }
        }
        let now = Instant::now();
        let mut due = match self.next_haptics {
            Some(due) if now.saturating_duration_since(due) < period * 4 => due,
            _ => now,
        };
        while due <= now && self.voice.active() {
            let frames = std::array::from_fn(|_| self.voice.next());
            let report = crate::synth::bt_haptics_report(self.haptics_counter, &frames);
            self.haptics_counter = self.haptics_counter.wrapping_add(1);
            let Some(pad) = self.pad.as_ref() else {
                return;
            };
            if let Err(e) = pad.write_raw(&report) {
                return self.lost(&e.to_string());
            }
            due += period;
        }
        self.next_haptics = self.voice.active().then_some(due);
    }

    /// Move to the vibration's next step once this one is over.
    fn advance_rumble(&mut self) {
        let now = Instant::now();
        match self.step_end {
            Some(end) if end > now => return,
            Some(_) => {
                self.rumble.pop();
            }
            None => {}
        }
        self.step_end = self.rumble.last().map(|&(_, length)| now + length);
    }

    /// Send what changed; a failed write means the pad is gone.
    fn flush(&mut self) {
        if self.pad.is_none() {
            return;
        }
        let motors = self.rumble.last().map_or([0, 0], |(m, _)| *m);
        let mut state = self.state;
        if matches!(self.lights, Lights::Waiting(_)) {
            state.lightbar = None;
            state.player_leds = None;
        }
        let sent = self.sent;
        if sent == Some((state, motors)) {
            return;
        }
        // Every report carries the motors: one without them stops the rumble.
        let mut first = Output {
            rumble: motors,
            ..Output::default()
        };
        if sent.is_none() {
            first.full_power = true;
        }
        let lit = state.lightbar.is_some() || state.player_leds.is_some();
        if lit && self.lights == Lights::Ready {
            first.release_leds = true;
            self.lights = Lights::Reset;
        }
        // The reset goes alone, before any colour (hid-playstation.c).
        if (first.full_power || first.release_leds) && !self.write(&first) {
            return;
        }
        let mut out = Output {
            rumble: motors,
            ..Output::default()
        };
        let old = sent.map(|(s, _)| s);
        if old.is_none_or(|s| s.triggers != state.triggers) {
            out.left_trigger = Some(state.triggers[0]);
            out.right_trigger = Some(state.triggers[1]);
        }
        if old.is_none_or(|s| s.lightbar != state.lightbar) {
            out.lightbar = state.lightbar;
        }
        if old.is_none_or(|s| s.player_leds != state.player_leds) {
            out.player_leds = state.player_leds;
        }
        if self.write(&out) {
            self.sent = Some((state, motors));
        }
    }

    /// Whether the pad still answers. hidapi's Windows back end can crash inside `hid_write`
    /// once the device has gone (Bluetooth sleep, unplug), so a write reads first; a failed
    /// read drops the handle and the thread looks for the pad again.
    fn probe(&mut self) -> bool {
        let Some(pad) = self.pad.as_ref() else {
            return false;
        };
        let mut buf = [0; 80];
        let failed = match pad.read(&mut buf, 0) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        };
        match failed {
            Some(why) => {
                self.lost(&why);
                false
            }
            None => true,
        }
    }

    /// Write a report, dropping the pad on any hidapi error: a handle that failed is never
    /// written through again.
    fn write(&mut self, out: &Output) -> bool {
        if !self.probe() {
            return false;
        }
        let result = match self.pad.as_mut() {
            Some(pad) => pad.write(out),
            None => return false,
        };
        match result {
            Ok(()) => true,
            Err(e) => {
                self.lost(&e.to_string());
                false
            }
        }
    }

    /// Drop a handle that hidapi failed on, and look for the pad again: one that slept or was
    /// unplugged may come back. Nothing is written through the old handle.
    fn lost(&mut self, why: &str) {
        log::info!("DualSense output closed: {why}");
        self.pad = None;
        self.sent = None;
        self.lights = Lights::Ready;
        self.open.store(0, Ordering::Relaxed);
        self.scans_left = SCAN_TRIES;
        self.next_scan = Instant::now();
    }
}

#[cfg(all(test, not(target_os = "ios")))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A pad whose reads and writes can fail, so the writer's state machine runs without
    /// hardware (the crash itself is inside hidapi's C on Windows).
    struct FakePad {
        writes: Arc<AtomicUsize>,
        reads: Arc<AtomicUsize>,
        read_fails: bool,
        write_fails: bool,
    }

    impl Handle for FakePad {
        fn transport(&self) -> Transport {
            Transport::Usb
        }
        fn write_raw(&self, _report: &[u8]) -> HidResult<()> {
            if self.write_fails {
                Err(hidapi::HidError::HidApiError {
                    message: "gone".into(),
                })
            } else {
                Ok(())
            }
        }
        fn write(&mut self, _out: &Output) -> HidResult<()> {
            self.writes.fetch_add(1, Ordering::Relaxed);
            if self.write_fails {
                Err(hidapi::HidError::HidApiError {
                    message: "gone".into(),
                })
            } else {
                Ok(())
            }
        }

        fn read(&self, _buf: &mut [u8], _millis: i32) -> HidResult<usize> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            if self.read_fails {
                Err(hidapi::HidError::HidApiError {
                    message: "gone".into(),
                })
            } else {
                Ok(0)
            }
        }
    }

    fn writer(
        open: &Arc<AtomicU8>,
        read_fails: bool,
        write_fails: bool,
    ) -> (Writer, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let writes = Arc::new(AtomicUsize::new(0));
        let reads = Arc::new(AtomicUsize::new(0));
        let mut writer = Writer::new(open.clone());
        writer.pad = Some(Box::new(FakePad {
            writes: writes.clone(),
            reads: reads.clone(),
            read_fails,
            write_fails,
        }));
        (writer, writes, reads)
    }

    #[test]
    fn a_failed_read_drops_the_pad_without_writing() {
        let open = Arc::new(AtomicU8::new(1));
        let (mut writer, writes, reads) = writer(&open, true, false);
        writer.flush();
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert_eq!(writes.load(Ordering::Relaxed), 0);
        assert!(writer.pad.is_none());
        assert_eq!(open.load(Ordering::Relaxed), 0);
        assert_eq!(writer.scans_left, SCAN_TRIES);
    }

    #[test]
    fn a_failed_write_drops_the_pad_and_schedules_a_rescan() {
        let open = Arc::new(AtomicU8::new(1));
        let (mut writer, writes, _) = writer(&open, false, true);
        writer.flush();
        assert_eq!(writes.load(Ordering::Relaxed), 1);
        assert!(writer.pad.is_none());
        assert_eq!(open.load(Ordering::Relaxed), 0);
        assert!(writer.sent.is_none());
        assert_eq!(writer.scans_left, SCAN_TRIES);
        // The handle that failed is not written through again.
        writer.flush();
        assert_eq!(writes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_live_pad_stays_open() {
        let open = Arc::new(AtomicU8::new(1));
        let (mut writer, writes, _) = writer(&open, false, false);
        writer.flush();
        assert_eq!(writes.load(Ordering::Relaxed), 2);
        assert!(writer.pad.is_some());
        assert!(writer.sent.is_some());
        assert_eq!(open.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn bluetooth_clock_offset() {
        let mut report = [0; 78];
        report[0] = 0x31;
        report[2 + 27..2 + 31].copy_from_slice(&ANIMATION_OVER.to_le_bytes());
        assert_eq!(bluetooth_clock(&report), Some(10_200_000));
        // The simple report, or a USB one, has no clock here.
        report[0] = 0x01;
        assert_eq!(bluetooth_clock(&report), None);
        assert_eq!(bluetooth_clock(&[0x31, 0, 0]), None);
    }
}
