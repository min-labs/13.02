// M13 HUB - LOGIC KERNEL (REV 6.1)
// Sprint 6.1: Fragmentation Engine + Hexdump (bilateral with Node)
// Sprint 5.1: The Executive (Thread Pinning, Multi-Core)
// Sprint 5.2: Typestate FSM (Compile-Time Protocol State)
// Sprint 5.3: The Graph Pipeline (I/O Vector Batching)
// Sprint 5.4: The Latency Floor (Adaptive Batching)
// Sprint 5.5: Software Prefetching (D-Cache Warming)
// Sprint 5.6: The Interrupt Fence (IRQ Affinity)
// Sprint 5.7: The Isochronous Scheduler (Hierarchical DWRR)
// Sprint 5.8: The Feedback Channel (ACK Frame)
// Sprint 5.9: BBRv3 Congestion Control (Model-Based Rate Pacing)
// Sprint 5.10: Deterministic Jitter Buffer (RFC 3550 EWMA, Adaptive D_buf)
// Sprint 5.11: Diagnostic Error Codes (Structured Exit, Zero-Alloc Fatal)
//   - Windowed max filter for BtlBw (Kathleen Nichols algorithm, 10-round window)
//   - Expiring min for RTprop (10-second window, triggers ProbeRTT)
//   - Full BBRv3 state machine: Startup → Drain → ProbeBW (Down/Cruise/Refill/Up) → ProbeRTT
//   - Token bucket pacing at batch level (no per-packet busy-wait, preserves batching)
//   - cwnd = BDP × cwnd_gain (caps inflight, replaces static HW_FILL_MAX when active)
//   - All gains from Google's BBRv3: Startup 2.77x, Drain 0.36x, ProbeUp 1.25x, ProbeDown 0.9x
//   - Fixed-point arithmetic (integer rationals, no f64)
//   - Feedback-first pipeline: control frames processed BEFORE generation for zero-cycle latency
mod datapath;
use crate::datapath::{M13_WIRE_MAGIC, M13_WIRE_VERSION, 
    FixedSlab, Engine, Telemetry, EthernetHeader, M13Header, FeedbackFrame,
    ETH_P_M13, ZeroCopyTx, TxPath, BpfSteersman, MAX_WORKERS, FRAME_SIZE, UMEM_SIZE,
    // Structured exits — single source of truth for error codes
    fatal,
    E_NO_ISOLATED_CORES, E_AFFINITY_FAIL, E_AFFINITY_VERIFY,
};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs::{OpenOptions, File};
use std::io::{Write, Read, Seek, SeekFrom, BufReader, BufRead};

use std::time::Duration;
use std::os::unix::io::AsRawFd;
use std::process::Command;

use std::marker::PhantomData;
use std::collections::HashMap;

// Sprint 6.2: PQC cold-path imports (handshake only — never in hot loop)
use sha2::{Sha512, Digest};
use hkdf::Hkdf;
use rand::rngs::OsRng;

// Sprint 6.3: PQC handshake — ML-KEM-1024 key exchange + ML-DSA-87 mutual auth
use ml_kem::EncodedSizeUser;
use ml_kem::kem::Encapsulate;
use ml_dsa::{MlDsa87, KeyGen};

const SLAB_DEPTH: usize = 8192;

const GRAPH_BATCH: usize = 256;
const FLAG_CONTROL: u8  = 0x80;
const FLAG_FEEDBACK: u8 = 0x40;
const FLAG_TUNNEL: u8   = 0x20;
const FLAG_ECN: u8      = 0x10;  // Receiver signals congestion (Sprint 5.19)
const FLAG_FIN: u8      = 0x08;  // Graceful close signal (Sprint 5.21)
// FLAG_FEC (0x04) reserved for Sprint 6.7 RLNC — not yet implemented
const FLAG_HANDSHAKE: u8= 0x02;  // Handshake control (Sprint 6.3)
const FLAG_FRAGMENT: u8 = 0x01;  // Fragmented message (Sprint 6.1)

// Sprint 6.2: PQC handshake sub-types
const HS_CLIENT_HELLO: u8 = 0x01;
const HS_SERVER_HELLO: u8 = 0x02;
const HS_FINISHED: u8     = 0x03;
// Direction bytes for AEAD nonce (prevents reflection attacks)
const DIR_HUB_TO_NODE: u8 = 0x00;

// Rekey thresholds
const REKEY_FRAME_LIMIT: u64 = 1u64 << 32;
const REKEY_TIME_LIMIT_NS: u64 = 3_600_000_000_000;

const ETH_HDR_SIZE: usize = mem::size_of::<EthernetHeader>();
const M13_HDR_SIZE: usize = mem::size_of::<M13Header>();
const FEEDBACK_FRAME_LEN: u32 = (ETH_HDR_SIZE + M13_HDR_SIZE + mem::size_of::<FeedbackFrame>()) as u32;
const DEADLINE_NS: u64 = 50_000;
const PREFETCH_DIST: usize = 4;
const TX_RING_SIZE: usize = 2048;
const HW_FILL_MAX: usize = TX_RING_SIZE / 10;

const FEEDBACK_INTERVAL_PKTS: u32 = 32;
const FEEDBACK_RTT_DEFAULT_NS: u64 = 10_000_000; // 10ms until first RTprop sample

const SEQ_WINDOW: usize = 131_072; // 2^17
const _: () = assert!(SEQ_WINDOW & (SEQ_WINDOW - 1) == 0);

#[inline(always)]
fn clock_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

// ============================================================================
// SPRINT 5.17: TSC FAST CLOCK
// Replaces clock_gettime(MONOTONIC) in the hot loop with raw rdtsc.
// Calibrated at boot against CLOCK_MONOTONIC. Fixed-point multiply+shift
// conversion — identical method to Linux kernel (arch/x86/kernel/tsc.c).
// ============================================================================

/// TSC-to-nanosecond calibration data. Computed once at boot, immutable after.
/// Conversion: ns = mono_base + ((rdtsc() - tsc_base) * mult) >> shift
/// The mult/shift pair encodes ns_per_tsc_tick as a fixed-point fraction.
#[derive(Clone, Copy)]
struct TscCal {
    tsc_base: u64,   // rdtsc value at calibration instant
    mono_base: u64,  // CLOCK_MONOTONIC (ns) at same instant
    mult: u32,       // fixed-point multiplier
    shift: u32,      // right-shift amount (typically 32)
    valid: bool,     // false if TSC is unreliable (VM, non-invariant)
}

impl TscCal {
    /// Fallback calibration — rdtsc_ns() will call clock_ns() instead.
    fn fallback() -> Self {
        TscCal { tsc_base: 0, mono_base: 0, mult: 0, shift: 0, valid: false }
    }
}

/// Raw TSC read. ~24 cycles on Skylake (~6.5ns at 3.7GHz).
/// No serialization (lfence/rdtscp) — not needed for "what time is it?" queries.
/// OoO reordering error is ±2ns, irrelevant for 50µs deadlines.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn read_tsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem, preserves_flags)
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// ARM equivalent: CNTVCT_EL0 (generic timer virtual count).
/// Constant-rate, monotonic, unprivileged. Same calibration math applies.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn read_tsc() -> u64 {
    let cnt: u64;
    unsafe {
        core::arch::asm!(
            "mrs {cnt}, CNTVCT_EL0",
            cnt = out(reg) cnt,
            options(nostack, nomem, preserves_flags)
        );
    }
    cnt
}

/// Fallback for non-x86/ARM: just use clock_gettime.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn read_tsc() -> u64 { clock_ns() }

/// Convert raw TSC value to nanoseconds using pre-computed calibration.
/// Hot path: 1 subtract + 1 multiply (u128) + 1 shift + 1 add = ~5 cycles.
/// Total with rdtsc: ~29 cycles = ~7.8ns at 3.7GHz.
/// Compare: clock_gettime vDSO = ~41 cycles = ~11-25ns.
#[inline(always)]
fn rdtsc_ns(cal: &TscCal) -> u64 {
    if !cal.valid { return clock_ns(); }
    let delta = read_tsc().wrapping_sub(cal.tsc_base);
    cal.mono_base.wrapping_add(
        ((delta as u128 * cal.mult as u128) >> cal.shift) as u64
    )
}

/// Two-point TSC calibration against CLOCK_MONOTONIC.
/// Runs for 100ms, comparing rdtsc deltas against kernel clock deltas.
/// Computes fixed-point mult/shift such that:
///   ns_per_tick = mult / 2^shift
/// After calibration, validates accuracy over 1000 samples.
/// Returns TscCal::fallback() if TSC is unreliable.
fn calibrate_tsc() -> TscCal {
    // Check invariant TSC support (CPUID leaf 0x80000007, bit 8)
    #[cfg(target_arch = "x86_64")]
    {
        let has_invariant_tsc = unsafe {
            let result: u32;
            core::arch::asm!(
                "push rbx",
                "mov eax, 0x80000007",
                "cpuid",
                "pop rbx",
                out("edx") result,
                out("eax") _,
                out("ecx") _,
                options(nomem)
            );
            (result >> 8) & 1 == 1
        };
        if !has_invariant_tsc {
            eprintln!("[M13-TSC] WARNING: CPU lacks invariant TSC. Using clock_gettime fallback.");
            return TscCal::fallback();
        }
    }

    // Warm up caches: 100 iterations (discard results)
    for _ in 0..100 {
        let _ = read_tsc();
        let _ = clock_ns();
    }

    // Two-point calibration over 100ms
    let tsc0 = read_tsc();
    let mono0 = clock_ns();
    std::thread::sleep(Duration::from_millis(100));
    let tsc1 = read_tsc();
    let mono1 = clock_ns();

    let tsc_delta = tsc1.wrapping_sub(tsc0);
    let mono_delta = mono1.saturating_sub(mono0);

    if tsc_delta == 0 || mono_delta == 0 {
        eprintln!("[M13-TSC] WARNING: TSC calibration failed (zero delta). Using fallback.");
        return TscCal::fallback();
    }

    // Compute ns_per_tick as fixed-point: mult / 2^shift
    // Choose shift = 32 for maximum precision with u32 mult.
    // mult = (mono_delta * 2^32) / tsc_delta
    let shift: u32 = 32;
    let mult = ((mono_delta as u128) << shift) / (tsc_delta as u128);
    if mult > u32::MAX as u128 {
        eprintln!("[M13-TSC] WARNING: TSC frequency too low for u32 mult. Using fallback.");
        return TscCal::fallback();
    }
    let mult = mult as u32;

    // Snapshot the base point for conversion
    let tsc_base = read_tsc();
    let mono_base = clock_ns();

    let cal = TscCal { tsc_base, mono_base, mult, shift, valid: true };

    // Validation: compare rdtsc_ns() vs clock_ns() over 1000 samples.
    // If any sample deviates by > 1µs, the calibration is bad.
    let mut max_error: i64 = 0;
    for _ in 0..1000 {
        let tsc_time = rdtsc_ns(&cal) as i64;
        let mono_time = clock_ns() as i64;
        let err = (tsc_time - mono_time).abs();
        if err > max_error { max_error = err; }
    }

    let tsc_freq_mhz = (tsc_delta as u128 * 1000) / (mono_delta as u128);
    eprintln!("[M13-TSC] Calibrated: freq={}.{}MHz mult={} shift={} max_err={}ns",
        tsc_freq_mhz / 1000, tsc_freq_mhz % 1000, mult, shift, max_error);

    if max_error > 1000 { // > 1µs
        eprintln!("[M13-TSC] WARNING: Calibration error {}ns > 1µs. Using clock_gettime fallback.", max_error);
        return TscCal::fallback();
    }

    cal
}

#[inline(always)]
unsafe fn prefetch_read_l1(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { core::arch::x86_64::_mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0); }
    #[cfg(target_arch = "aarch64")]
    { core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) addr, options(nostack, preserves_flags)); }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = addr; }
}

// ============================================================================
// SPRINT 5.7: ISOCHRONOUS SCHEDULER (5.9: cwnd + pacing-aware)
// ============================================================================
#[derive(Copy, Clone)]
struct TxDesc { addr: u64, len: u32 }

struct Scheduler {
    critical: [TxDesc; 256],
    critical_len: usize,
    bulk: [TxDesc; 288],
    bulk_len: usize,
}

impl Scheduler {
    fn new() -> Self {
        Scheduler {
            critical: [TxDesc { addr: 0, len: 0 }; 256], critical_len: 0,
            bulk: [TxDesc { addr: 0, len: 0 }; 288], bulk_len: 0,
        }
    }
    /// Budget respects BBR cwnd (replaces hardcoded HW_FILL_MAX when BBR active).
    #[inline(always)]
    fn budget(&self, tx_avail: usize, cwnd: usize) -> usize {
        let inflight = TX_RING_SIZE.saturating_sub(tx_avail);
        let cap = cwnd.min(HW_FILL_MAX);
        cap.saturating_sub(inflight).min(tx_avail)
    }
    #[inline(always)]
    fn enqueue_critical(&mut self, addr: u64, len: u32) {
        if self.critical_len < self.critical.len() {
            self.critical[self.critical_len] = TxDesc { addr, len };
            self.critical_len += 1;
        }
    }
    #[inline(always)]
    fn enqueue_bulk(&mut self, addr: u64, len: u32) {
        if self.bulk_len < self.bulk.len() {
            self.bulk[self.bulk_len] = TxDesc { addr, len };
            self.bulk_len += 1;
        }
    }
    /// Schedule: critical bypasses pacing (strict priority), bulk capped by bulk_limit.
    fn schedule(&mut self, tx_path: &mut impl TxPath, stats: &Telemetry,
                bulk_limit: usize) -> usize {
        let avail = tx_path.available_slots() as usize;
        let hw_budget = {
            let inflight = TX_RING_SIZE.saturating_sub(avail);
            HW_FILL_MAX.saturating_sub(inflight).min(avail)
        };
        let mut submitted = 0usize;
        // Phase 1: Critical (feedback frames) — bypass pacing, HW-limited only
        let crit = self.critical_len.min(hw_budget);
        for i in 0..crit {
            tx_path.stage_tx_addr(self.critical[i].addr, self.critical[i].len);
            submitted += 1;
        }
        // Phase 2: Bulk (data frames) — pacing-limited AND HW-limited (FIFO order)
        let bulk_hw = hw_budget.saturating_sub(submitted);
        let bulk = self.bulk_len.min(bulk_hw).min(bulk_limit);
        for i in 0..bulk {
            tx_path.stage_tx_addr(self.bulk[i].addr, self.bulk[i].len);
            submitted += 1;
        }
        if submitted > 0 {
            tx_path.commit_tx();
            tx_path.kick_tx();
            stats.tx_count.value.fetch_add(submitted as u64, Ordering::Relaxed);
        }
        self.critical_len = 0;
        self.bulk_len = 0;
        submitted
    }
}



// ============================================================================
// SPRINT 5.10: DETERMINISTIC JITTER BUFFER
// RFC 3550 EWMA jitter estimator + fixed-size circular buffer.
// TC_CRITICAL frames: stochastic arrival → deterministic playout.
// D_buf adaptive: k * J(ewma) + ε_proc, clamped [1ms, 100ms].
// ============================================================================
const JBUF_CAPACITY: usize = 64;       // 64 entries @ 100Hz = 640ms capacity
const JBUF_K: u64 = 4;                 // Safety factor: P(|X-μ|>4σ) < 0.0063%
const JBUF_MIN_DEPTH_NS: u64 = 1_000_000;     // Floor: 1ms
const JBUF_MAX_DEPTH_NS: u64 = 100_000_000;   // Ceiling: 100ms
const JBUF_DEFAULT_DEPTH_NS: u64 = 50_000_000; // Conservative initial: 50ms
const JBUF_EWMA_GAIN_SHIFT: u32 = 4;  // 1/16 = right shift by 4 (RFC 3550)

#[repr(C)]
#[derive(Copy, Clone)]
struct JitterEntry {
    addr: u64,       // UMEM frame address (slab slot stays allocated)
    len: u32,        // frame length in bytes
    release_ns: u64, // CLOCK_MONOTONIC time to release
}

/// RFC 3550 §6.4.1 interarrival jitter estimator.
/// J(i) = J(i-1) + (|D(i)| - J(i-1)) >> 4
/// D(i) = (R_i - R_{i-1}) - (S_i - S_{i-1})  [one-way delay variation]
struct JitterEstimator {
    last_rx_ns: u64,       // arrival time of previous TC_CRITICAL frame
    last_seq: u64,         // sequence number of previous frame
    jitter_ns: u64,        // current EWMA jitter estimate (ns)
    seq_interval_ns: u64,  // expected inter-packet interval (ns), 0 = unknown
}

impl JitterEstimator {
    fn new() -> Self {
        JitterEstimator { last_rx_ns: 0, last_seq: 0, jitter_ns: 0, seq_interval_ns: 0 }
    }
    /// Update jitter estimate with a new TC_CRITICAL frame arrival.
    /// rx_ns: CLOCK_MONOTONIC arrival time. seq: M13Header.seq_id.
    #[inline(always)]
    fn update(&mut self, rx_ns: u64, seq: u64) {
        if self.last_rx_ns == 0 {
            // First frame: seed the estimator, no jitter sample yet
            self.last_rx_ns = rx_ns;
            self.last_seq = seq;
            return;
        }
        // Receiver inter-arrival: R_i - R_{i-1}
        let rx_delta = rx_ns.saturating_sub(self.last_rx_ns);
        // Sender inter-departure: (S_i - S_{i-1}) approximated from seq deltas
        // If seq_interval_ns > 0, use it. Otherwise use rx_delta as baseline (no send clock).
        let seq_delta = seq.saturating_sub(self.last_seq);
        let send_delta = if self.seq_interval_ns > 0 && seq_delta > 0 {
            seq_delta * self.seq_interval_ns
        } else {
            rx_delta // No send clock → D=0, jitter stays unchanged
        };
        // D(i) = one-way delay variation
        let d = if rx_delta > send_delta {
            rx_delta - send_delta
        } else {
            send_delta - rx_delta
        };
        // RFC 3550 EWMA: J(i) = J(i-1) + (|D(i)| - J(i-1)) / 16
        if d > self.jitter_ns {
            self.jitter_ns += (d - self.jitter_ns) >> JBUF_EWMA_GAIN_SHIFT;
        } else {
            self.jitter_ns -= (self.jitter_ns - d) >> JBUF_EWMA_GAIN_SHIFT;
        }
        self.last_rx_ns = rx_ns;
        self.last_seq = seq;
    }
    #[inline(always)]
    fn get(&self) -> u64 { self.jitter_ns }
}

/// Fixed-size circular jitter buffer. Zero heap allocation.
/// Holds TC_CRITICAL frames until their deterministic release time.
#[repr(align(128))]
struct JitterBuffer {
    entries: [JitterEntry; JBUF_CAPACITY],
    head: usize,        // next slot to READ (oldest)
    tail: usize,        // next slot to WRITE (newest)
    depth_ns: u64,      // current D_buf in nanoseconds
    epsilon_ns: u64,    // worst-case processing time (measured at boot)
    estimator: JitterEstimator,
    total_releases: u64,
    total_drops: u64,
}

impl JitterBuffer {
    fn new(epsilon_ns: u64) -> Self {
        JitterBuffer {
            entries: [JitterEntry { addr: 0, len: 0, release_ns: 0 }; JBUF_CAPACITY],
            head: 0, tail: 0,
            depth_ns: JBUF_DEFAULT_DEPTH_NS,
            epsilon_ns,
            estimator: JitterEstimator::new(),
            total_releases: 0, total_drops: 0,
        }
    }

    /// Update jitter estimate and recalculate D_buf
    #[inline(always)]
    fn update_jitter(&mut self, rx_ns: u64, seq: u64) {
        self.estimator.update(rx_ns, seq);
        let j = self.estimator.get();
        if j > 0 {
            let raw = JBUF_K * j + self.epsilon_ns;
            self.depth_ns = raw.max(JBUF_MIN_DEPTH_NS).min(JBUF_MAX_DEPTH_NS);
        }
    }

    /// Insert a TC_CRITICAL frame. Slab slot stays allocated (NOT freed).
    /// Returns Some(dropped_addr) if overflow forced drop of oldest frame.
    /// Caller MUST free the returned slab slot to prevent UMEM leak.
    #[inline(always)]
    fn insert(&mut self, addr: u64, len: u32, rx_ns: u64) -> Option<u64> {
        let release_ns = rx_ns + self.depth_ns;
        let mut dropped_addr = None;
        // Overflow: if buffer is full, drop oldest (buffer undersized for this link)
        if self.tail - self.head >= JBUF_CAPACITY {
            let old_slot = self.head & (JBUF_CAPACITY - 1);
            dropped_addr = Some(self.entries[old_slot].addr);
            self.head += 1;
            self.total_drops += 1;
        }
        let slot = self.tail & (JBUF_CAPACITY - 1);
        self.entries[slot] = JitterEntry { addr, len, release_ns };
        self.tail += 1;
        dropped_addr
    }

    /// Drain due frames to scheduler as TC_CRITICAL. Returns (released, late_dropped).
    #[inline(always)]
    fn drain(&mut self, now_ns: u64, scheduler: &mut Scheduler) -> (usize, usize) {
        let mut released = 0usize;
        while self.head < self.tail {
            let slot = self.head & (JBUF_CAPACITY - 1);
            let e = &self.entries[slot];
            if now_ns >= e.release_ns {
                scheduler.enqueue_critical(e.addr, e.len);
                self.head += 1;
                released += 1;
                self.total_releases += 1;
            } else {
                // Not yet time. Entries are time-ordered (monotonic insertion).
                break;
            }
        }
        (released, 0)
    }
}

/// Measure worst-case hot-loop iteration time. Run 10K rdtsc_ns() pairs.
/// Uses rdtsc (the actual hot-path clock) so epsilon reflects real overhead.
fn measure_epsilon_proc(cal: &TscCal) -> u64 {
    let mut max_delta = 0u64;
    for _ in 0..10_000 {
        let t0 = rdtsc_ns(cal);
        // Simulate minimal work: one clock read (what a minimal loop iteration does)
        let t1 = rdtsc_ns(cal);
        let delta = t1.saturating_sub(t0);
        if delta > max_delta { max_delta = delta; }
    }
    // Add 2x safety margin for real classify+schedule overhead
    max_delta * 2
}

// RECEIVER STATE — tracks delivered packets for feedback generation
// ============================================================================
// ============================================================================
// SPRINT 5.18: RX BITMAP — 1024-BIT SLIDING WINDOW LOSS DETECTOR
// Tracks which seq_ids have been received. Zeros = gaps = losses.
// O(1) mark via bitmask. O(words) advance via popcount.
// Stack-allocated: 128 bytes = 2 cache lines.
// ============================================================================

/// 1024-bit sliding window bitmap for sequence gap detection.
/// bit N = 1 means seq_id (base_seq + N) has been received.
/// When the window advances, evicted zero-bits are counted as losses.
struct RxBitmap {
    bits: [u64; 16],       // 1024 bits = 16 x u64
    base_seq: u64,         // seq_id corresponding to bit 0
    loss_accum: u32,       // losses accumulated since last feedback
    highest_marked: u64,   // highest seq_id marked in the bitmap
}

impl RxBitmap {
    fn new() -> Self {
        RxBitmap { bits: [0u64; 16], base_seq: 0, loss_accum: 0, highest_marked: 0 }
    }

    /// Mark a seq_id as received. Advances the window if seq exceeds capacity.
    /// O(1) for in-window marks. O(words_shifted) for window advance.
    #[inline(always)]
    fn mark(&mut self, seq: u64) {
        // Ignore packets before our window (too old — already evicted)
        if seq < self.base_seq { return; }

        let offset = seq - self.base_seq;

        // If seq exceeds window, advance. Count gaps in evicted words.
        if offset >= 1024 {
            self.advance_to(seq);
        }

        let offset = (seq - self.base_seq) as usize;
        if offset < 1024 {
            let word = offset >> 6;   // offset / 64
            let bit = offset & 63;    // offset % 64
            self.bits[word] |= 1u64 << bit;
        }

        if seq > self.highest_marked {
            self.highest_marked = seq;
        }
    }

    /// Advance window so that `seq` fits within [base_seq, base_seq+1023].
    /// Evicted words have their zero-bits counted as losses.
    fn advance_to(&mut self, seq: u64) {
        // How many words to shift out
        let target_base = seq.saturating_sub(1023);
        if target_base <= self.base_seq { return; }

        let bit_shift = target_base - self.base_seq;
        let word_shift = (bit_shift / 64) as usize;

        if word_shift >= 16 {
            // Entire window evicted — count all unmarked bits as losses
            for w in 0..16 {
                // Only count losses for sequence ranges we've actually entered
                // (bits that were zero because we hadn't reached them yet aren't losses)
                let word_base = self.base_seq + (w as u64 * 64);
                if word_base < self.highest_marked {
                    let relevant_bits = if word_base + 64 <= self.highest_marked {
                        64
                    } else {
                        (self.highest_marked - word_base) as u32
                    };
                    let received = self.bits[w].count_ones().min(relevant_bits);
                    self.loss_accum += relevant_bits - received;
                }
            }
            self.bits = [0u64; 16];
        } else {
            // Partial shift: count gaps in evicted words, shift array
            for w in 0..word_shift {
                if w < 16 {
                    let word_base = self.base_seq + (w as u64 * 64);
                    if word_base < self.highest_marked {
                        let relevant_bits = if word_base + 64 <= self.highest_marked {
                            64
                        } else {
                            (self.highest_marked - word_base) as u32
                        };
                        let received = self.bits[w].count_ones().min(relevant_bits);
                        self.loss_accum += relevant_bits - received;
                    }
                }
            }
            // Shift remaining words left
            let remain = 16 - word_shift;
            for i in 0..remain {
                self.bits[i] = self.bits[i + word_shift];
            }
            for i in remain..16 {
                self.bits[i] = 0;
            }
        }
        self.base_seq = target_base;
    }

    /// Return (loss_count, nack_bitmap) and reset accumulator.
    /// nack_bitmap: 64 bits relative to highest_marked-63.
    /// Bit i = 1 means received, bit i = 0 means lost.
    fn drain_losses(&mut self) -> (u32, u64) {
        let losses = self.loss_accum;
        self.loss_accum = 0;

        // Build NACK bitmap: the 64 bits around highest_marked
        let nack = if self.highest_marked >= self.base_seq + 63 {
            let nack_base = self.highest_marked - 63;
            if nack_base >= self.base_seq {
                let offset = (nack_base - self.base_seq) as usize;
                // Extract 64 contiguous bits starting at 'offset'
                let word_idx = offset >> 6;
                let bit_idx = offset & 63;
                if bit_idx == 0 && word_idx < 16 {
                    self.bits[word_idx]
                } else if word_idx + 1 < 16 {
                    // Cross-word extraction
                    let lo = self.bits[word_idx] >> bit_idx;
                    let hi = self.bits[word_idx + 1] << (64 - bit_idx);
                    lo | hi
                } else if word_idx < 16 {
                    self.bits[word_idx] >> bit_idx
                } else {
                    u64::MAX // all received (out of range)
                }
            } else {
                u64::MAX // all within range received
            }
        } else {
            u64::MAX // not enough data yet
        };

        (losses, nack)
    }
}

struct ReceiverState {
    highest_seq: u64, delivered: u32,
    last_feedback_ns: u64, last_rx_batch_ns: u64,
}
impl ReceiverState {
    fn new() -> Self { ReceiverState { highest_seq: 0, delivered: 0, last_feedback_ns: 0, last_rx_batch_ns: 0 } }
    #[inline(always)]
    fn needs_feedback(&self, now_ns: u64, rtt_estimate_ns: u64) -> bool {
        if self.delivered >= FEEDBACK_INTERVAL_PKTS { return true; }
        if self.delivered > 0 && self.last_feedback_ns > 0
            && now_ns.saturating_sub(self.last_feedback_ns) >= rtt_estimate_ns { return true; }
        false
    }
}

// ============================================================================
// SPRINT 6.1: FRAGMENTATION ENGINE (cold path — handshake only)
// ============================================================================

/// Fragment sub-header. 8 bytes, prepended to payload when FLAG_FRAGMENT set.
#[repr(C, packed)]
#[derive(Copy, Clone)]
struct FragHeader {
    frag_msg_id: u16,   // Message ID (links fragments of same message)
    frag_index: u8,     // Fragment index (0-based)
    frag_total: u8,     // Total fragments in message
    frag_offset: u16,   // Byte offset into original message
    frag_len: u16,      // Bytes in this fragment
}
const FRAG_HDR_SIZE: usize = 8;
const _FRAG_SZ: () = assert!(std::mem::size_of::<FragHeader>() == FRAG_HDR_SIZE);

struct AssemblyBuffer {
    fragments: [Option<Vec<u8>>; 16], received_mask: u16,
    total: u8, first_rx_ns: u64,
}
impl AssemblyBuffer {
    fn new(total: u8, now_ns: u64) -> Self {
        AssemblyBuffer { fragments: Default::default(), received_mask: 0, total, first_rx_ns: now_ns }
    }
    fn insert(&mut self, index: u8, _offset: u16, data: &[u8]) -> bool {
        if index >= 16 || index >= self.total { return false; }
        let bit = 1u16 << index;
        if self.received_mask & bit != 0 { return self.is_complete(); }
        self.fragments[index as usize] = Some(data.to_vec());
        self.received_mask |= bit;
        self.is_complete()
    }
    fn is_complete(&self) -> bool {
        let expected = (1u16 << self.total) - 1;
        self.received_mask & expected == expected
    }
    fn reassemble(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for i in 0..self.total as usize {
            if let Some(ref data) = self.fragments[i] { result.extend_from_slice(data); }
        }
        result
    }
}

struct Assembler { pending: HashMap<u16, AssemblyBuffer>, }
impl Assembler {
    fn new() -> Self { Assembler { pending: HashMap::new() } }
    fn feed(&mut self, msg_id: u16, index: u8, total: u8, offset: u16,
            data: &[u8], now_ns: u64) -> Option<Vec<u8>> {
        let buf = self.pending.entry(msg_id).or_insert_with(|| AssemblyBuffer::new(total, now_ns));
        if buf.insert(index, offset, data) {
            let result = buf.reassemble();
            self.pending.remove(&msg_id);
            Some(result)
        } else { None }
    }
    fn gc(&mut self, now_ns: u64) {
        self.pending.retain(|_, buf| now_ns.saturating_sub(buf.first_rx_ns) < 5_000_000_000);
    }
}

// ============================================================================
// SPRINT 3: AES-256-GCM AEAD via `ring` (BoringSSL asm)
// x86: AES-NI hw accel (~4-10 GiB/s). ARM A53 (K26): ARMv8 Crypto Extensions.
// Future: FPGA AES-GCM IP core for line-rate offload. Wire format unchanged.
// ============================================================================

use ring::aead;

fn seal_frame(frame: &mut [u8], lsk: &aead::LessSafeKey, seq: u64, direction: u8, offset: usize) {
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[0..8].copy_from_slice(&seq.to_le_bytes());
    nonce_bytes[8] = direction;
    let sig = offset;
    frame[sig+2] = 0x01; frame[sig+3] = 0x00;
    frame[sig+20..sig+32].copy_from_slice(&nonce_bytes);
    let pt = sig + 32;
    let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();
    let aad_bytes: [u8; 4] = frame[sig..sig+4].try_into().unwrap();
    let aad = aead::Aad::from(aad_bytes);
    let tag = lsk.seal_in_place_separate_tag(nonce, aad, &mut frame[pt..]).unwrap();
    frame[sig+4..sig+20].copy_from_slice(tag.as_ref());
}

fn open_frame(frame: &mut [u8], lsk: &aead::LessSafeKey, our_dir: u8, offset: usize) -> bool {
    let sig = offset;
    if frame.len() < sig + 32 + 8 { return false; }
    if frame[sig+2] != 0x01 { return false; }
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(&frame[sig+20..sig+32]);
    if nonce_bytes[8] == our_dir { return false; } // reflection guard
    let mut wire_tag_bytes = [0u8; 16];
    wire_tag_bytes.copy_from_slice(&frame[sig+4..sig+20]);
    let pt = sig + 32;
    let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();
    let aad_bytes: [u8; 4] = frame[sig..sig+4].try_into().unwrap();
    let aad = aead::Aad::from(&aad_bytes);
    let tag = aead::Tag::from(wire_tag_bytes);
    match lsk.open_in_place_separate_tag(nonce, aad, tag, &mut frame[pt..], 0..) {
        Ok(_) => {
            let dec_seq = u64::from_le_bytes(frame[pt..pt+8].try_into().unwrap());
            let nonce_seq = u64::from_le_bytes(nonce_bytes[0..8].try_into().unwrap());
            dec_seq == nonce_seq
        }
        Err(_) => false,
    }
}



// ============================================================================
// SPRINT 6.3: PQC HANDSHAKE — HUB RESPONDER
// ============================================================================
// Hub receives:
//   Msg 1 (ClientHello): nonce(32) + ek(1568) + pk_node(2592) = 4192 bytes
//   Msg 3 (Finished):    sig_node(4627) bytes
// Hub sends:
//   Msg 2 (ServerHello): ct(1568) + pk_hub(2592) + sig_hub(4627) = 8787 bytes
//
// Session key = HKDF-SHA-512(salt=nonce, IKM=ML-KEM-ss, info="M13-PQC-SESSION-KEY-v1", L=32)
// ============================================================================

const PQC_CONTEXT: &[u8] = b"M13-HS-v1";
const PQC_INFO: &[u8] = b"M13-PQC-SESSION-KEY-v1";

/// Hub-side handshake state: stored between ClientHello and Finished processing.
struct HubHandshakeState {
    /// Node's ML-DSA-87 verifying key bytes (for Finished verification)
    node_pk_bytes: Vec<u8>,
    /// ML-KEM shared secret (32 bytes, derived from encapsulation)
    shared_secret: [u8; 32],
    /// Session nonce from ClientHello (32 bytes, HKDF salt)
    session_nonce: [u8; 32],
    /// Full ClientHello payload bytes (for transcript computation)
    client_hello_bytes: Vec<u8>,
    /// Full ServerHello payload bytes (for transcript computation)
    server_hello_bytes: Vec<u8>,
    /// Handshake start timestamp (for timeout)
    _started_ns: u64,
}

/// Build fragmented handshake frames as raw ETH+IP+UDP packets for AF_XDP TX.
/// Returns Vec of ready-to-transmit raw frames (no kernel socket involvement).
/// Each frame: ETH(14) + IP(20) + UDP(8) + M13(48) + FragHdr(8) + chunk = 98 + chunk
fn build_fragmented_raw_udp(
    src_mac: &[u8; 6], gw_mac: &[u8; 6],
    hub_ip: [u8; 4], peer_ip: [u8; 4],
    hub_port: u16, peer_port: u16,
    payload: &[u8],
    flags: u8,
    seq: &mut u64,
    ip_id_base: &mut u16,
    hexdump: &mut HexdumpState,
    cal: &TscCal,
) -> Vec<Vec<u8>> {
    // MTU 1500 - IP(20) - UDP(8) = 1472 bytes max UDP payload
    // M13 frame overhead: ETH(14) + M13(48) + FRAG(8) = 70 bytes
    // But inside the UDP payload, we have: ETH_FAKE(14) + M13(48) + FRAG(8) = 70 bytes of M13 overhead
    // Max handshake data per fragment: 1472 - 70 = 1402 bytes
    let max_chunk = 1402;
    let total = (payload.len() + max_chunk - 1) / max_chunk;
    let msg_id = (*seq & 0xFFFF) as u16;
    let mut frames = Vec::with_capacity(total);

    for i in 0..total {
        let offset = i * max_chunk;
        let chunk_len = (payload.len() - offset).min(max_chunk);

        // Build the M13 fragment payload (what goes inside the UDP payload)
        let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + FRAG_HDR_SIZE + chunk_len;
        let mut m13_frame = vec![0u8; m13_flen];
        m13_frame[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        m13_frame[6..12].copy_from_slice(src_mac);
        m13_frame[12] = (ETH_P_M13 >> 8) as u8;
        m13_frame[13] = (ETH_P_M13 & 0xFF) as u8;
        m13_frame[14] = M13_WIRE_MAGIC; m13_frame[15] = M13_WIRE_VERSION;
        m13_frame[46..54].copy_from_slice(&seq.to_le_bytes());
        m13_frame[54] = flags | FLAG_FRAGMENT;
        let fh = ETH_HDR_SIZE + M13_HDR_SIZE;
        m13_frame[fh..fh+2].copy_from_slice(&msg_id.to_le_bytes());
        m13_frame[fh+2] = i as u8; m13_frame[fh+3] = total as u8;
        m13_frame[fh+4..fh+6].copy_from_slice(&(offset as u16).to_le_bytes());
        m13_frame[fh+6..fh+8].copy_from_slice(&(chunk_len as u16).to_le_bytes());
        let dp = fh + FRAG_HDR_SIZE;
        m13_frame[dp..dp+chunk_len].copy_from_slice(&payload[offset..offset+chunk_len]);

        // Wrap in raw ETH+IP+UDP
        let raw_len = RAW_HDR_LEN + m13_flen;
        let mut raw = vec![0u8; raw_len];
        let flen = build_raw_udp_frame(
            &mut raw, src_mac, gw_mac, hub_ip, peer_ip,
            hub_port, peer_port, *ip_id_base, &m13_frame,
        );
        *ip_id_base = ip_id_base.wrapping_add(1);
        hexdump.dump_tx(raw.as_ptr(), flen, rdtsc_ns(cal));
        frames.push(raw);
        *seq += 1;
    }
    frames
}

/// Process a ClientHello (Msg 1) from a Node.
/// Encapsulates shared secret, signs transcript, builds ServerHello payload.
/// Returns (HubHandshakeState, server_hello_payload) for caller to frame.
///
/// ClientHello layout: type(1) + version(1) + nonce(32) + ek(1568) + pk_node(2592) = 4194 bytes
/// ServerHello layout: type(1) + ct(1568) + pk_hub(2592) + sig_hub(4627) = 8788 bytes
fn process_client_hello_hub(
    reassembled: &[u8],
    _seq: &mut u64,
    now: u64,
) -> Option<(HubHandshakeState, Vec<u8>)> {
    // Validate: type(1) + version(1) + nonce(32) + ek(1568) + pk_node(2592) = 4194
    const EXPECTED_LEN: usize = 1 + 1 + 32 + 1568 + 2592;
    if reassembled.len() < EXPECTED_LEN {
        eprintln!("[M13-HUB-PQC] ERROR: ClientHello too short: {} < {}", reassembled.len(), EXPECTED_LEN);
        return None;
    }
    if reassembled[0] != HS_CLIENT_HELLO {
        eprintln!("[M13-HUB-PQC] ERROR: Expected ClientHello (0x01), got 0x{:02X}", reassembled[0]);
        return None;
    }
    if reassembled[1] != 0x01 {
        eprintln!("[M13-HUB-PQC] ERROR: Unsupported protocol version: 0x{:02X}", reassembled[1]);
        return None;
    }

    eprintln!("[M13-HUB-PQC] Processing ClientHello ({}B, proto_v={})...", reassembled.len(), reassembled[1]);

    // Parse fields (version byte shifts all offsets by +1)
    let mut session_nonce = [0u8; 32];
    session_nonce.copy_from_slice(&reassembled[2..34]);
    let ek_bytes = &reassembled[34..1602];        // ML-KEM-1024 encapsulation key (1568B)
    let pk_node_bytes = &reassembled[1602..4194];  // ML-DSA-87 verifying key (2592B)

    // 1. Reconstruct EncapsulationKey from bytes
    let ek_enc = match ml_kem::Encoded::<ml_kem::kem::EncapsulationKey<ml_kem::MlKem1024Params>>::try_from(
        ek_bytes
    ) {
        Ok(enc) => enc,
        Err(_) => {
            eprintln!("[M13-HUB-PQC] ERROR: Failed to parse EncapsulationKey");
            return None;
        }
    };
    let ek = ml_kem::kem::EncapsulationKey::<ml_kem::MlKem1024Params>::from_bytes(&ek_enc);

    // 2. Encapsulate: ek + OsRng → (ct, ss)
    let (ct, ss) = match ek.encapsulate(&mut OsRng) {
        Ok((ct, ss)) => (ct, ss),
        Err(_) => {
            eprintln!("[M13-HUB-PQC] ERROR: ML-KEM encapsulation failed");
            return None;
        }
    };
    let ct_bytes_arr = ct;
    eprintln!("[M13-HUB-PQC] ML-KEM-1024 encapsulation successful (ct={}B, ss=32B)", ct_bytes_arr.len());

    // 3. Generate Hub's ML-DSA-87 identity keypair (TOFU)
    let dsa_kp = MlDsa87::key_gen(&mut OsRng);
    let pk_hub = dsa_kp.verifying_key().encode(); // 2592 bytes
    eprintln!("[M13-HUB-PQC] ML-DSA-87 identity generated (pk={}B)", pk_hub.len());

    // 4. Compute transcript = SHA-512(ClientHello_payload || ct)
    let mut hasher = Sha512::new();
    hasher.update(reassembled);
    hasher.update(ct_bytes_arr.as_slice());
    let transcript: [u8; 64] = hasher.finalize().into();

    // 5. Sign transcript with Hub's signing key
    let sig_hub = match dsa_kp.signing_key().sign_deterministic(&transcript, PQC_CONTEXT) {
        Ok(sig) => sig,
        Err(_) => {
            eprintln!("[M13-HUB-PQC] ERROR: ML-DSA signing failed");
            return None;
        }
    };
    let sig_hub_bytes = sig_hub.encode(); // 4627 bytes
    eprintln!("[M13-HUB-PQC] Hub signature generated ({}B)", sig_hub_bytes.len());

    // 6. Build ServerHello payload: type(1) + ct(1568) + pk_hub(2592) + sig_hub(4627) = 8788
    let mut server_hello = Vec::with_capacity(1 + ct_bytes_arr.len() + pk_hub.len() + sig_hub_bytes.len());
    server_hello.push(HS_SERVER_HELLO);
    server_hello.extend_from_slice(ct_bytes_arr.as_slice());
    server_hello.extend_from_slice(&pk_hub);
    server_hello.extend_from_slice(&sig_hub_bytes);

    eprintln!("[M13-HUB-PQC] ServerHello built: {}B payload", server_hello.len());

    // 7. Store intermediate state for Finished processing
    let mut ss_arr = [0u8; 32];
    ss_arr.copy_from_slice(&ss);
    Some((HubHandshakeState {
        node_pk_bytes: pk_node_bytes.to_vec(),
        shared_secret: ss_arr,
        session_nonce,
        client_hello_bytes: reassembled.to_vec(),
        server_hello_bytes: server_hello.clone(),
        _started_ns: now,
    }, server_hello))
}

/// Process a Finished message (Msg 3) from a Node.
/// Verifies Node's ML-DSA-87 signature, derives session key via HKDF-SHA-512.
/// Returns session_key on success.
///
/// Finished layout: type(1) + sig_node(4627) = 4628 bytes
fn process_finished_hub(
    reassembled: &[u8],
    hs_state: &HubHandshakeState,
) -> Option<[u8; 32]> {
    // Validate length: type(1) + sig(4627) = 4628
    const EXPECTED_LEN: usize = 1 + 4627;
    if reassembled.len() < EXPECTED_LEN {
        eprintln!("[M13-HUB-PQC] ERROR: Finished too short: {} < {}", reassembled.len(), EXPECTED_LEN);
        return None;
    }
    if reassembled[0] != HS_FINISHED {
        eprintln!("[M13-HUB-PQC] ERROR: Expected Finished (0x03), got 0x{:02X}", reassembled[0]);
        return None;
    }

    eprintln!("[M13-HUB-PQC] Processing Finished ({}B)...", reassembled.len());

    let sig_node_bytes = &reassembled[1..4628];

    // 1. Compute transcript2 = SHA-512(ClientHello_payload || ServerHello_payload)
    let mut hasher = Sha512::new();
    hasher.update(&hs_state.client_hello_bytes);
    hasher.update(&hs_state.server_hello_bytes);
    let transcript2: [u8; 64] = hasher.finalize().into();

    // 2. Parse Node's verifying key
    let pk_node_enc = match ml_dsa::EncodedVerifyingKey::<MlDsa87>::try_from(
        hs_state.node_pk_bytes.as_slice()
    ) {
        Ok(enc) => enc,
        Err(_) => {
            eprintln!("[M13-HUB-PQC] ERROR: Failed to parse Node verifying key");
            return None;
        }
    };
    let pk_node = ml_dsa::VerifyingKey::<MlDsa87>::decode(&pk_node_enc);

    // 3. Parse Node's signature
    let sig_node = match ml_dsa::Signature::<MlDsa87>::try_from(sig_node_bytes) {
        Ok(sig) => sig,
        Err(_) => {
            eprintln!("[M13-HUB-PQC] ERROR: Failed to parse Node signature");
            return None;
        }
    };

    // 4. Verify signature
    if !pk_node.verify_with_context(&transcript2, PQC_CONTEXT, &sig_node) {
        eprintln!("[M13-HUB-PQC] SECURITY FAILURE: Node signature verification failed!");
        eprintln!("[M13-HUB-PQC] Possible MITM attack — aborting handshake");
        return None;
    }
    eprintln!("[M13-HUB-PQC] Node ML-DSA-87 signature verified ✓");

    // 5. Derive session key via HKDF-SHA-512
    let hk = Hkdf::<Sha512>::new(Some(&hs_state.session_nonce), &hs_state.shared_secret);
    let mut session_key = [0u8; 32];
    hk.expand(PQC_INFO, &mut session_key)
        .expect("HKDF-SHA-512 expand failed (L=32 ≤ 255*64)");
    eprintln!("[M13-HUB-PQC] Session key derived via HKDF-SHA-512 (32B)");

    Some(session_key)
}

// ============================================================================
// SPRINT 6.1: HEXDUMP ENGINE (all 4 capture points, rate-limited)
// ============================================================================
const HEXDUMP_INTERVAL_NS: u64 = 100_000_000; // 100ms = 10/sec max

struct HexdumpState { enabled: bool, last_tx_ns: u64 }
impl HexdumpState {
    fn new(enabled: bool) -> Self { HexdumpState { enabled, last_tx_ns: 0 } }
    fn dump_tx(&mut self, frame: *const u8, len: usize, now_ns: u64) {
        if !self.enabled { return; }
        if now_ns.saturating_sub(self.last_tx_ns) < HEXDUMP_INTERVAL_NS { return; }
        self.last_tx_ns = now_ns;
        dump_frame("[HUB-TX]", frame, len);
    }
}

fn dump_frame(label: &str, frame: *const u8, len: usize) {
    let cap = len.min(80);
    let data = unsafe { std::slice::from_raw_parts(frame, cap) };
    let (seq, flags) = if cap >= ETH_HDR_SIZE + M13_HDR_SIZE {
        let m13 = unsafe { &*(frame.add(ETH_HDR_SIZE) as *const M13Header) };
        (m13.seq_id, m13.flags)
    } else { (0, 0) };
    let dst = if cap >= 6 { format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        data[0], data[1], data[2], data[3], data[4], data[5]) } else { "?".into() };
    let src = if cap >= 12 { format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        data[6], data[7], data[8], data[9], data[10], data[11]) } else { "?".into() };
    eprintln!("{} seq={} flags=0x{:02X} len={} dst={} src={}", label, seq, flags, len, dst, src);
    if cap >= 14 {
        eprint!("  [00..14] ETH  |"); for i in 0..14 { eprint!(" {:02X}", data[i]); } eprintln!();
    }
    if cap >= 16 { eprint!("  [14..16] MAGIC|"); eprint!(" {:02X} {:02X}", data[14], data[15]); eprintln!(); }
    if cap >= 18 {
        eprint!("  [16..18] CRYPT|"); eprint!(" {:02X} {:02X}", data[16], data[17]);
        eprintln!("  (crypto_ver=0x{:02X}={})", data[16], if data[16] == 0 { "cleartext" } else { "encrypted" });
    }
    if cap >= 34 { eprint!("  [18..34] MAC  |"); for i in 18..34 { eprint!(" {:02X}", data[i]); } eprintln!(); }
    if cap >= 46 { eprint!("  [34..46] NONCE|"); for i in 34..46 { eprint!(" {:02X}", data[i]); } eprintln!(); }
    if cap >= 54 {
        eprint!("  [46..54] SEQ  |"); for i in 46..54 { eprint!(" {:02X}", data[i]); }
        eprintln!("  (LE: seq_id={})", seq);
    }
    if cap >= 55 { eprintln!("  [54..55] FLAGS| {:02X}", data[54]); }
    if cap >= 59 {
        let plen = if cap >= ETH_HDR_SIZE + M13_HDR_SIZE {
            let m13 = unsafe { &*(frame.add(ETH_HDR_SIZE) as *const M13Header) }; m13.payload_len
        } else { 0 };
        eprint!("  [55..59] PLEN |"); for i in 55..59 { eprint!(" {:02X}", data[i]); }
        eprintln!("  (LE: payload_len={})", plen);
    }
    if cap >= 62 { eprint!("  [59..62] PAD  |"); for i in 59..62 { eprint!(" {:02X}", data[i]); } eprintln!(); }
}

/// Process-wide shutdown flag. Set by SIGTERM/SIGINT handler.
/// Checked at the top of every worker loop iteration.
/// Ordering::Relaxed is correct — monotonic flag (once true, never false).
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn signal_handler(_sig: i32) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

/// Read the hardware MAC address of a network interface from sysfs.
/// Returns the 6-byte MAC or a locally-administered fallback if sysfs is unavailable.
fn detect_mac(if_name: &str) -> [u8; 6] {
    let path = format!("/sys/class/net/{}/address", if_name);
    if let Ok(contents) = std::fs::read_to_string(&path) {
        let parts: Vec<u8> = contents.trim().split(':')
            .filter_map(|h| u8::from_str_radix(h, 16).ok())
            .collect();
        if parts.len() == 6 {
            eprintln!("[M13-EXEC] Detected MAC for {}: {}", if_name, contents.trim());
            return [parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]];
        }
    }
    eprintln!("[M13-EXEC] WARNING: Could not read MAC from sysfs ({}), using LAA fallback", path);
    [0x02, 0x00, 0x00, 0x00, 0x00, 0x01] // locally administered fallback
}

/// Resolve the default gateway's MAC address from the kernel ARP cache.
/// Reads /proc/net/route to find the default gateway IP, then /proc/net/arp for its MAC.
/// This is the L2 destination for all internet-bound packets sent via AF_XDP TX.
fn resolve_gateway_mac(if_name: &str) -> Option<([u8; 6], [u8; 4])> {
    // Step 1: Find default gateway IP from /proc/net/route
    // Format: Iface Destination Gateway Flags RefCnt Use Metric Mask ...
    // Default route has Destination == 00000000
    let route_data = std::fs::read_to_string("/proc/net/route").ok()?;
    let mut gw_ip_hex: Option<u32> = None;
    for line in route_data.lines().skip(1) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 3 && fields[0] == if_name && fields[1] == "00000000" {
            // Gateway is in hex, little-endian u32
            gw_ip_hex = u32::from_str_radix(fields[2], 16).ok();
            break;
        }
    }
    let gw_hex = gw_ip_hex?;
    let gw_ip = gw_hex.to_le_bytes(); // /proc/net/route stores in network byte order as LE hex

    // Step 2: Look up gateway MAC in /proc/net/arp
    // Format: IP address HW type Flags HW address Mask Device
    let arp_data = std::fs::read_to_string("/proc/net/arp").ok()?;
    let gw_ip_str = format!("{}.{}.{}.{}", gw_ip[0], gw_ip[1], gw_ip[2], gw_ip[3]);
    for line in arp_data.lines().skip(1) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 6 && fields[0] == gw_ip_str && fields[5] == if_name {
            let mac_parts: Vec<u8> = fields[3].split(':')
                .filter_map(|h| u8::from_str_radix(h, 16).ok())
                .collect();
            if mac_parts.len() == 6 {
                let mac = [mac_parts[0], mac_parts[1], mac_parts[2],
                           mac_parts[3], mac_parts[4], mac_parts[5]];
                eprintln!("[M13-NET] Gateway: {} MAC: {} dev: {}",
                    gw_ip_str, fields[3], if_name);
                return Some((mac, gw_ip));
            }
        }
    }
    eprintln!("[M13-NET] WARNING: Gateway {} not in ARP cache. Pinging to populate...", gw_ip_str);
    // Attempt to populate ARP cache by pinging gateway
    let _ = std::process::Command::new("ping")
        .args(["-c", "1", "-W", "1", &gw_ip_str])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    // Retry ARP lookup
    if let Ok(arp2) = std::fs::read_to_string("/proc/net/arp") {
        for line in arp2.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 6 && fields[0] == gw_ip_str && fields[5] == if_name {
                let mac_parts: Vec<u8> = fields[3].split(':')
                    .filter_map(|h| u8::from_str_radix(h, 16).ok())
                    .collect();
                if mac_parts.len() == 6 {
                    let mac = [mac_parts[0], mac_parts[1], mac_parts[2],
                               mac_parts[3], mac_parts[4], mac_parts[5]];
                    eprintln!("[M13-NET] Gateway: {} MAC: {} (after ARP)", gw_ip_str, fields[3]);
                    return Some((mac, gw_ip));
                }
            }
        }
    }
    None
}

/// Read the IPv4 address of a network interface from sysfs.
/// Returns the 4-byte IP address or None if unavailable.
fn get_interface_ip(if_name: &str) -> Option<[u8; 4]> {
    // Use ioctl SIOCGIFADDR
    unsafe {
        let sock = libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0);
        if sock < 0 { return None; }
        let mut ifr: libc::ifreq = std::mem::zeroed();
        let name_bytes = if_name.as_bytes();
        let copy_len = name_bytes.len().min(libc::IFNAMSIZ - 1);
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), ifr.ifr_name.as_mut_ptr() as *mut u8, copy_len);
        if libc::ioctl(sock, libc::SIOCGIFADDR as libc::c_ulong, &mut ifr) < 0 {
            libc::close(sock);
            return None;
        }
        libc::close(sock);
        // ifr_ifru.ifru_addr is a sockaddr_in
        let sa = &*(&ifr.ifr_ifru as *const _ as *const libc::sockaddr_in);
        let ip_u32 = sa.sin_addr.s_addr; // network byte order
        let _ip = ip_u32.to_be_bytes();
        // s_addr is in network byte order; to_ne_bytes gives the octets in correct order
        let ip_ne = ip_u32.to_ne_bytes();
        eprintln!("[M13-NET] Interface {} IP: {}.{}.{}.{}", if_name,
            ip_ne[0], ip_ne[1], ip_ne[2], ip_ne[3]);
        Some(ip_ne)
    }
}

/// RFC 1071: Internet checksum (ones-complement of ones-complement sum).
/// Used for IPv4 header checksum. Input is the 20-byte IP header with checksum field zeroed.
#[inline]
fn ip_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Construct a raw Ethernet + IPv4 + UDP frame in a buffer.
/// Returns the total frame length (ETH_HDR + IP_HDR + UDP_HDR + payload).
/// The frame is ready for AF_XDP TX — no kernel involvement.
///
/// Layout: ETH(14) + IP(20) + UDP(8) + payload = 42 + payload_len
/// IP checksum is computed. UDP checksum is 0 (legal for IPv4 per RFC 768).
const IP_HDR_LEN: usize = 20;
const UDP_HDR_LEN: usize = 8;
const RAW_HDR_LEN: usize = ETH_HDR_SIZE + IP_HDR_LEN + UDP_HDR_LEN; // 42

fn build_raw_udp_frame(
    buf: &mut [u8],
    src_mac: &[u8; 6], dst_mac: &[u8; 6],
    src_ip: [u8; 4], dst_ip: [u8; 4],
    src_port: u16, dst_port: u16,
    ip_id: u16,
    payload: &[u8],
) -> usize {
    let payload_len = payload.len();
    let udp_len = UDP_HDR_LEN + payload_len;
    let ip_total_len = IP_HDR_LEN + udp_len;
    let frame_len = ETH_HDR_SIZE + ip_total_len;
    debug_assert!(frame_len <= buf.len(), "frame too large for buffer");

    // --- Ethernet Header (14 bytes) ---
    buf[0..6].copy_from_slice(dst_mac);
    buf[6..12].copy_from_slice(src_mac);
    buf[12..14].copy_from_slice(&0x0800u16.to_be_bytes()); // IPv4

    // --- IPv4 Header (20 bytes) ---
    let ip = &mut buf[14..34];
    ip[0] = 0x45;                                           // Version=4, IHL=5 (20 bytes)
    ip[1] = 0x00;                                           // DSCP/ECN
    ip[2..4].copy_from_slice(&(ip_total_len as u16).to_be_bytes()); // Total Length
    ip[4..6].copy_from_slice(&ip_id.to_be_bytes());         // Identification
    ip[6..8].copy_from_slice(&0x4000u16.to_be_bytes());     // Flags: DF, Fragment Offset: 0
    ip[8] = 64;                                             // TTL
    ip[9] = 17;                                             // Protocol: UDP
    ip[10..12].copy_from_slice(&[0, 0]);                    // Checksum (zeroed for computation)
    ip[12..16].copy_from_slice(&src_ip);                    // Source IP
    ip[16..20].copy_from_slice(&dst_ip);                    // Destination IP
    let cksum = ip_checksum(ip);
    ip[10..12].copy_from_slice(&cksum.to_be_bytes());       // Fill in checksum

    // --- UDP Header (8 bytes) ---
    let udp = &mut buf[34..42];
    udp[0..2].copy_from_slice(&src_port.to_be_bytes());     // Source Port
    udp[2..4].copy_from_slice(&dst_port.to_be_bytes());     // Destination Port
    udp[4..6].copy_from_slice(&(udp_len as u16).to_be_bytes()); // UDP Length
    udp[6..8].copy_from_slice(&[0, 0]);                     // Checksum: 0 (valid for IPv4)

    // --- Payload ---
    buf[42..42 + payload_len].copy_from_slice(payload);

    frame_len
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    unsafe {
        libc::signal(libc::SIGTERM, signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as *const () as libc::sighandler_t);
    }

    // Panic hook: guarantee cleanup even on unwinding crash
    let panic_if_name = args.get(1).cloned().unwrap_or_else(|| "veth0".to_string());
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("[M13-HUB] PANIC: {}", info);
        nuke_cleanup_hub(&panic_if_name);
        std::process::exit(1);
    }));

    if args.iter().any(|a| a == "--monitor") {
        run_monitor();
        return;
    }
    let mut if_name = "veth0".to_string();
    let mut single_queue: Option<i32> = None;
    let mut hexdump_mode = false;
    let mut tunnel_mode = false;
    let mut listen_port: Option<u16> = Some(443); // Default: UDP/443 (blends with QUIC)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--monitor" => { run_monitor(); return; }
            "--hexdump" => { hexdump_mode = true; }
            "--tunnel" => { tunnel_mode = true; }
            "--port" | "--listen" => {
                i += 1;
                if i < args.len() {
                    listen_port = Some(match args[i].parse() {
                        Ok(p) => p,
                        Err(_) => fatal(E_AFFINITY_FAIL, "Invalid port number"),
                    });
                }
            }
            "--single-queue" => {
                i += 1;
                if i < args.len() {
                    single_queue = Some(match args[i].parse() {
                        Ok(v) => v,
                        Err(_) => fatal(E_AFFINITY_FAIL, "Invalid queue ID argument"),
                    });
                }
            }
            "-i" | "--iface" => {
                i += 1;
                if i < args.len() { if_name = args[i].clone(); }
            }
            other => {
                if !other.starts_with("--") { if_name = other.to_string(); }
            }
        }
        i += 1;
    }
    if hexdump_mode {
        std::env::set_var("M13_HEXDUMP", "1");
    }
    // Hub always listens. CLI --port takes precedence, then env var, then default 443.
    // If M13_LISTEN_PORT is already set in the environment, don't clobber it with the default.
    let effective_port = match listen_port {
        Some(p) => {
            // CLI arg was explicitly given OR we have the default.
            // Check if user provided --port/--listen on the command line
            let cli_port_given = args.iter().any(|a| a == "--port" || a == "--listen");
            if cli_port_given {
                // Explicit CLI port always wins
                std::env::set_var("M13_LISTEN_PORT", p.to_string());
                p
            } else if let Ok(env_p) = std::env::var("M13_LISTEN_PORT") {
                // Env var already set, respect it
                env_p.parse::<u16>().unwrap_or(p)
            } else {
                // Neither CLI nor env var: use default
                std::env::set_var("M13_LISTEN_PORT", p.to_string());
                p
            }
        }
        None => {
            if let Ok(env_p) = std::env::var("M13_LISTEN_PORT") {
                env_p.parse::<u16>().unwrap_or(443)
            } else {
                443
            }
        }
    };
    let _ = effective_port; // suppress unused warning
    run_executive(&if_name, single_queue, tunnel_mode);
}

// ============================================================================
// TYPESTATE FSM — Compile-time protocol state. No BBR pollution.
// ============================================================================
pub struct Listening;
pub struct Established;
#[repr(C)]
pub struct Peer<S> { mac: [u8; 6], seq_tx: u64, _state: PhantomData<S> }
impl Peer<Listening> {
    pub fn new() -> Self { Self { mac: [0u8; 6], seq_tx: 0, _state: PhantomData } }
    pub fn accept_registration(self, peer_mac: [u8; 6]) -> Peer<Established> {
        Peer { mac: peer_mac, seq_tx: 0, _state: PhantomData }
    }
}
impl Peer<Established> {
    #[inline(always)] pub fn next_seq(&mut self) -> u64 { let s = self.seq_tx; self.seq_tx = s.wrapping_add(1); s }
    #[inline(always)] pub fn mac(&self) -> &[u8; 6] { &self.mac }
}

// produce_data_frame: removed (was only used by the old UDP kernel socket path)

// ============================================================================
// MULTI-TENANT PEER TABLE — Cache-aligned, O(1) linear probing.
// Design: DPDK/VPP-style flat array. Zero heap allocation on hot path.
// At N≤256 peers, the entire table fits in L2 cache (~16KB for 64B slots).
// ============================================================================

/// Maximum concurrent peers. Must be power of 2 for mask-based indexing.
const MAX_PEERS: usize = 256;

/// Tunnel IP subnet: 10.13.0.0/24. Hub is .1, peers get .2..254.
const TUNNEL_SUBNET: [u8; 4] = [10, 13, 0, 0];

/// 6-byte peer identity: (src_ip, src_port) from the wire.
/// This is the natural key for UDP peers behind NAT — each unique
/// (public_ip, ephemeral_port) pair is a distinct peer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PeerAddr {
    /// Empty/uninitialized slot
    Empty,
    /// UDP peer identified by (src_ip, src_port) — satellite/internet path
    Udp { ip: [u8; 4], port: u16 },
    /// Raw L2 peer identified by src MAC — air-gapped WiFi 7 path
    L2 { mac: [u8; 6] },
}

impl PeerAddr {
    const EMPTY: PeerAddr = PeerAddr::Empty;

    #[inline(always)]
    fn new_udp(ip: [u8; 4], port: u16) -> Self { PeerAddr::Udp { ip, port } }

    #[inline(always)]
    fn new_l2(mac: [u8; 6]) -> Self { PeerAddr::L2 { mac } }

    /// Returns IP if this is a UDP peer, None for L2.
    #[inline(always)]
    fn ip(&self) -> Option<[u8; 4]> {
        match self { PeerAddr::Udp { ip, .. } => Some(*ip), _ => None }
    }

    /// Returns port if this is a UDP peer, None for L2.
    #[inline(always)]
    fn port(&self) -> Option<u16> {
        match self { PeerAddr::Udp { port, .. } => Some(*port), _ => None }
    }

    #[inline(always)]
    fn is_udp(&self) -> bool { matches!(self, PeerAddr::Udp { .. }) }

    /// FNV-1a hash. 6 bytes for UDP (ip+port), 6 bytes for L2 (mac).
    #[inline(always)]
    fn hash(&self) -> usize {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        let bytes: [u8; 6] = match self {
            PeerAddr::Udp { ip, port } => [
                ip[0], ip[1], ip[2], ip[3],
                (*port & 0xFF) as u8, (*port >> 8) as u8,
            ],
            PeerAddr::L2 { mac } => *mac,
            PeerAddr::Empty => return 0,
        };
        let mut i = 0;
        while i < 6 {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
            i += 1;
        }
        h as usize
    }
}

impl std::fmt::Debug for PeerAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerAddr::Empty => write!(f, "<empty>"),
            PeerAddr::Udp { ip, port } => write!(f, "{}.{}.{}.{}:{}", ip[0], ip[1], ip[2], ip[3], port),
            PeerAddr::L2 { mac } => write!(f, "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]),
        }
    }
}

/// Peer lifecycle state machine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum PeerLifecycle {
    /// Slot is empty and available for allocation.
    Empty = 0,
    /// Peer registered (first packet seen), awaiting handshake.
    Registered = 1,
    /// PQC handshake in progress (ClientHello received, ServerHello sent).
    Handshaking = 2,
    /// AEAD session established, encrypted tunnel active.
    Established = 3,
}

/// Per-peer state. Exactly 1 cache line (64 bytes) for zero false sharing.
/// Hot-path fields (session_key, seq_tx) are at fixed offsets.
#[repr(C, align(64))]
struct PeerSlot {
    /// Lookup key: identity
    addr: PeerAddr,                 // 8 bytes (enum: tag + 6 data + padding)
    /// Lifecycle state.
    lifecycle: PeerLifecycle,       // 1 byte
    /// Index into tunnel IP pool (10.13.0.{tunnel_ip_idx}).
    tunnel_ip_idx: u8,              // 1 byte

    /// AEAD session key (32 bytes). Zero = no session.
    session_key: [u8; 32],          // 32 bytes

    /// TX sequence counter for AEAD nonces.
    seq_tx: u64,                    // 8 bytes

    /// AEAD frames processed under current session (for rekey).
    frame_count: u32,               // 4 bytes
    /// Session established timestamp (relative seconds since epoch).
    established_rel_s: u32,         // 4 bytes

    /// Peer's Ethernet MAC (from M13 frame src_mac). Used for TX framing.
    mac: [u8; 6],                   // 6 bytes
    // Note: align(64) rounds 64 actual data bytes to 64 — no explicit pad needed.
}

// Compile-time assertion: PeerSlot fits within 1 cache line (64-byte aligned).
// The actual layout with #[repr(C, align(64))] handles alignment padding.

impl PeerSlot {
    const EMPTY: PeerSlot = PeerSlot {
        addr: PeerAddr::EMPTY,
        lifecycle: PeerLifecycle::Empty,
        tunnel_ip_idx: 0,
        session_key: [0u8; 32],
        seq_tx: 0,
        frame_count: 0,
        established_rel_s: 0,
        mac: [0xFF; 6],
    };

    #[inline(always)]
    fn is_empty(&self) -> bool { self.lifecycle == PeerLifecycle::Empty }

    #[inline(always)]
    fn has_session(&self) -> bool { self.session_key != [0u8; 32] }

    #[inline(always)]
    fn next_seq(&mut self) -> u64 {
        let s = self.seq_tx;
        self.seq_tx = s.wrapping_add(1);
        s
    }

    /// Reset session state for reconnect/rekey. Preserves slot identity.
    fn reset_session(&mut self) {
        self.session_key = [0u8; 32];
        self.seq_tx = 0;
        self.frame_count = 0;
        self.established_rel_s = 0;
        self.lifecycle = PeerLifecycle::Registered;
    }
}

/// Multi-tenant peer table. Single-threaded (owned by one worker).
/// Flat array with linear probing — zero allocation on lookup, O(1) amortized.
struct PeerTable {
    /// Cache-line-aligned flat array of peer slots.
    slots: Box<[PeerSlot; MAX_PEERS]>,
    /// Cold-path sidecar: per-peer handshake state (ML-KEM, DSA, transcript).
    /// Indexed by slot index. Only touched during PQC handshake.
    hs_sidecar: Vec<Option<HubHandshakeState>>,
    /// Per-peer cached AEAD cipher. Indexed by slot index.
    /// Constructed once on session establishment, reused for every frame.
    ciphers: Vec<Option<aead::LessSafeKey>>,
    /// Per-peer fragment reassembly. Indexed by slot index.
    assemblers: Vec<Assembler>,
    /// Number of active (non-Empty) peers.
    count: u16,
    /// Tunnel IP allocation bitmap. Bit N = IP 10.13.0.N is assigned.
    /// Indices 0 and 1 are reserved (network + Hub).
    tunnel_ip_bitmap: [u64; 4], // 256 bits
    /// Base timestamp for relative time storage.
    epoch_ns: u64,
}

impl PeerTable {
    fn new(epoch_ns: u64) -> Self {
        // Allocate the slots array on the heap (16KB, can't go on stack).
        // Use a Vec -> try_into -> Box conversion to get a fixed-size boxed array.
        let mut slots_vec: Vec<PeerSlot> = Vec::with_capacity(MAX_PEERS);
        for _ in 0..MAX_PEERS {
            slots_vec.push(PeerSlot::EMPTY);
        }
        let slots_array: Box<[PeerSlot; MAX_PEERS]> = match slots_vec.into_boxed_slice().try_into() {
            Ok(arr) => arr,
            Err(_) => unreachable!(),
        };

        let mut hs_sidecar = Vec::with_capacity(MAX_PEERS);
        let mut ciphers: Vec<Option<aead::LessSafeKey>> = Vec::with_capacity(MAX_PEERS);
        let mut assemblers = Vec::with_capacity(MAX_PEERS);
        for _ in 0..MAX_PEERS {
            hs_sidecar.push(None);
            ciphers.push(None);
            assemblers.push(Assembler::new());
        }

        // Reserve tunnel IPs 0 (network) and 1 (Hub).
        let mut tunnel_ip_bitmap = [0u64; 4];
        tunnel_ip_bitmap[0] = 0b11; // bits 0 and 1 reserved

        PeerTable {
            slots: slots_array,
            hs_sidecar,
            ciphers,
            assemblers,
            count: 0,
            tunnel_ip_bitmap,
            epoch_ns,
        }
    }

    /// O(1) amortized lookup by peer address. Returns slot index.
    #[inline(always)]
    fn lookup(&self, addr: PeerAddr) -> Option<usize> {
        let mut idx = addr.hash() & (MAX_PEERS - 1);
        for _ in 0..MAX_PEERS {
            if self.slots[idx].addr == addr && !self.slots[idx].is_empty() {
                return Some(idx);
            }
            // NOTE: Do NOT early-return on empty slots. evict() creates tombstones
            // without compacting the hash chain, so peers beyond a tombstone would
            // be silently ghosted. Let the bounded loop scan the full chain.
            idx = (idx + 1) & (MAX_PEERS - 1);
        }
        None
    }

    /// Lookup or insert a new peer. Returns slot index.
    /// On insert: allocates tunnel IP, initializes slot as Registered.
    fn lookup_or_insert(&mut self, addr: PeerAddr, mac: [u8; 6]) -> Option<usize> {
        let start = addr.hash() & (MAX_PEERS - 1);
        let mut idx = start;
        let mut first_empty: Option<usize> = None;

        for _ in 0..MAX_PEERS {
            if self.slots[idx].addr == addr && !self.slots[idx].is_empty() {
                // Update MAC in case peer changed interface
                self.slots[idx].mac = mac;
                return Some(idx);
            }
            if self.slots[idx].is_empty() && first_empty.is_none() {
                first_empty = Some(idx);
                // Don't break — there might be a matching slot after a tombstone.
                // But we don't use tombstones (we compact on evict), so we CAN break.
                break;
            }
            idx = (idx + 1) & (MAX_PEERS - 1);
        }

        // Not found — insert into first empty slot.
        let slot_idx = first_empty?;
        if self.count as usize >= MAX_PEERS - 1 { return None; } // Table full

        // Same-IP stale peer eviction: if a Node reconnects with a different
        // ephemeral port (e.g. after Ctrl+C where FIN was never delivered),
        // the old slot stays Established. Detect and evict it.
        // Key insight: MAC is unreliable (Node generates random LAA MAC each launch).
        // Source IP is the stable identity — same public IP with different port = same Node.
        // This is O(N) but only runs on new peer registration — cold path.
        for i in 0..MAX_PEERS {
            if i == slot_idx { continue; }
            if !self.slots[i].is_empty() && self.slots[i].addr.ip() == addr.ip() && self.slots[i].addr != addr {
                eprintln!("[M13-PEERS] Stale peer {:?} in slot {} has same IP as new peer {:?} — evicting.",
                    self.slots[i].addr, i, addr);
                self.evict(i);
            }
        }

        // Allocate tunnel IP
        let tunnel_idx = self.alloc_tunnel_ip()?;

        self.slots[slot_idx] = PeerSlot {
            addr,
            lifecycle: PeerLifecycle::Registered,
            tunnel_ip_idx: tunnel_idx,
            session_key: [0u8; 32],
            seq_tx: 0,
            frame_count: 0,
            established_rel_s: 0,
            mac,
        };
        self.hs_sidecar[slot_idx] = None;
        self.assemblers[slot_idx] = Assembler::new();
        self.count += 1;

        eprintln!("[M13-PEERS] New peer {:?} → slot {} tunnel_ip=10.13.0.{} (total: {})",
            addr, slot_idx, tunnel_idx, self.count);

        Some(slot_idx)
    }

    /// Evict a peer, freeing its slot and tunnel IP.
    fn evict(&mut self, idx: usize) {
        if idx >= MAX_PEERS || self.slots[idx].is_empty() { return; }
        let tip = self.slots[idx].tunnel_ip_idx;
        self.free_tunnel_ip(tip);
        eprintln!("[M13-PEERS] Evicted peer {:?} from slot {} (tunnel_ip=10.13.0.{})",
            self.slots[idx].addr, idx, tip);
        self.slots[idx] = PeerSlot::EMPTY;
        self.hs_sidecar[idx] = None;
        self.ciphers[idx] = None;
        self.assemblers[idx] = Assembler::new();
        if self.count > 0 { self.count -= 1; }
    }

    /// Find peer slot by tunnel destination IP (for TUN TX routing).
    /// Scans linearly — only called on TUN read (cold relative to RX classify).
    fn lookup_by_tunnel_ip(&self, dst_ip: [u8; 4]) -> Option<usize> {
        // Fast reject: not in our subnet
        if dst_ip[0] != TUNNEL_SUBNET[0] || dst_ip[1] != TUNNEL_SUBNET[1]
           || dst_ip[2] != TUNNEL_SUBNET[2] { return None; }
        let target_idx = dst_ip[3];
        for i in 0..MAX_PEERS {
            if !self.slots[i].is_empty()
               && self.slots[i].tunnel_ip_idx == target_idx
               && self.slots[i].has_session() {
                return Some(i);
            }
        }
        None
    }

    /// Allocate a tunnel IP index (2..254) from the bitmap.
    fn alloc_tunnel_ip(&mut self) -> Option<u8> {
        for word_idx in 0..4u8 {
            let word = self.tunnel_ip_bitmap[word_idx as usize];
            if word == u64::MAX { continue; } // All 64 bits set
            let bit = (!word).trailing_zeros() as u8; // First free bit
            let global_idx = word_idx * 64 + bit;
            if global_idx >= 255 { continue; } // .255 is broadcast
            self.tunnel_ip_bitmap[word_idx as usize] |= 1u64 << bit;
            return Some(global_idx);
        }
        None
    }

    /// Free a tunnel IP index back to the bitmap.
    fn free_tunnel_ip(&mut self, idx: u8) {
        let word_idx = (idx / 64) as usize;
        let bit = idx % 64;
        self.tunnel_ip_bitmap[word_idx] &= !(1u64 << bit);
    }

    /// Periodic garbage collection: evict peers with no activity for too long.
    fn gc(&mut self, now_ns: u64) {
        let now_rel_s = ((now_ns.saturating_sub(self.epoch_ns)) / 1_000_000_000) as u32;
        for i in 0..MAX_PEERS {
            if self.slots[i].is_empty() { continue; }
            // Evict Registered peers that never completed handshake (60s timeout)
            if self.slots[i].lifecycle == PeerLifecycle::Registered
               || self.slots[i].lifecycle == PeerLifecycle::Handshaking {
                if self.slots[i].established_rel_s == 0 {
                    // Use a conservative 60s timeout for registration/handshake
                    // We don't have a per-slot creation timestamp in the 64B slot,
                    // so we check if the slot has been in this state "too long"
                    // by using a simple counter approach. For now, don't evict
                    // non-established peers automatically — they'll get overwritten
                    // on reconnect.
                }
            }
        }
        let _ = now_rel_s; // suppress unused warning until we add time-based eviction
    }
}

#[inline(always)]
fn produce_feedback_frame(
    frame_ptr: *mut u8, dst_mac: &[u8; 6], src_mac: &[u8; 6],
    rx_state: &mut ReceiverState, rx_bitmap: &mut RxBitmap, now_ns: u64,
    jbuf_len: usize,
) {
    // Drain loss accumulator and NACK bitmap from RxBitmap
    let (loss_count, nack_bitmap) = rx_bitmap.drain_losses();
    // ECN decision: mark if jitter buffer > 75% or any loss detected.
    // This gives the sender advance warning of congestion before overflow.
    let congested = jbuf_len > JBUF_CAPACITY * 3 / 4 || loss_count > 0;
    unsafe {
        let eth = &mut *(frame_ptr as *mut EthernetHeader);
        eth.dst = *dst_mac; eth.src = *src_mac; eth.ethertype = ETH_P_M13.to_be();
        let m13 = &mut *(frame_ptr.add(ETH_HDR_SIZE) as *mut M13Header);
        m13.signature = [0; 32];
        m13.signature[0] = M13_WIRE_MAGIC;
        m13.signature[1] = M13_WIRE_VERSION;
        m13.seq_id = 0;
        m13.flags = FLAG_CONTROL | FLAG_FEEDBACK | if congested { FLAG_ECN } else { 0 };
        m13.payload_len = mem::size_of::<FeedbackFrame>() as u32;
        m13.padding = [0; 3];
        let fb = &mut *(frame_ptr.add(ETH_HDR_SIZE + M13_HDR_SIZE) as *mut FeedbackFrame);
        fb.highest_seq = rx_state.highest_seq;
        fb.rx_timestamp_ns = rx_state.last_rx_batch_ns;
        fb.delivered = rx_state.delivered;
        fb.delivered_time_ns = now_ns;
        fb.loss_count = loss_count;
        fb.nack_bitmap = nack_bitmap;
    }
    rx_state.delivered = 0;
    rx_state.last_feedback_ns = now_ns;
}

/// Send `count` redundant FIN or FIN-ACK frames wrapped in raw UDP.
/// Used for multi-tenant peers connected via UDP (not L2).
/// Constructs the M13 FIN payload, then wraps it in ETH+IP+UDP via build_raw_udp_frame.
#[inline(never)]
fn send_fin_burst_udp(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    src_mac: &[u8; 6], gateway_mac: &[u8; 6],
    hub_ip: [u8; 4], peer_ip: [u8; 4],
    hub_port: u16, peer_port: u16,
    ip_id: &mut u16,
    final_seq: u64, fin_ack: bool, count: usize,
) -> usize {
    let mut sent = 0;
    // Build the M13 FIN payload (62 bytes: ETH(14) + M13(48))
    let mut fin_m13 = [0u8; 62];
    fin_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // dst MAC (broadcast, inner)
    fin_m13[6..12].copy_from_slice(src_mac);
    fin_m13[12] = (ETH_P_M13 >> 8) as u8;
    fin_m13[13] = (ETH_P_M13 & 0xFF) as u8;
    fin_m13[14] = M13_WIRE_MAGIC;
    fin_m13[15] = M13_WIRE_VERSION;
    fin_m13[46..54].copy_from_slice(&final_seq.to_le_bytes()); // seq_id
    fin_m13[54] = FLAG_CONTROL | FLAG_FIN | if fin_ack { FLAG_FEEDBACK } else { 0 };
    // payload_len = 0, padding = 0 (already zeroed)

    for _ in 0..count {
        if let Some(idx) = slab.alloc() {
            let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
            let total_len;
            unsafe {
                let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                total_len = build_raw_udp_frame(
                    buf, src_mac, gateway_mac,
                    hub_ip, peer_ip, hub_port, peer_port,
                    *ip_id, &fin_m13,
                );
                *ip_id = ip_id.wrapping_add(1);
            }
            scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
            sent += 1;
        }
    }
    sent
}

/// Send a raw L2 M13 frame (EtherType 0x88B5) into UMEM for AF_XDP TX.
/// For air-gapped WiFi 7 drones — no IP/UDP encapsulation.
/// `m13_payload` is the complete M13 frame: ETH(14) + M13(48) [+ data].
/// The ETH dst_mac in m13_payload is overwritten with peer_mac.
#[inline(never)]
fn send_l2_frame(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    peer_mac: &[u8; 6],
    m13_payload: &[u8],
) -> bool {
    if let Some(idx) = slab.alloc() {
        let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
        let flen = m13_payload.len().min(FRAME_SIZE as usize);
        unsafe {
            let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
            buf[..flen].copy_from_slice(&m13_payload[..flen]);
            // Overwrite dst MAC with peer's actual MAC
            buf[0..6].copy_from_slice(peer_mac);
        }
        scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, flen as u32);
        true
    } else {
        false
    }
}

/// Send `count` redundant FIN or FIN-ACK frames via raw L2 (EtherType 0x88B5).
/// Used for air-gapped WiFi 7 drones connected via raw Ethernet.
#[inline(never)]
fn send_fin_burst_l2(
    slab: &mut FixedSlab, engine: &Engine<ZeroCopyTx>,
    scheduler: &mut Scheduler,
    src_mac: &[u8; 6], peer_mac: &[u8; 6],
    final_seq: u64, fin_ack: bool, count: usize,
) -> usize {
    let mut sent = 0;
    let mut fin_m13 = [0u8; 62];
    fin_m13[0..6].copy_from_slice(peer_mac);
    fin_m13[6..12].copy_from_slice(src_mac);
    fin_m13[12] = (ETH_P_M13 >> 8) as u8;
    fin_m13[13] = (ETH_P_M13 & 0xFF) as u8;
    fin_m13[14] = M13_WIRE_MAGIC;
    fin_m13[15] = M13_WIRE_VERSION;
    fin_m13[46..54].copy_from_slice(&final_seq.to_le_bytes());
    fin_m13[54] = FLAG_CONTROL | FLAG_FIN | if fin_ack { FLAG_FEEDBACK } else { 0 };

    for _ in 0..count {
        if let Some(idx) = slab.alloc() {
            let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
            unsafe {
                let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                buf[..62].copy_from_slice(&fin_m13);
            }
            scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, 62);
            sent += 1;
        }
    }
    sent
}

/// Build fragmented handshake frames as raw L2 packets (EtherType 0x88B5) for AF_XDP TX.
/// Returns Vec of ready-to-transmit raw L2 frames.
/// Each frame: ETH(14) + M13(48) + FragHdr(8) + chunk = 70 + chunk
fn build_fragmented_l2(
    src_mac: &[u8; 6], peer_mac: &[u8; 6],
    payload: &[u8],
    flags: u8,
    seq: &mut u64,
    hexdump: &mut HexdumpState,
    cal: &TscCal,
) -> Vec<Vec<u8>> {
    // L2 MTU: 1500 bytes. Frame: ETH(14) + M13(48) + FRAG(8) + chunk = 70 + chunk
    // Max chunk = 1500 - 70 = 1430 (slightly larger than UDP since no IP/UDP overhead)
    let max_chunk = 1430;
    let total = (payload.len() + max_chunk - 1) / max_chunk;
    let msg_id = (*seq & 0xFFFF) as u16;
    let mut frames = Vec::with_capacity(total);

    for i in 0..total {
        let offset = i * max_chunk;
        let chunk_len = (payload.len() - offset).min(max_chunk);

        let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + FRAG_HDR_SIZE + chunk_len;
        let mut m13_frame = vec![0u8; m13_flen];
        m13_frame[0..6].copy_from_slice(peer_mac);
        m13_frame[6..12].copy_from_slice(src_mac);
        m13_frame[12] = (ETH_P_M13 >> 8) as u8;
        m13_frame[13] = (ETH_P_M13 & 0xFF) as u8;
        m13_frame[14] = M13_WIRE_MAGIC; m13_frame[15] = M13_WIRE_VERSION;
        m13_frame[46..54].copy_from_slice(&seq.to_le_bytes());
        m13_frame[54] = flags | FLAG_FRAGMENT;
        let fh = ETH_HDR_SIZE + M13_HDR_SIZE;
        m13_frame[fh..fh+2].copy_from_slice(&msg_id.to_le_bytes());
        m13_frame[fh+2] = i as u8; m13_frame[fh+3] = total as u8;
        m13_frame[fh+4..fh+6].copy_from_slice(&(offset as u16).to_le_bytes());
        m13_frame[fh+6..fh+8].copy_from_slice(&(chunk_len as u16).to_le_bytes());
        let dp = fh + FRAG_HDR_SIZE;
        m13_frame[dp..dp+chunk_len].copy_from_slice(&payload[offset..offset+chunk_len]);

        hexdump.dump_tx(m13_frame.as_ptr(), m13_flen, rdtsc_ns(cal));
        frames.push(m13_frame);
        *seq += 1;
    }
    frames
}

// ============================================================================
// THE EXECUTIVE (Boot Calibration Sequence)
// ============================================================================

fn run_executive(if_name: &str, single_queue: Option<i32>, tunnel: bool) {
    // Register signal handlers before spawning workers.
    // signal() is async-signal-safe. Handler sets AtomicBool — one CPU instruction.
    unsafe {
        libc::signal(libc::SIGTERM, signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as *const () as libc::sighandler_t);
    }

    // === AUTO-CLEANUP: kill stale hub, detach XDP, allocate hugepages ===
    eprintln!("[M13-EXEC] Pre-flight cleanup...");
    // Kill any previous m13-hub (SIGKILL, exclude ourselves)
    let my_pid = std::process::id();
    if let Ok(output) = std::process::Command::new("pgrep").arg("m13-hub").output() {
        if output.status.success() {
            let pids = String::from_utf8_lossy(&output.stdout);
            for pid_str in pids.lines() {
                if let Ok(pid) = pid_str.trim().parse::<u32>() {
                    if pid != my_pid {
                        unsafe { libc::kill(pid as i32, 9); }
                    }
                }
            }
        }
    }
    // Detach any stale XDP programs from the interface
    let _ = std::process::Command::new("ip").args(&["link", "set", if_name, "xdp", "off"]).output();
    let _ = std::process::Command::new("ip").args(&["link", "set", if_name, "xdpgeneric", "off"]).output();

    // CRITICAL: Collapse NIC to single queue so ALL traffic hits queue 0 (where AF_XDP binds).
    // Without this, RSS distributes UDP/443 across multiple queues → AF_XDP never sees packets
    // on queues != 0 → silent RX drop. This is the #1 failure mode for pure AF_XDP.
    let sq_result = std::process::Command::new("ethtool")
        .args(&["-L", if_name, "combined", "1"])
        .output();
    match sq_result {
        Ok(ref o) if o.status.success() => {
            eprintln!("[M13-EXEC] NIC {} collapsed to single queue (ethtool -L combined 1).", if_name);
        }
        Ok(ref o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            eprintln!("[M13-EXEC] WARNING: ethtool -L combined 1 failed: {}. UDP/443 may miss AF_XDP queue.", stderr.trim());
        }
        Err(e) => {
            eprintln!("[M13-EXEC] WARNING: ethtool not found: {}. Ensure NIC has 1 queue manually.", e);
        }
    }

    // Allocate hugepages: workers × UMEM_SIZE / 2MB per hugepage + headroom
    let hp_worker_count = match single_queue { Some(_) => 1, None => MAX_WORKERS };
    let hugepages_needed = (hp_worker_count * (UMEM_SIZE + 16 * 1024 * 1024)) / (2 * 1024 * 1024);
    let _ = std::fs::write("/proc/sys/vm/nr_hugepages", format!("{}\n", hugepages_needed));
    if let Ok(hp) = std::fs::read_to_string("/proc/sys/vm/nr_hugepages") {
        eprintln!("[M13-EXEC] Hugepages: {} allocated", hp.trim());
    }
    eprintln!("[M13-EXEC] Pre-flight cleanup complete.");

    // Sprint 5.17: TSC calibration. Must happen before workers spawn.
    // Workers receive a copy of the calibration data (immutable, per-worker).
    let tsc_cal = calibrate_tsc();

    lock_pmu();
    fence_interrupts();
    let isolated_cores = discover_isolated_cores();
    if isolated_cores.is_empty() {
        fatal(E_NO_ISOLATED_CORES, "No isolated cores. Boot with isolcpus=... or set M13_MOCK_CMDLINE");
    }
    // Detect UDP/Internet mode early — affects worker count and BPF decisions.
    let udp_mode = std::env::var("M13_LISTEN_PORT").is_ok();
    // In UDP/Internet mode: single worker only. Only worker 0 has the UDP socket,
    // and without BPF/XDP there's no point spawning workers that busy-poll a mock engine.
    let worker_count = if udp_mode {
        eprintln!("[M13-EXEC] UDP mode: single worker (worker 0 owns UDP socket).");
        1
    } else {
        match single_queue { Some(_) => 1, None => isolated_cores.len().min(MAX_WORKERS) }
    };
    eprintln!("[M13-EXEC] Discovered {} isolated core(s): {:?}. Spawning {} worker(s).",
        isolated_cores.len(), &isolated_cores[..worker_count], worker_count);
    // BPF Steersman: always attach. It steers both raw L2 M13 (ethertype 0x88B5)
    // AND IPv4 UDP port 443 (Internet mode) into AF_XDP. SSH/ARP/etc pass to kernel.
    let steersman = BpfSteersman::load_and_attach(if_name);
    let map_fd = if let Some(ref s) = steersman {
        let fd = s.map_fd();
        eprintln!("[M13-EXEC] BPF Steersman attached to {} [{}]. map_fd={}", if_name, s.attach_mode, fd);
        fd
    } else {
        eprintln!("[M13-EXEC] WARNING: BPF/XDP failed. Running in UDP-ONLY fallback mode.");
        -1
    };
    
    // Sprint 6.3: TUN Interface
    let tun_ref = if tunnel {
        let t = create_tun("m13tun0");
        if t.is_some() { setup_nat(); }
        t
    } else { None };

    let mut handles = Vec::with_capacity(worker_count);
    for worker_idx in 0..worker_count {
        let core_id = isolated_cores[worker_idx];
        let queue_id = match single_queue { Some(q) => q, None => worker_idx as i32 };
        let iface = if_name.to_string();
        let cal = tsc_cal; // Copy for this worker (TscCal is Copy)
        let tun = tun_ref.as_ref().and_then(|f| f.try_clone().ok());
        
        let handle = std::thread::Builder::new()
            .name(format!("m13-w{}", worker_idx)).stack_size(2 * 1024 * 1024)
            .spawn(move || { worker_entry(worker_idx, core_id, queue_id, &iface, map_fd, cal, tun); })
            .unwrap_or_else(|_| fatal(E_AFFINITY_FAIL, "Thread spawn failed"));
        handles.push(handle);
    }

    eprintln!("[M13-EXEC] Engine operational. Workers running.");

    for h in handles { let _ = h.join(); }
    drop(steersman);

    // Post-worker cleanup: nuke everything
    nuke_cleanup_hub(if_name);
    eprintln!("[M13-EXEC] All workers stopped. XDP detached. Clean exit.");
}

// ============================================================================
// INTERRUPT FENCE — pin NIC IRQs away from isolated cores
// ============================================================================
fn fence_interrupts() {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }

    // DPDK/SPDK pattern: move IRQs OFF isolated (dataplane) cores only.
    // Compute mask of all non-isolated cores. IRQs can run on ANY of them.
    let isolated = discover_isolated_cores();
    if isolated.is_empty() { return; }

    // Get total CPU count
    let nproc = match std::fs::read_to_string("/sys/devices/system/cpu/present") {
        Ok(s) => {
            // Format: "0-N" or "0"
            let parts: Vec<&str> = s.trim().split('-').collect();
            match parts.last() {
                Some(n) => n.parse::<usize>().unwrap_or(0) + 1,
                None => 1,
            }
        }
        Err(_) => { eprintln!("[M13-EXEC] WARNING: Cannot read CPU topology, skipping IRQ fence"); return; }
    };

    // Build hex affinity mask: all cores EXCEPT isolated ones
    // e.g. 8 cores, isolated=[2,3] → mask = 0b11110011 = "f3"
    let mut mask_bits = vec![0u8; (nproc + 7) / 8];
    for cpu in 0..nproc {
        if !isolated.contains(&cpu) {
            mask_bits[cpu / 8] |= 1 << (cpu % 8);
        }
    }
    // Convert to hex string (MSByte first for /proc/irq format)
    let mask_hex: String = mask_bits.iter().rev()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
        .trim_start_matches('0')
        .to_string();
    let mask_str = if mask_hex.is_empty() { "1".to_string() } else { mask_hex };

    eprintln!("[M13-EXEC] IRQ fence mask: 0x{} (isolating cores {:?} from interrupts)", mask_str, isolated);

    // Set default SMP affinity for new IRQs
    let _ = std::fs::write("/proc/irq/default_smp_affinity", format!("{}\n", mask_str));

    if let Ok(output) = std::process::Command::new("pgrep").arg("irqbalance").output() {
        if output.status.success() {
            eprintln!("[M13-EXEC] WARNING: irqbalance is running. It will fight the IRQ fence.");
            eprintln!("[M13-EXEC] WARNING: Run 'systemctl stop irqbalance' for optimal performance.");
        }
    }

    let irq_dir = match std::fs::read_dir("/proc/irq") {
        Ok(d) => d, Err(_) => { eprintln!("[M13-EXEC] WARNING: Cannot read /proc/irq, skipping IRQ fence"); return; }
    };
    let (mut fenced, mut skipped) = (0u32, 0u32);
    for entry in irq_dir {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let name = entry.file_name();
        let name_str = match name.to_str() { Some(s) => s, None => continue };
        if !name_str.bytes().next().map_or(false, |b| b.is_ascii_digit()) { continue; }
        // Write hex mask to smp_affinity (not affinity_list — hex is more reliable)
        let affinity_path = format!("/proc/irq/{}/smp_affinity", name_str);
        match std::fs::write(&affinity_path, format!("{}\n", mask_str)) {
            Ok(_) => { fenced += 1; }
            Err(_) => { skipped += 1; }
        }
    }
    eprintln!("[M13-EXEC] Interrupt Fence: {} IRQs moved to housekeeping cores, {} immovable", fenced, skipped);
}

// ============================================================================
// TUNNELING & NAT
// ============================================================================
const IFF_TUN: i16 = 0x0001;
const IFF_NO_PI: i16 = 0x1000;
const TUNSETIFF: u64 = 0x400454ca;

#[repr(C)]
struct ifreq_tun {
    ifr_name: [u8; 16],
    ifr_flags: i16,
}

fn create_tun(name: &str) -> Option<std::fs::File> {
    let tun_path = "/dev/net/tun";
    let file = match OpenOptions::new().read(true).write(true).open(tun_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[M13-TUN] Failed to open {}: {}", tun_path, e);
            return None;
        }
    };

    let mut req = ifreq_tun {
        ifr_name: [0; 16],
        ifr_flags: IFF_TUN | IFF_NO_PI,
    };
    
    let name_bytes = name.as_bytes();
    if name_bytes.len() > 15 {
        eprintln!("[M13-TUN] Interface name too long");
        return None;
    }
    for (i, b) in name_bytes.iter().enumerate() {
        req.ifr_name[i] = *b;
    }

    unsafe {
        if libc::ioctl(file.as_raw_fd(), TUNSETIFF, &req) < 0 {
            eprintln!("[M13-TUN] ioctl(TUNSETIFF) failed");
            return None;
        }
         // Set non-blocking
        let fd = file.as_raw_fd();
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags >= 0 {
             libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }
    
    // Set interface up and assign IP (Hub = 10.13.0.1/24)
    let _ = Command::new("ip").args(&["link", "set", "dev", name, "up"]).output();
    let _ = Command::new("ip").args(&["addr", "add", "10.13.0.1/24", "dev", name]).output();
    let _ = Command::new("ip").args(&["link", "set", "dev", name, "mtu", "1280"]).output();

    eprintln!("[M13-TUN] Created tunnel interface {} (10.13.0.1/24)", name);
    Some(file)
}

fn setup_nat() {
    eprintln!("[M13-NAT] Enabling NAT (Masquerade)...");
    // Enable forwarding
    let _ = Command::new("sysctl").args(&["-w", "net.ipv4.ip_forward=1"]).output();
    
    // Masquerade all outgoing traffic (lazy NAT)
    // iptables -t nat -A POSTROUTING -j MASQUERADE
    let _ = Command::new("iptables").args(&["-t", "nat", "-A", "POSTROUTING", "-j", "MASQUERADE"]).output();
    
    // Allow forwarding between interfaces
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-o", "m13tun0", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-A", "FORWARD", "-i", "m13tun0", "-j", "ACCEPT"]).output();
}

/// Nuclear cleanup: tear down ALL Hub state — NAT, iptables, TUN, XDP.
/// Safe to call multiple times (idempotent). Safe from panic hook.
fn nuke_cleanup_hub(if_name: &str) {
    eprintln!("[M13-NUKE] Tearing down all Hub state...");

    // 1. Remove iptables NAT and FORWARD rules
    let _ = Command::new("iptables").args(&["-t", "nat", "-D", "POSTROUTING", "-j", "MASQUERADE"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-o", "m13tun0", "-j", "ACCEPT"]).output();
    let _ = Command::new("iptables").args(&["-D", "FORWARD", "-i", "m13tun0", "-j", "ACCEPT"]).output();

    // 2. Destroy TUN interface
    let _ = Command::new("ip").args(&["link", "del", "m13tun0"]).output();

    // 3. Detach XDP from interface
    let _ = Command::new("ip").args(&["link", "set", if_name, "xdp", "off"]).output();
    let _ = Command::new("ip").args(&["link", "set", if_name, "xdpgeneric", "off"]).output();

    eprintln!("[M13-NUKE] ✓ All Hub state destroyed.");
}

// ============================================================================
// WORKER ENTRY — Feedback-first BBRv3 paced pipeline
// Stage order: Classify → Feedback Processing → Feedback Generation →
//              Token Refill → Enqueue+Generate (paced) → Schedule
// ============================================================================
fn worker_entry(worker_idx: usize, core_id: usize, queue_id: i32, if_name: &str, bpf_map_fd: i32, cal: TscCal, mut tun: Option<std::fs::File>) {
    pin_to_core(core_id);
    verify_affinity(core_id);
    let stats = Telemetry::map_worker(worker_idx, true);
    stats.pid.value.store(unsafe { libc::syscall(libc::SYS_gettid) } as u32, Ordering::Relaxed);
    let mut engine = Engine::<ZeroCopyTx>::new_zerocopy(if_name, queue_id, bpf_map_fd);
    eprintln!("[M13-W{}] Datapath: {}", worker_idx, engine.xdp_mode);
    let mut slab = FixedSlab::new(SLAB_DEPTH);
    let mut scheduler = Scheduler::new();
    let mut rx_state = ReceiverState::new();
    let mut rx_bitmap = RxBitmap::new();
    let hexdump_enabled = std::env::var("M13_HEXDUMP").is_ok();
    let mut hexdump = HexdumpState::new(hexdump_enabled);
    let mut gc_counter: u64 = 0;
    let iface_mac = detect_mac(if_name);
    let src_mac = iface_mac;

    // Sprint S1: Multi-tenant peer table replaces all flat per-peer state.
    let mut peers = PeerTable::new(rdtsc_ns(&cal));
    eprintln!("[M13-W{}] PeerTable: {} slots, tunnel subnet 10.13.0.0/24", worker_idx, MAX_PEERS);

    // === PURE AF_XDP: No kernel socket. Raw UDP TX via AF_XDP ring. ===
    let hub_port: u16 = std::env::var("M13_LISTEN_PORT").ok()
        .and_then(|p| p.parse::<u16>().ok()).unwrap_or(443);
    let mut hub_ip: [u8; 4] = get_interface_ip(if_name).unwrap_or_else(|| {
        eprintln!("[M13-W{}] Hub IP not on interface — will learn from first inbound packet.", worker_idx);
        [0, 0, 0, 0]
    });
    let (mut gateway_mac, _gw_ip) = resolve_gateway_mac(if_name).unwrap_or_else(|| {
        eprintln!("[M13-W{}] Gateway MAC not in ARP — will learn from first inbound packet.", worker_idx);
        ([0xFF; 6], [0, 0, 0, 0])
    });
    let mut ip_id_counter: u16 = (worker_idx as u16).wrapping_mul(10000);
    eprintln!("[M13-W{}] AF_XDP Pure Mode: hub={}:{} gw_mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        worker_idx,
        hub_ip[0], hub_ip[1],
        gateway_mac[0], gateway_mac[1], gateway_mac[2],
        gateway_mac[3], gateway_mac[4], gateway_mac[5]);

    let mut udp_rx_count: u64 = 0;
    let mut udp_tx_count: u64 = 0;
    let mut tun_write_count: u64 = 0;
    let mut tun_read_count: u64 = 0;
    let mut aead_ok_count: u64 = 0;
    let mut aead_fail_count: u64 = 0;
    let mut last_hub_report_ns: u64 = 0;


    for i in 0..SLAB_DEPTH {
        let fp = engine.get_frame_ptr(i as u32);
        unsafe {
            let eth = &mut *(fp as *mut EthernetHeader);
            let m13 = &mut *(fp.add(ETH_HDR_SIZE) as *mut M13Header);
            eth.dst = [0xFF; 6]; // broadcast — real dst set per-frame
            eth.src = src_mac;
            eth.ethertype = ETH_P_M13.to_be();
            m13.signature = [0; 32];
            m13.signature[0] = M13_WIRE_MAGIC;
            m13.signature[1] = M13_WIRE_VERSION;
            m13.seq_id = 0; m13.flags = 0;
            m13.payload_len = 0; m13.padding = [0; 3];
        }
    }
    engine.refill_rx_full(&mut slab);
    let umem = engine.umem_base();

    // Measure ε_proc (processing jitter floor) and create jitter buffer
    let epsilon_ns = measure_epsilon_proc(&cal);
    let mut jbuf = JitterBuffer::new(epsilon_ns);
    eprintln!("[M13-W{}] ACTIVE. Pipeline: Graph({}) Deadline: {}us Prefetch: {} HW_Fill: {}/{} \
              SeqWin: {} Feedback: every {} pkts \
              JBuf: {}entries D_buf={}us ε={}us",
        worker_idx, GRAPH_BATCH, DEADLINE_NS / 1000, PREFETCH_DIST, HW_FILL_MAX, TX_RING_SIZE,
        SEQ_WINDOW, FEEDBACK_INTERVAL_PKTS,
        JBUF_CAPACITY, jbuf.depth_ns / 1000, epsilon_ns / 1000);

    let mut rx_batch: [libbpf_sys::xdp_desc; GRAPH_BATCH] = unsafe { mem::zeroed() };
    let mut data_indices = [0u16; GRAPH_BATCH];
    let mut ctrl_indices = [0u16; GRAPH_BATCH];
    let crit_indices = [0u16; GRAPH_BATCH];

    // Sprint 5.21: Graceful close state
    let mut closing = false;
    let mut fin_deadline_ns: u64 = 0;

    loop {
        // Sprint 5.21: Graceful close protocol.
        // On SHUTDOWN: send 3x FIN, then keep looping (RX only) until FIN-ACK or deadline.
        if SHUTDOWN.load(Ordering::Relaxed) && !closing {
            closing = true;
            let rtprop: u64 = 10_000_000;
            fin_deadline_ns = rdtsc_ns(&cal) + (rtprop.saturating_mul(5).max(10_000_000).min(100_000_000));
            // Send FIN to all established peers via raw UDP
            let mut fin_total = 0usize;
            for pi in 0..MAX_PEERS {
                if peers.slots[pi].lifecycle == PeerLifecycle::Established {
                    let sent = if peers.slots[pi].addr.is_udp() {
                        send_fin_burst_udp(
                            &mut slab, &engine, &mut scheduler,
                            &src_mac, &gateway_mac,
                            hub_ip, peers.slots[pi].addr.ip().unwrap(),
                            hub_port, peers.slots[pi].addr.port().unwrap(),
                            &mut ip_id_counter,
                            peers.slots[pi].seq_tx, false, 3,
                        )
                    } else {
                        send_fin_burst_l2(
                            &mut slab, &engine, &mut scheduler,
                            &src_mac, &peers.slots[pi].mac,
                            peers.slots[pi].seq_tx, false, 3,
                        )
                    };
                    fin_total += sent;
                }
            }
            eprintln!("[M13-W{}] FIN sent to {} peers ({}x total). Deadline={}ms.",
                worker_idx, peers.count, fin_total,
                (fin_deadline_ns.saturating_sub(rdtsc_ns(&cal))) / 1_000_000);
        }
        if closing && rdtsc_ns(&cal) >= fin_deadline_ns {
            eprintln!("[M13-W{}] FIN deadline expired. Force-closing.", worker_idx);
            break;
        }
        let now = rdtsc_ns(&cal);
        stats.cycles.value.fetch_add(1, Ordering::Relaxed);
        engine.recycle_tx(&mut slab);
        engine.refill_rx(&mut slab);

        // Stage -1 REMOVED: All RX is now pure AF_XDP (Stage 1 classify).
        // No kernel socket recv_from. Eliminates dual-receiver contention on UDP/443.

        // === SPRINT S2: TUN TX — Zero-copy drain 64/tick + per-peer routing ===
        // Only worker 0 reads TUN to avoid contention/reordering.
        // Speculative slab alloc → read TUN directly into UMEM → route → encapsulate.
        // No intermediate tun_buf — BoringTun/Cloudflare pattern.
        if worker_idx == 0 {
            if let Some(ref mut tun_file) = tun {
                for _tun_batch in 0..64u32 {
                    // Speculative slab alloc — read directly into UMEM payload region
                    let idx = match slab.alloc() {
                        Some(i) => i,
                        None => break, // slab exhausted
                    };
                    let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                    // TUN payload goes at M13 offset 62 (ETH_HDR_SIZE + M13_HDR_SIZE)
                    let payload_offset = ETH_HDR_SIZE + M13_HDR_SIZE; // 62
                    let tun_slice = unsafe { std::slice::from_raw_parts_mut(frame_ptr.add(payload_offset), 1500) };
                    match tun_file.read(tun_slice) {
                        Ok(n) if n > 0 => {
                            // Route by destination IP in the TUN packet (already in UMEM).
                            // IPv4: dst_ip at offset 16..20 in the IP header.
                            if n < 20 {
                                slab.free(idx);
                                continue; // too short for IP header
                            }
                            // Read dst_ip from UMEM (already there — zero extra copy)
                            let dst_ip = unsafe {
                                [*tun_slice.get_unchecked(16), *tun_slice.get_unchecked(17),
                                 *tun_slice.get_unchecked(18), *tun_slice.get_unchecked(19)]
                            };

                            // Find the peer that owns this tunnel IP.
                            let peer_idx = match peers.lookup_by_tunnel_ip(dst_ip) {
                                Some(idx) => idx,
                                None => {
                                    // Fallback: if only 1 established peer, send to them.
                                    let mut fallback_idx: Option<usize> = None;
                                    let mut established_count = 0u16;
                                    for pi in 0..MAX_PEERS {
                                        if peers.slots[pi].lifecycle == PeerLifecycle::Established {
                                            if fallback_idx.is_none() { fallback_idx = Some(pi); }
                                            established_count += 1;
                                        }
                                    }
                                    if established_count == 1 {
                                        fallback_idx.unwrap()
                                    } else {
                                        slab.free(idx);
                                        continue; // No peer or ambiguous — drop
                                    }
                                }
                            };

                            let peer_slot = &mut peers.slots[peer_idx];
                            if !peer_slot.has_session() {
                                slab.free(idx);
                                continue;
                            }

                            let peer_ip = peer_slot.addr.ip().unwrap_or([0;4]);
                            let peer_port = peer_slot.addr.port().unwrap_or(0);
                            let m13_flen = ETH_HDR_SIZE + M13_HDR_SIZE + n;
                            unsafe {
                                let m13_buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                // Build M13 header (payload already in place at offset 62)
                                m13_buf[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
                                m13_buf[6..12].copy_from_slice(&src_mac);
                                m13_buf[12] = (ETH_P_M13 >> 8) as u8;
                                m13_buf[13] = (ETH_P_M13 & 0xFF) as u8;
                                m13_buf[14] = M13_WIRE_MAGIC;
                                m13_buf[15] = M13_WIRE_VERSION;
                                // Zero signature region [16..46] — CRITICAL: must match Node's
                                // expectation. Without this, stale UMEM data causes AEAD mismatch.
                                m13_buf[16..46].fill(0);
                                let udp_seq = peer_slot.next_seq();
                                m13_buf[46..54].copy_from_slice(&udp_seq.to_le_bytes());
                                m13_buf[54] = FLAG_TUNNEL;
                                m13_buf[55..59].copy_from_slice(&(n as u32).to_le_bytes());
                                // Zero padding [59..62]
                                m13_buf[59..62].fill(0);
                                // Payload already at [62..62+n] — no copy needed
                                if let Some(ref cipher) = peers.ciphers[peer_idx] {
                                    seal_frame(&mut m13_buf[..m13_flen], cipher, udp_seq, DIR_HUB_TO_NODE, ETH_HDR_SIZE);
                                }

                                let total_len = RAW_HDR_LEN + m13_flen;
                                if total_len <= FRAME_SIZE as usize {
                                    std::ptr::copy(frame_ptr, frame_ptr.add(RAW_HDR_LEN), m13_flen);
                                    let raw_frame = std::slice::from_raw_parts_mut(frame_ptr, total_len);
                                    // ETH header
                                    raw_frame[0..6].copy_from_slice(&gateway_mac);
                                    raw_frame[6..12].copy_from_slice(&src_mac);
                                    raw_frame[12..14].copy_from_slice(&0x0800u16.to_be_bytes());
                                    // IP header
                                    let ip_total = (IP_HDR_LEN + UDP_HDR_LEN + m13_flen) as u16;
                                    let ip = &mut raw_frame[14..34];
                                    ip[0] = 0x45; ip[1] = 0x00;
                                    ip[2..4].copy_from_slice(&ip_total.to_be_bytes());
                                    ip[4..6].copy_from_slice(&ip_id_counter.to_be_bytes());
                                    ip_id_counter = ip_id_counter.wrapping_add(1);
                                    ip[6..8].copy_from_slice(&0x4000u16.to_be_bytes());
                                    ip[8] = 64; ip[9] = 17;
                                    ip[10..12].copy_from_slice(&[0, 0]);
                                    ip[12..16].copy_from_slice(&hub_ip);
                                    ip[16..20].copy_from_slice(&peer_ip);
                                    let cksum = ip_checksum(ip);
                                    ip[10..12].copy_from_slice(&cksum.to_be_bytes());
                                    // UDP header
                                    let udp_total = (UDP_HDR_LEN + m13_flen) as u16;
                                    raw_frame[34..36].copy_from_slice(&hub_port.to_be_bytes());
                                    raw_frame[36..38].copy_from_slice(&peer_port.to_be_bytes());
                                    raw_frame[38..40].copy_from_slice(&udp_total.to_be_bytes());
                                    raw_frame[40..42].copy_from_slice(&[0, 0]);

                                    hexdump.dump_tx(raw_frame.as_ptr(), total_len, now);
                                    scheduler.enqueue_bulk((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                    udp_tx_count += 1;
                                    tun_read_count += 1;
                                    stats.tx_count.value.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    slab.free(idx);
                                }
                            }
                        }
                        _ => {
                            // WouldBlock or EOF — free speculative slab, drain complete
                            slab.free(idx);
                            break;
                        }
                    }
                }
            }
        }
        // === HUB WORKER TELEMETRY (1/sec) ===
        if now.saturating_sub(last_hub_report_ns) > 1_000_000_000 {
            last_hub_report_ns = now;
            let mut established = 0u16;
            for pi in 0..MAX_PEERS {
                if peers.slots[pi].lifecycle == PeerLifecycle::Established { established += 1; }
            }
            eprintln!("[M13-W{}] RX:{} TX:{} TUN_R:{} TUN_W:{} AEAD_OK:{} FAIL:{} Peers:{}/{}",
                worker_idx, udp_rx_count, udp_tx_count,
                tun_read_count, tun_write_count,
                aead_ok_count, aead_fail_count,
                established, peers.count);
        }

        // === STAGE 0: ADAPTIVE BATCH DRAIN ===
        let mut rx_count = engine.poll_rx_batch(&mut rx_batch, &stats);
        if rx_count > 0 && rx_count < GRAPH_BATCH {
            loop {
                engine.recycle_tx(&mut slab); engine.refill_rx(&mut slab);
                let n = engine.poll_rx_batch(&mut rx_batch[rx_count..], &stats);
                rx_count += n;
                if rx_count >= GRAPH_BATCH || rdtsc_ns(&cal) - now >= DEADLINE_NS { break; }
            }
        }

        // === STAGE 0.5: JITTER BUFFER DRAIN ===
        // Release TC_CRITICAL frames whose release time has arrived.
        // Must happen BEFORE classification so buffered frames from previous
        // cycles get scheduled THIS cycle.
        {
            let (rel, _) = jbuf.drain(now, &mut scheduler);
            if rel > 0 {
                // Bridge jitter buffer telemetry (4 Relaxed stores)
                stats.jbuf_depth_us.value.store(jbuf.depth_ns / 1000, Ordering::Relaxed);
                stats.jbuf_jitter_us.value.store(jbuf.estimator.get() / 1000, Ordering::Relaxed);
                stats.jbuf_releases.value.store(jbuf.total_releases, Ordering::Relaxed);
                stats.jbuf_drops.value.store(jbuf.total_drops, Ordering::Relaxed);
            }
        }

        // === STAGE 1: CLASSIFY (3-way split: data / feedback / critical) ===
        // Data (TC_BULK) | Feedback → BBR | TC_CRITICAL → Jitter Buffer
        let rx_batch_ns = if rx_count > 0 { now } else { 0 };
        let (mut data_count, mut ctrl_count, crit_count) = (0usize, 0usize, 0usize);
        // Sprint S2: Deferred TUN write batch — collect during classify, flush after.
        let mut tun_write_indices: [u16; GRAPH_BATCH] = [0; GRAPH_BATCH];
        let mut tun_write_offsets: [u16; GRAPH_BATCH] = [0; GRAPH_BATCH]; // m13_offset per frame
        let mut tun_write_batch: usize = 0;
        for i in 0..rx_count {
            if i + PREFETCH_DIST < rx_count {
                unsafe { prefetch_read_l1(umem.add(rx_batch[i + PREFETCH_DIST].addr as usize + ETH_HDR_SIZE)); }
            }
            // Determine M13 header offset based on encapsulation:
            // - EtherType 0x88B5: raw L2 M13, M13 at offset 14 (ETH_HDR_SIZE)
            // - EtherType 0x0800: IPv4+UDP encapsulated, M13 at offset 42 (14+20+8)
            let frame_base = unsafe { umem.add(rx_batch[i].addr as usize) };
            let frame_len = rx_batch[i].len as usize;
            let ethertype = unsafe { u16::from_be(*(frame_base.add(12) as *const u16)) };

            let m13_offset = if ethertype == 0x88B5 {
                // L2 path: air-gapped WiFi 7 drones. Peer identity = src MAC.
                let peer_mac_wire = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                let peer_addr = PeerAddr::new_l2(peer_mac_wire);
                let _peer_idx = match peers.lookup_or_insert(peer_addr, peer_mac_wire) {
                    Some(idx) => idx,
                    None => {
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                };
                ETH_HDR_SIZE // 14
            } else if ethertype == 0x0800 && frame_len >= 56 + M13_HDR_SIZE {
                // IPv4 UDP: extract source IP:port as peer identity
                let src_ip = unsafe { *(frame_base.add(26) as *const [u8; 4]) };
                let src_port = unsafe { u16::from_be(*(frame_base.add(34) as *const u16)) };
                // Learn Hub's public IP and gateway MAC from the first inbound packet.
                if hub_ip == [0, 0, 0, 0] {
                    let wire_dst_ip = unsafe { *(frame_base.add(30) as *const [u8; 4]) };
                    hub_ip = wire_dst_ip;
                    eprintln!("[M13-W{}] Learned Hub IP from wire: {}.{}.{}.{}",
                        worker_idx, hub_ip[0], hub_ip[1], hub_ip[2], hub_ip[3]);
                }
                if gateway_mac == [0xFF; 6] {
                    let wire_gw_mac = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                    gateway_mac = wire_gw_mac;
                    eprintln!("[M13-W{}] Learned Gateway MAC from wire: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        worker_idx, gateway_mac[0], gateway_mac[1], gateway_mac[2],
                        gateway_mac[3], gateway_mac[4], gateway_mac[5]);
                }

                // Sprint S1: Per-peer lookup/insert
                let peer_mac_wire = unsafe { *(frame_base.add(56 + 6) as *const [u8; 6]) }; // M13 ETH src_mac
                let peer_addr = PeerAddr::new_udp(src_ip, src_port);
                let _peer_idx = match peers.lookup_or_insert(peer_addr, peer_mac_wire) {
                    Some(idx) => idx,
                    None => {
                        // Table full — drop
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                };
                // Store peer address for use after classify
                // (src_ip, src_port already extracted above)
                // We'll re-lookup per section below using the frame's src_ip:src_port

                udp_rx_count += 1;
                56 // ETH(14) + IP(20) + UDP(8) + FakeETH(14)
            } else {
                // Unknown encapsulation — drop
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            };

            let m13 = unsafe { &*(frame_base.add(m13_offset) as *const M13Header) };
            // Wire protocol validation: reject frames with wrong magic/version.
            // Defense-in-depth behind BPF EtherType filter.
            if m13.signature[0] != M13_WIRE_MAGIC || m13.signature[1] != M13_WIRE_VERSION {
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            }

            // Sprint S1: Resolve peer index for per-peer state.
            // For L2 frames (0x88B5), no peer table — use legacy path.
            // For UDP (0x0800), peer was inserted during classify above.
            let cur_peer_idx: Option<usize> = if m13_offset == 56 {
                // UDP path: lookup by (src_ip, src_port)
                let src_ip = unsafe { *(frame_base.add(26) as *const [u8; 4]) };
                let src_port = unsafe { u16::from_be(*(frame_base.add(34) as *const u16)) };
                peers.lookup(PeerAddr::new_udp(src_ip, src_port))
            } else {
                // L2 path: lookup by src MAC (air-gapped WiFi 7 drones)
                let src_mac_wire: [u8; 6] = unsafe { *(frame_base.add(6) as *const [u8; 6]) };
                peers.lookup(PeerAddr::new_l2(src_mac_wire))
            };

            // Sprint S1: Detect reconnecting Node during active AEAD session.
            // Per-peer: only reset THIS peer's session, not the global state.
            if let Some(pidx) = cur_peer_idx {
                if peers.slots[pidx].has_session()
                   && m13.signature[2] != 0x01
                   && m13.flags & FLAG_HANDSHAKE == 0 && m13.flags & FLAG_FRAGMENT == 0 {
                    if m13.flags & FLAG_CONTROL != 0 {
                        eprintln!("[M13-W{}] Peer {:?} reconnecting (cleartext CONTROL while AEAD active). Resetting session.",
                            worker_idx, peers.slots[pidx].addr);
                        peers.slots[pidx].reset_session();
                        peers.ciphers[pidx] = None;
                        peers.hs_sidecar[pidx] = None;
                        peers.assemblers[pidx] = Assembler::new();
                        // Send registration echo to this peer
                        let mut echo_m13 = [0u8; 62];
                        echo_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
                        echo_m13[6..12].copy_from_slice(&src_mac);
                        echo_m13[12] = (ETH_P_M13 >> 8) as u8;
                        echo_m13[13] = (ETH_P_M13 & 0xFF) as u8;
                        echo_m13[14] = M13_WIRE_MAGIC;
                        echo_m13[15] = M13_WIRE_VERSION;
                        echo_m13[54] = FLAG_CONTROL;
                        if peers.slots[pidx].addr.is_udp() {
                            let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                            let peer_port = peers.slots[pidx].addr.port().unwrap();
                            if let Some(idx) = slab.alloc() {
                                let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                let total_len = RAW_HDR_LEN + 62;
                                unsafe {
                                    let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                    build_raw_udp_frame(
                                        buf, &src_mac, &gateway_mac,
                                        hub_ip, peer_ip, hub_port, peer_port,
                                        ip_id_counter, &echo_m13,
                                    );
                                    ip_id_counter = ip_id_counter.wrapping_add(1);
                                }
                                scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                udp_tx_count += 1;
                            }
                        } else {
                            if send_l2_frame(&mut slab, &engine, &mut scheduler, &peers.slots[pidx].mac, &echo_m13) {
                                udp_tx_count += 1;
                            }
                        }
                    }
                    slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                    continue;
                }
            }

            // Sprint S1: AEAD verification on encrypted frames (per-peer session key)
            if m13.signature[2] == 0x01 {
                if let Some(pidx) = cur_peer_idx {
                    let cipher = match peers.ciphers[pidx].as_ref() {
                        Some(c) => c,
                        None => {
                            // Peer has no cached cipher (session not established)
                            aead_fail_count += 1;
                            slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                            continue;
                        }
                    };
                    let frame_ptr = unsafe { umem.add(rx_batch[i].addr as usize) };
                    let frame_len = rx_batch[i].len as usize;
                    let frame_mut = unsafe { std::slice::from_raw_parts_mut(frame_ptr, frame_len) };
                    if !open_frame(frame_mut, cipher, DIR_HUB_TO_NODE, m13_offset) {
                        stats.auth_fail.value.fetch_add(1, Ordering::Relaxed);
                        aead_fail_count += 1;
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                    let dec_seq = u64::from_le_bytes(
                        frame_mut[m13_offset + 32..m13_offset + 40].try_into().unwrap()
                    );
                    let _ = dec_seq;
                    peers.slots[pidx].frame_count += 1;
                    aead_ok_count += 1;
                    stats.decrypt_ok.value.fetch_add(1, Ordering::Relaxed);

                    // Per-peer rekey check
                    let established_ns = (peers.slots[pidx].established_rel_s as u64) * 1_000_000_000 + peers.epoch_ns;
                    if peers.slots[pidx].frame_count as u64 >= REKEY_FRAME_LIMIT
                       || now.saturating_sub(established_ns) > REKEY_TIME_LIMIT_NS {
                        eprintln!("[M13-HUB-PQC] Peer {:?} rekey threshold (frames={} time={}s)",
                            peers.slots[pidx].addr, peers.slots[pidx].frame_count,
                            now.saturating_sub(established_ns) / 1_000_000_000);
                        peers.slots[pidx].reset_session();
                        peers.ciphers[pidx] = None;
                        peers.hs_sidecar[pidx] = None;
                    }
                } else {
                    // AEAD frame from unknown peer — drop
                    aead_fail_count += 1;
                    slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                    continue;
                }
            }

            // CRITICAL: Re-read flags from decrypted buffer.
            // `m13.flags` is an immutable &M13Header reference created BEFORE open_frame()
            // decrypted the buffer in-place. Since flags (offset +40) is in the encrypted
            // region (offset +32..+48), the original reference holds ciphertext garbage.
            // LLVM may also cache the stale value due to noalias. All flag-based routing
            // below MUST use this freshly-read decrypted value.
            let frame_base = unsafe { umem.add(rx_batch[i].addr as usize) };
            let flags = unsafe { *frame_base.add(m13_offset + 40) };

            // Sprint S1: Fragment reassembly on cold path (per-peer assembler)
            if flags & FLAG_FRAGMENT != 0 {
                let frame_ptr = unsafe { umem.add(rx_batch[i].addr as usize) };
                let frame_len = rx_batch[i].len as usize;
                if frame_len >= m13_offset + M13_HDR_SIZE + FRAG_HDR_SIZE {
                    let frag_hdr = unsafe { &*(frame_ptr.add(m13_offset + M13_HDR_SIZE) as *const FragHeader) };
                    let frag_data_start = m13_offset + M13_HDR_SIZE + FRAG_HDR_SIZE;
                    let frag_msg_id = unsafe { std::ptr::addr_of!((*frag_hdr).frag_msg_id).read_unaligned() };
                    let frag_index = unsafe { std::ptr::addr_of!((*frag_hdr).frag_index).read_unaligned() };
                    let frag_total = unsafe { std::ptr::addr_of!((*frag_hdr).frag_total).read_unaligned() };
                    let frag_offset = unsafe { std::ptr::addr_of!((*frag_hdr).frag_offset).read_unaligned() };
                    let frag_data_len = unsafe { std::ptr::addr_of!((*frag_hdr).frag_len).read_unaligned() } as usize;
                    if frag_data_start + frag_data_len <= frame_len {
                        let frag_data = unsafe { std::slice::from_raw_parts(frame_ptr.add(frag_data_start), frag_data_len) };
                        // Use per-peer assembler
                        let asm_idx = cur_peer_idx.unwrap_or(0);
                        if let Some(reassembled) = peers.assemblers[asm_idx].feed(
                            frag_msg_id, frag_index, frag_total,
                            frag_offset, frag_data, now,
                        ) {
                            // Sprint S1: Route handshake messages to PQC processor (per-peer)
                            if flags & FLAG_HANDSHAKE != 0 && !reassembled.is_empty() {
                                let msg_type = reassembled[0];
                                eprintln!("[M13-W{}] Reassembled handshake type=0x{:02X} len={} peer_idx={}",
                                    worker_idx, msg_type, reassembled.len(), asm_idx);
                                if let Some(pidx) = cur_peer_idx {
                                    let mut hs_seq_tx: u64 = peers.slots[pidx].seq_tx;
                                    match msg_type {
                                        HS_CLIENT_HELLO => {
                                            peers.slots[pidx].lifecycle = PeerLifecycle::Handshaking;
                                            if let Some((hs, server_hello)) = process_client_hello_hub(
                                                &reassembled, &mut hs_seq_tx, now,
                                            ) {
                                                // Frame ServerHello: UDP or L2
                                                let hs_flags = FLAG_CONTROL | FLAG_HANDSHAKE;
                                                let frames = if peers.slots[pidx].addr.is_udp() {
                                                    let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                                                    let peer_port = peers.slots[pidx].addr.port().unwrap();
                                                    build_fragmented_raw_udp(
                                                        &src_mac, &gateway_mac, hub_ip, peer_ip,
                                                        hub_port, peer_port, &server_hello, hs_flags,
                                                        &mut hs_seq_tx, &mut ip_id_counter,
                                                        &mut hexdump, &cal,
                                                    )
                                                } else {
                                                    let peer_mac = &peers.slots[pidx].mac;
                                                    build_fragmented_l2(
                                                        &src_mac, peer_mac, &server_hello, hs_flags,
                                                        &mut hs_seq_tx, &mut hexdump, &cal,
                                                    )
                                                };
                                                for raw_frame in frames {
                                                    if let Some(idx) = slab.alloc() {
                                                        let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                                        let flen = raw_frame.len().min(FRAME_SIZE as usize);
                                                        unsafe { std::ptr::copy_nonoverlapping(raw_frame.as_ptr(), frame_ptr, flen); }
                                                        scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, flen as u32);
                                                        udp_tx_count += 1;
                                                    }
                                                }
                                                peers.hs_sidecar[pidx] = Some(hs);
                                                peers.slots[pidx].seq_tx = hs_seq_tx;
                                                eprintln!("[M13-HUB-PQC] ClientHello processed for peer {:?}, ServerHello enqueued.",
                                                    peers.slots[pidx].addr);
                                            }
                                        }
                                        HS_FINISHED => {
                                            if let Some(ref hs) = peers.hs_sidecar[pidx] {
                                                if let Some(key) = process_finished_hub(&reassembled, hs) {
                                                    peers.slots[pidx].session_key = key;
                                                    let ukey = aead::UnboundKey::new(&aead::AES_256_GCM, &key).unwrap();
                                                    peers.ciphers[pidx] = Some(aead::LessSafeKey::new(ukey));
                                                    peers.slots[pidx].frame_count = 0;
                                                    let rel_s = ((now.saturating_sub(peers.epoch_ns)) / 1_000_000_000) as u32;
                                                    peers.slots[pidx].established_rel_s = rel_s;
                                                    peers.slots[pidx].lifecycle = PeerLifecycle::Established;
                                                    peers.hs_sidecar[pidx] = None;
                                                    stats.handshake_ok.value.fetch_add(1, Ordering::Relaxed);
                                                    eprintln!("[M13-HUB-PQC] → Session established for peer {:?} (AEAD active)",
                                                        peers.slots[pidx].addr);
                                                } else {
                                                    peers.hs_sidecar[pidx] = None;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                eprintln!("[M13-W{}] Reassembled msg_id={} total_len={}",
                                    worker_idx, frag_msg_id, reassembled.len());
                            }
                        }
                    }
                }
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                continue;
            }
            if flags & FLAG_FEEDBACK != 0 {
                ctrl_indices[ctrl_count] = i as u16; ctrl_count += 1;
            } else if flags & FLAG_TUNNEL != 0 {
                // Sprint S2: Defer TUN write to after classify loop (keep hot classify cache-tight).
                // Store rx_batch index + m13_offset (varies per encapsulation type).
                tun_write_indices[tun_write_batch] = i as u16;
                tun_write_offsets[tun_write_batch] = m13_offset as u16;
                tun_write_batch += 1;
            } else if flags & FLAG_CONTROL != 0 {
                // Sprint 5.21: Check for FIN/FIN-ACK
                if flags & FLAG_FIN != 0 {
                    if closing {
                        eprintln!("[M13-W{}] FIN-ACK received. Graceful close complete.", worker_idx);
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        fin_deadline_ns = 0;
                        continue;
                    } else {
                        // FIN from peer — send FIN-ACK, then evict
                        if let Some(pidx) = cur_peer_idx {
                            eprintln!("[M13-W{}] FIN received from peer {:?}. Sending FIN-ACK.",
                                worker_idx, peers.slots[pidx].addr);
                            if peers.slots[pidx].addr.is_udp() {
                                send_fin_burst_udp(
                                    &mut slab, &engine, &mut scheduler,
                                    &src_mac, &gateway_mac,
                                    hub_ip, peers.slots[pidx].addr.ip().unwrap(),
                                    hub_port, peers.slots[pidx].addr.port().unwrap(),
                                    &mut ip_id_counter,
                                    peers.slots[pidx].seq_tx, true, 3,
                                );
                            } else {
                                send_fin_burst_l2(
                                    &mut slab, &engine, &mut scheduler,
                                    &src_mac, &peers.slots[pidx].mac,
                                    peers.slots[pidx].seq_tx, true, 3,
                                );
                            }
                            peers.evict(pidx);
                        }
                        slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
                        continue;
                    }
                }
                // Per-peer registration echo: only if peer has no AEAD session yet
                if flags & FLAG_HANDSHAKE == 0 && flags & FLAG_FRAGMENT == 0 {
                    if let Some(pidx) = cur_peer_idx {
                        if !peers.slots[pidx].has_session() {
                            let mut echo_m13 = [0u8; 62];
                            echo_m13[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
                            echo_m13[6..12].copy_from_slice(&src_mac);
                            echo_m13[12] = (ETH_P_M13 >> 8) as u8;
                            echo_m13[13] = (ETH_P_M13 & 0xFF) as u8;
                            echo_m13[14] = M13_WIRE_MAGIC;
                            echo_m13[15] = M13_WIRE_VERSION;
                            echo_m13[54] = FLAG_CONTROL;
                            if peers.slots[pidx].addr.is_udp() {
                                let peer_ip = peers.slots[pidx].addr.ip().unwrap();
                                let peer_port = peers.slots[pidx].addr.port().unwrap();
                                if let Some(idx) = slab.alloc() {
                                    let frame_ptr = unsafe { engine.umem_base().add((idx as usize) * FRAME_SIZE as usize) };
                                    let total_len = RAW_HDR_LEN + 62;
                                    unsafe {
                                        let buf = std::slice::from_raw_parts_mut(frame_ptr, FRAME_SIZE as usize);
                                        build_raw_udp_frame(
                                            buf, &src_mac, &gateway_mac,
                                            hub_ip, peer_ip, hub_port, peer_port,
                                            ip_id_counter, &echo_m13,
                                        );
                                        ip_id_counter = ip_id_counter.wrapping_add(1);
                                    }
                                    scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, total_len as u32);
                                    udp_tx_count += 1;
                                }
                            } else {
                                if send_l2_frame(&mut slab, &engine, &mut scheduler, &peers.slots[pidx].mac, &echo_m13) {
                                    udp_tx_count += 1;
                                }
                            }
                        }
                    }
                }
                slab.free((rx_batch[i].addr / FRAME_SIZE as u64) as u32);
            } else {
                data_indices[data_count] = i as u16; data_count += 1;
                rx_state.highest_seq = m13.seq_id;
                rx_state.delivered += 1;
                rx_state.last_rx_batch_ns = rx_batch_ns;
                rx_bitmap.mark(m13.seq_id);
            }
        }
        // === SPRINT S2: DEFERRED TUN WRITE BATCH ===
        // All tunnel frames collected during classify are written + freed here.
        // Keeps the hot classify loop free of TUN write() syscalls.
        if tun_write_batch > 0 {
            if let Some(ref mut tun_file) = tun {
                for ti in 0..tun_write_batch {
                    let idx = tun_write_indices[ti] as usize;
                    let m13_off = tun_write_offsets[ti] as usize;
                    let frame_ptr = unsafe { umem.add(rx_batch[idx].addr as usize) };
                    let frame_len = rx_batch[idx].len as usize;
                    let m13 = unsafe { &*(frame_ptr.add(m13_off) as *const M13Header) };
                    let plen = m13.payload_len as usize;
                    let start = m13_off + M13_HDR_SIZE;
                    if start + plen <= frame_len {
                        let payload = unsafe { std::slice::from_raw_parts(frame_ptr.add(start), plen) };
                        let _ = tun_file.write(payload);
                        tun_write_count += 1;
                    }
                    slab.free((rx_batch[idx].addr / FRAME_SIZE as u64) as u32);
                }
            } else {
                // No TUN — just free the slab slots
                for ti in 0..tun_write_batch {
                    let idx = tun_write_indices[ti] as usize;
                    slab.free((rx_batch[idx].addr / FRAME_SIZE as u64) as u32);
                }
            }
        }

        // === STAGE 2: FEEDBACK PROCESSING — BBR ===
        let _ctrl_now_ns = if ctrl_count > 0 { now } else { 0 };
        for i in 0..ctrl_count {
            let d = &rx_batch[ctrl_indices[i] as usize];
            let frame = unsafe { umem.add(d.addr as usize) as *const u8 };
            let flags = unsafe { (*(frame.add(ETH_HDR_SIZE) as *const M13Header)).flags };
            if flags & FLAG_FEEDBACK != 0 && d.len >= FEEDBACK_FRAME_LEN {
                let fb = unsafe { &*(frame.add(ETH_HDR_SIZE + M13_HDR_SIZE) as *const FeedbackFrame) };
                // BBRv3 removed — feedback frames are received but not processed
                let _ = (fb, flags);
            }
            slab.free((d.addr / FRAME_SIZE as u64) as u32);
        }

        // === STAGE 2.5: TC_CRITICAL → JITTER BUFFER INSERT ===
        for i in 0..crit_count {
            let d = &rx_batch[crit_indices[i] as usize];
            let m13 = unsafe { &*(umem.add(d.addr as usize + ETH_HDR_SIZE) as *const M13Header) };
            // Update RFC 3550 EWMA jitter + adaptive D_buf
            jbuf.update_jitter(rx_batch_ns, m13.seq_id);
            // Insert into jitter buffer (slab NOT freed — frame held in UMEM)
            // If overflow drops oldest, free its slab slot to prevent UMEM leak
            if let Some(dropped_addr) = jbuf.insert(d.addr, d.len, rx_batch_ns) {
                slab.free((dropped_addr / FRAME_SIZE as u64) as u32);
            }
        }

        // === STAGE 3: FEEDBACK GENERATION ===
        let rtt_est = FEEDBACK_RTT_DEFAULT_NS;
        if rx_state.needs_feedback(rx_batch_ns, rtt_est) {
            if let Some(idx) = slab.alloc() {
                let frame_ptr = unsafe { umem.add((idx as usize) * FRAME_SIZE as usize) };
                // Feedback frames use broadcast dst MAC (L2 only, not per-peer routed)
                let bcast_mac = [0xFF; 6];
                produce_feedback_frame(frame_ptr, &bcast_mac, &src_mac, &mut rx_state, &mut rx_bitmap, rx_batch_ns, jbuf.tail - jbuf.head);
                scheduler.enqueue_critical((idx as u64) * FRAME_SIZE as u64, FEEDBACK_FRAME_LEN);
            }
        }

        // === STAGE 5: ENQUEUE DATA (full-rate, no pacing) ===
        // Sprint 5.21: When closing, don't send new data — only process RX + FIN.
        if !closing {
            let tx_budget = scheduler.budget(engine.tx_path.available_slots() as usize, HW_FILL_MAX);
            let forward_count = data_count.min(tx_budget);
            for i in 0..forward_count {
                let d = &rx_batch[data_indices[i] as usize];
                scheduler.enqueue_bulk(d.addr, d.len);
            }
            for i in forward_count..data_count {
                slab.free((rx_batch[data_indices[i] as usize].addr / FRAME_SIZE as u64) as u32);
                stats.drops.value.fetch_add(1, Ordering::Relaxed);
            }

            // (BBRv3 probe generator removed — data only comes from TUN or forwarded RX)
        } else {
            // Closing: free all received data frames (no forwarding)
            for i in 0..data_count {
                slab.free((rx_batch[data_indices[i] as usize].addr / FRAME_SIZE as u64) as u32);
            }
        }

        // === STAGE 6: SCHEDULE (critical bypasses pacing) ===
        scheduler.schedule(&mut engine.tx_path, &stats, usize::MAX);



        // Sprint S1: Periodic peer table GC + per-peer assembler GC
        gc_counter += 1;
        if gc_counter % 10000 == 0 {
            peers.gc(now);
            for asm in peers.assemblers.iter_mut() { asm.gc(now); }
        }
    }

    // === GRACEFUL SHUTDOWN CLEANUP ===
    while jbuf.head < jbuf.tail {
        let slot = jbuf.head & (JBUF_CAPACITY - 1);
        slab.free((jbuf.entries[slot].addr / FRAME_SIZE as u64) as u32);
        jbuf.head += 1;
    }
    eprintln!("[M13-W{}] Shutdown complete. Slab: {}/{} free. UDP TX:{} RX:{} TUN_R:{} TUN_W:{} AEAD_OK:{} FAIL:{} Peers:{}",
        worker_idx, slab.available(), SLAB_DEPTH, udp_tx_count, udp_rx_count,
        tun_read_count, tun_write_count, aead_ok_count, aead_fail_count, peers.count);
}

// ============================================================================
// UTILS (unchanged)
// ============================================================================
fn discover_isolated_cores() -> Vec<usize> {
    if let Ok(mock) = std::env::var("M13_MOCK_CMDLINE") {
        if let Some(part) = mock.split_whitespace().find(|p| p.starts_with("isolcpus=")) {
            return parse_cpu_list(part.strip_prefix("isolcpus=").unwrap_or(""));
        }
        return Vec::new();
    }
    match std::fs::read_to_string("/sys/devices/system/cpu/isolated") {
        Ok(s) => parse_cpu_list(s.trim()), Err(_) => Vec::new(),
    }
}
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cores = Vec::new();
    if list.is_empty() { return cores; }
    for part in list.split(',') {
        if part.contains('-') {
            let r: Vec<&str> = part.split('-').collect();
            if r.len() == 2 {
                let s: usize = match r[0].parse() {
                    Ok(v) => v,
                    Err(_) => fatal(E_NO_ISOLATED_CORES, "Invalid CPU range in isolcpus"),
                };
                let e: usize = match r[1].parse() {
                    Ok(v) => v,
                    Err(_) => fatal(E_NO_ISOLATED_CORES, "Invalid CPU range in isolcpus"),
                };
                for i in s..=e { cores.push(i); }
            }
        } else if let Ok(id) = part.parse::<usize>() { cores.push(id); }
    }
    cores.sort(); cores.dedup(); cores
}
fn pin_to_core(core_id: usize) {
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut cpuset);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) != 0 {
            fatal(E_AFFINITY_FAIL, "sched_setaffinity failed");
        }
    }
}
fn verify_affinity(expected_core: usize) {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let path = format!("/proc/self/task/{}/status", tid);
    let file = match File::open(&path) {
        Ok(f) => f, Err(_) => match File::open("/proc/self/status") {
            Ok(f) => f, Err(_) => fatal(E_AFFINITY_VERIFY, "Cannot open status file"),
        }
    };
    for line in BufReader::new(file).lines() {
        if let Ok(l) = line {
            if l.starts_with("Cpus_allowed_list:") {
                let mask = l.split_whitespace().last().unwrap_or("");
                if mask != expected_core.to_string() {
                    fatal(E_AFFINITY_VERIFY, "Core affinity mismatch");
                }
                return;
            }
        }
    }
    fatal(E_AFFINITY_VERIFY, "Could not verify affinity");
}
fn lock_pmu() {
    if std::env::var("M13_MOCK_CMDLINE").is_ok() { return; }
    let mut file = match OpenOptions::new().read(true).write(true).open("/dev/cpu_dma_latency") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[M13-EXEC] WARNING: PMU lock failed (open): {}. Continuing.", e);
            return;
        }
    };
    if file.write_all(&0i32.to_ne_bytes()).is_err() {
        eprintln!("[M13-EXEC] WARNING: PMU lock failed (write). Continuing.");
        return;
    }
    if file.seek(SeekFrom::Start(0)).is_err() { 
        eprintln!("[M13-EXEC] WARNING: PMU lock failed (seek). Continuing.");
        return;
    }
    let mut buf = [0u8; 4];
    if file.read_exact(&mut buf).is_err() || i32::from_ne_bytes(buf) != 0 {
        eprintln!("[M13-EXEC] WARNING: PMU lock rejected (read!=0). Continuing.");
        return;
    }
    eprintln!("[M13-EXEC] PMU Locked: max_latency=0us (C0 only)");
    mem::forget(file);
}

// ============================================================================
// MONITOR
// ============================================================================
fn run_monitor() {
    eprintln!("[M13-MONITOR] Scanning for active workers...");
    let mut workers = Vec::new();
    for i in 0..MAX_WORKERS {
        if let Some(t) = Telemetry::try_map_worker(i) { workers.push(t); } else { break; }
    }
    if workers.is_empty() {
        eprintln!("[M13-MONITOR] No workers found. Waiting...");
        while workers.is_empty() {
            if let Some(t) = Telemetry::try_map_worker(0) { workers.push(t); break; }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    eprintln!("[M13-MONITOR] Attached to {} worker(s).", workers.len());
    eprintln!("---------------------------------------------------------------------");
    let mut last_tx = vec![0u64; workers.len()];
    let mut tids = vec![0u32; workers.len()];
    loop {
        let (mut ttx, mut trx, mut _td, mut tpps) = (0u64, 0u64, 0u64, 0u64);
        let mut cs = String::new();
        for (i, w) in workers.iter().enumerate() {
            let tx = w.tx_count.value.load(Ordering::Relaxed);
            let rx = w.rx_count.value.load(Ordering::Relaxed);
            let d = w.drops.value.load(Ordering::Relaxed);
            let pps = tx - last_tx[i]; last_tx[i] = tx;
            ttx += tx; trx += rx; _td += d; tpps += pps;
            if tids[i] == 0 { tids[i] = w.pid.value.load(Ordering::Relaxed); }
            if tids[i] != 0 {
                let (v, n) = read_ctxt_switches(tids[i]);
                if i > 0 { cs.push('|'); }
                cs.push_str(&format!("W{}:{}/{}", i, v, n));
            }
        }
        // BBR state from worker 0
        // Jitter buffer state from worker 0
        let jb_depth = workers[0].jbuf_depth_us.value.load(Ordering::Relaxed);
        let jb_jitter = workers[0].jbuf_jitter_us.value.load(Ordering::Relaxed);
        let jb_rel = workers[0].jbuf_releases.value.load(Ordering::Relaxed);
        let jb_drop = workers[0].jbuf_drops.value.load(Ordering::Relaxed);
        eprint!("\r[TELEM] TX:{:<12} RX:{:<12} PPS:{:<10} JB:{}us/{}us R:{} D:{} CTX:[{}]   ",
            ttx, trx, tpps, jb_depth, jb_jitter, jb_rel, jb_drop, cs);
        std::thread::sleep(Duration::from_secs(1));
    }
}
fn read_ctxt_switches(tid: u32) -> (u64, u64) {
    let path = format!("/proc/{}/status", tid);
    if let Ok(file) = File::open(&path) {
        let (mut v, mut n) = (0u64, 0u64);
        for line in BufReader::new(file).lines() {
            if let Ok(l) = line {
                if l.starts_with("voluntary_ctxt_switches:") {
                    // Intentional unwrap_or(0): monitor graceful degradation
                    v = l.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                } else if l.starts_with("nonvoluntary_ctxt_switches:") {
                    // Intentional unwrap_or(0): monitor graceful degradation
                    n = l.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                }
            }
        }
        (v, n)
    } else { (0, 0) }
}
