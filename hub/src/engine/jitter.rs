// M13 HUB — DETERMINISTIC JITTER BUFFER
// RFC 3550 EWMA jitter estimator + fixed-size circular buffer.
// D_buf adaptive: k * J(ewma) + ε_proc, clamped [1ms, 100ms].

use crate::engine::scheduler::Scheduler;
use crate::engine::clock::{TscCal, rdtsc_ns};

pub const JBUF_CAPACITY: usize = 64;
const JBUF_K: u64 = 4;
const JBUF_MIN_DEPTH_NS: u64 = 1_000_000;
const JBUF_MAX_DEPTH_NS: u64 = 100_000_000;
const JBUF_DEFAULT_DEPTH_NS: u64 = 50_000_000;
const JBUF_EWMA_GAIN_SHIFT: u32 = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct JitterEntry { pub addr: u64, pub len: u32, pub release_ns: u64 }

pub struct JitterEstimator {
    last_rx_ns: u64, last_seq: u64, jitter_ns: u64, seq_interval_ns: u64,
}

impl JitterEstimator {
    pub fn new() -> Self {
        JitterEstimator { last_rx_ns: 0, last_seq: 0, jitter_ns: 0, seq_interval_ns: 0 }
    }
    #[inline(always)]
    pub fn update(&mut self, rx_ns: u64, seq: u64) {
        if self.last_rx_ns == 0 { self.last_rx_ns = rx_ns; self.last_seq = seq; return; }
        let rx_delta = rx_ns.saturating_sub(self.last_rx_ns);
        let seq_delta = seq.saturating_sub(self.last_seq);
        let send_delta = if self.seq_interval_ns > 0 && seq_delta > 0 {
            seq_delta * self.seq_interval_ns
        } else { rx_delta };
        let d = if rx_delta > send_delta { rx_delta - send_delta } else { send_delta - rx_delta };
        if d > self.jitter_ns {
            self.jitter_ns += (d - self.jitter_ns) >> JBUF_EWMA_GAIN_SHIFT;
        } else {
            self.jitter_ns -= (self.jitter_ns - d) >> JBUF_EWMA_GAIN_SHIFT;
        }
        self.last_rx_ns = rx_ns; self.last_seq = seq;
    }
    #[inline(always)]
    pub fn get(&self) -> u64 { self.jitter_ns }
}

#[repr(align(128))]
pub struct JitterBuffer {
    pub entries: [JitterEntry; JBUF_CAPACITY],
    pub head: usize, pub tail: usize,
    pub depth_ns: u64, pub epsilon_ns: u64,
    pub estimator: JitterEstimator,
    pub total_releases: u64, pub total_drops: u64,
}

impl JitterBuffer {
    pub fn new(epsilon_ns: u64) -> Self {
        JitterBuffer {
            entries: [JitterEntry { addr: 0, len: 0, release_ns: 0 }; JBUF_CAPACITY],
            head: 0, tail: 0, depth_ns: JBUF_DEFAULT_DEPTH_NS, epsilon_ns,
            estimator: JitterEstimator::new(), total_releases: 0, total_drops: 0,
        }
    }
    #[inline(always)]
    pub fn update_jitter(&mut self, rx_ns: u64, seq: u64) {
        self.estimator.update(rx_ns, seq);
        let j = self.estimator.get();
        if j > 0 { self.depth_ns = (JBUF_K * j + self.epsilon_ns).max(JBUF_MIN_DEPTH_NS).min(JBUF_MAX_DEPTH_NS); }
    }
    #[inline(always)]
    pub fn insert(&mut self, addr: u64, len: u32, rx_ns: u64) -> Option<u64> {
        let release_ns = rx_ns + self.depth_ns;
        let mut dropped_addr = None;
        if self.tail - self.head >= JBUF_CAPACITY {
            let old_slot = self.head & (JBUF_CAPACITY - 1);
            dropped_addr = Some(self.entries[old_slot].addr);
            self.head += 1; self.total_drops += 1;
        }
        let slot = self.tail & (JBUF_CAPACITY - 1);
        self.entries[slot] = JitterEntry { addr, len, release_ns };
        self.tail += 1;
        dropped_addr
    }
    #[inline(always)]
    pub fn drain(&mut self, now_ns: u64, scheduler: &mut Scheduler) -> (usize, usize) {
        let mut released = 0usize;
        while self.head < self.tail {
            let slot = self.head & (JBUF_CAPACITY - 1);
            let e = &self.entries[slot];
            if now_ns >= e.release_ns {
                scheduler.enqueue_critical(e.addr, e.len);
                self.head += 1; released += 1; self.total_releases += 1;
            } else { break; }
        }
        (released, 0)
    }
    pub fn len(&self) -> usize { self.tail - self.head }
}

pub fn measure_epsilon_proc(cal: &TscCal) -> u64 {
    let mut max_delta = 0u64;
    for _ in 0..10_000 {
        let t0 = rdtsc_ns(cal); let t1 = rdtsc_ns(cal);
        let delta = t1.saturating_sub(t0);
        if delta > max_delta { max_delta = delta; }
    }
    max_delta * 2
}
