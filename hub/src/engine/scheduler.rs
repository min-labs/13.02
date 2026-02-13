// M13 HUB — ISOCHRONOUS SCHEDULER
// Strict priority: critical (feedback) bypasses pacing, bulk (data) pacing-limited.
// BBR cwnd-aware budget calculation.

use std::sync::atomic::{AtomicU64, Ordering};

pub const TX_RING_SIZE: usize = 2048;
pub const HW_FILL_MAX: usize = TX_RING_SIZE / 2;

/// Trait for AF_XDP TX path operations.
/// Concrete implementation is ZeroCopyTx in datapath.rs.
pub trait TxSubmit {
    fn available_slots(&mut self) -> u32;
    fn stage_tx_addr(&mut self, addr: u64, len: u32);
    fn commit_tx(&mut self);
    fn kick_tx(&mut self);
}

/// Atomic counter for telemetry (decoupled from datapath::Telemetry).
pub struct TxCounter {
    pub value: AtomicU64,
}

impl TxCounter {
    pub fn new() -> Self { TxCounter { value: AtomicU64::new(0) } }
}

#[derive(Copy, Clone)]
pub struct TxDesc { pub addr: u64, pub len: u32 }

pub struct Scheduler {
    critical: [TxDesc; 256],
    critical_len: usize,
    bulk: [TxDesc; 512],
    bulk_len: usize,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            critical: [TxDesc { addr: 0, len: 0 }; 256], critical_len: 0,
            bulk: [TxDesc { addr: 0, len: 0 }; 512], bulk_len: 0,
        }
    }
    #[inline(always)]
    pub fn budget(&self, tx_avail: usize, cwnd: usize) -> usize {
        let inflight = TX_RING_SIZE.saturating_sub(tx_avail);
        let cap = cwnd.min(HW_FILL_MAX);
        cap.saturating_sub(inflight).min(tx_avail)
    }
    #[inline(always)]
    pub fn enqueue_critical(&mut self, addr: u64, len: u32) {
        if self.critical_len < self.critical.len() {
            self.critical[self.critical_len] = TxDesc { addr, len };
            self.critical_len += 1;
        }
    }
    #[inline(always)]
    pub fn enqueue_bulk(&mut self, addr: u64, len: u32) {
        if self.bulk_len < self.bulk.len() {
            self.bulk[self.bulk_len] = TxDesc { addr, len };
            self.bulk_len += 1;
        }
    }
    /// Schedule: critical bypasses pacing (strict priority), bulk capped by bulk_limit.
    pub fn schedule(&mut self, tx_path: &mut impl TxSubmit, tx_count: &TxCounter,
                bulk_limit: usize) -> usize {
        let avail = tx_path.available_slots() as usize;
        let hw_budget = {
            let inflight = TX_RING_SIZE.saturating_sub(avail);
            HW_FILL_MAX.saturating_sub(inflight).min(avail)
        };
        let mut submitted = 0usize;
        let crit = self.critical_len.min(hw_budget);
        for i in 0..crit {
            tx_path.stage_tx_addr(self.critical[i].addr, self.critical[i].len);
            submitted += 1;
        }
        let bulk_hw = hw_budget.saturating_sub(submitted);
        let bulk = self.bulk_len.min(bulk_hw).min(bulk_limit);
        for i in 0..bulk {
            tx_path.stage_tx_addr(self.bulk[i].addr, self.bulk[i].len);
            submitted += 1;
        }
        if submitted > 0 {
            tx_path.commit_tx();
            tx_path.kick_tx();
            tx_count.value.fetch_add(submitted as u64, Ordering::Relaxed);
        }
        self.critical_len = 0;
        self.bulk_len = 0;
        submitted
    }
}
