// M13 HUB — RECEIVER STATE
// RxBitmap: 1024-bit sliding window loss detector (O(1) mark, O(words) advance).

pub const FEEDBACK_INTERVAL_PKTS: u32 = 32;
pub const FEEDBACK_RTT_DEFAULT_NS: u64 = 10_000_000;

/// 1024-bit sliding window bitmap for sequence gap detection.
pub struct RxBitmap {
    bits: [u64; 16],
    base_seq: u64,
    loss_accum: u32,
    highest_marked: u64,
}

impl RxBitmap {
    pub fn new() -> Self {
        RxBitmap { bits: [0u64; 16], base_seq: 0, loss_accum: 0, highest_marked: 0 }
    }

    #[inline(always)]
    pub fn mark(&mut self, seq: u64) {
        if seq < self.base_seq { return; }
        let offset = seq - self.base_seq;
        if offset >= 1024 { self.advance_to(seq); }
        let offset = (seq - self.base_seq) as usize;
        if offset < 1024 {
            let word = offset >> 6;
            let bit = offset & 63;
            self.bits[word] |= 1u64 << bit;
        }
        if seq > self.highest_marked { self.highest_marked = seq; }
    }

    fn advance_to(&mut self, seq: u64) {
        let target_base = seq.saturating_sub(1023);
        if target_base <= self.base_seq { return; }
        let bit_shift = target_base - self.base_seq;
        let word_shift = (bit_shift / 64) as usize;

        if word_shift >= 16 {
            for w in 0..16 {
                let word_base = self.base_seq + (w as u64 * 64);
                if word_base < self.highest_marked {
                    let relevant_bits = if word_base + 64 <= self.highest_marked { 64 }
                    else { (self.highest_marked - word_base) as u32 };
                    let received = self.bits[w].count_ones().min(relevant_bits);
                    self.loss_accum += relevant_bits - received;
                }
            }
            self.bits = [0u64; 16];
        } else {
            for w in 0..word_shift {
                if w < 16 {
                    let word_base = self.base_seq + (w as u64 * 64);
                    if word_base < self.highest_marked {
                        let relevant_bits = if word_base + 64 <= self.highest_marked { 64 }
                        else { (self.highest_marked - word_base) as u32 };
                        let received = self.bits[w].count_ones().min(relevant_bits);
                        self.loss_accum += relevant_bits - received;
                    }
                }
            }
            let remain = 16 - word_shift;
            for i in 0..remain { self.bits[i] = self.bits[i + word_shift]; }
            for i in remain..16 { self.bits[i] = 0; }
        }
        self.base_seq = target_base;
    }

    pub fn drain_losses(&mut self) -> (u32, u64) {
        let losses = self.loss_accum;
        self.loss_accum = 0;
        let nack = if self.highest_marked >= self.base_seq + 63 {
            let nack_base = self.highest_marked - 63;
            if nack_base >= self.base_seq {
                let offset = (nack_base - self.base_seq) as usize;
                let word_idx = offset >> 6;
                let bit_idx = offset & 63;
                if bit_idx == 0 && word_idx < 16 { self.bits[word_idx] }
                else if word_idx + 1 < 16 {
                    (self.bits[word_idx] >> bit_idx) | (self.bits[word_idx + 1] << (64 - bit_idx))
                } else if word_idx < 16 { self.bits[word_idx] >> bit_idx }
                else { u64::MAX }
            } else { u64::MAX }
        } else { u64::MAX };
        (losses, nack)
    }
}

pub struct ReceiverState {
    pub highest_seq: u64,
    pub delivered: u32,
    pub last_feedback_ns: u64,
    pub last_rx_batch_ns: u64,
}

impl ReceiverState {
    pub fn new() -> Self {
        ReceiverState { highest_seq: 0, delivered: 0, last_feedback_ns: 0, last_rx_batch_ns: 0 }
    }
    #[inline(always)]
    pub fn needs_feedback(&self, now_ns: u64, rtt_estimate_ns: u64) -> bool {
        if self.delivered >= FEEDBACK_INTERVAL_PKTS { return true; }
        if self.delivered > 0 && self.last_feedback_ns > 0
            && now_ns.saturating_sub(self.last_feedback_ns) >= rtt_estimate_ns { return true; }
        false
    }
}
