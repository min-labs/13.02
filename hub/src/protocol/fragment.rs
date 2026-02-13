// M13 HUB — FRAGMENTATION ENGINE
// Cold path — handshake only. Heap-allocated reassembly buffers.
// Per-peer assembler with 5-second GC timeout.

use std::collections::HashMap;

/// Fragment reassembly buffer. Tracks up to 16 fragments per message.
pub struct AssemblyBuffer {
    fragments: [Option<Vec<u8>>; 16],
    received_mask: u16,
    total: u8,
    pub first_rx_ns: u64,
}

impl AssemblyBuffer {
    pub fn new(total: u8, now_ns: u64) -> Self {
        AssemblyBuffer { fragments: Default::default(), received_mask: 0, total, first_rx_ns: now_ns }
    }
    pub fn insert(&mut self, index: u8, _offset: u16, data: &[u8]) -> bool {
        if index >= 16 || index >= self.total { return false; }
        let bit = 1u16 << index;
        if self.received_mask & bit != 0 { return self.is_complete(); }
        self.fragments[index as usize] = Some(data.to_vec());
        self.received_mask |= bit;
        self.is_complete()
    }
    pub fn is_complete(&self) -> bool {
        let expected = (1u16 << self.total) - 1;
        self.received_mask & expected == expected
    }
    pub fn reassemble(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for i in 0..self.total as usize {
            if let Some(ref data) = self.fragments[i] { result.extend_from_slice(data); }
        }
        result
    }
}

/// Per-peer fragment assembler with parallel message tracking.
pub struct Assembler {
    pending: HashMap<u16, AssemblyBuffer>,
}

impl Assembler {
    pub fn new() -> Self { Assembler { pending: HashMap::new() } }
    pub fn feed(&mut self, msg_id: u16, index: u8, total: u8, offset: u16,
            data: &[u8], now_ns: u64) -> Option<Vec<u8>> {
        let buf = self.pending.entry(msg_id).or_insert_with(|| AssemblyBuffer::new(total, now_ns));
        if buf.insert(index, offset, data) {
            let result = buf.reassemble();
            self.pending.remove(&msg_id);
            Some(result)
        } else { None }
    }
    pub fn gc(&mut self, now_ns: u64) {
        self.pending.retain(|_, buf| now_ns.saturating_sub(buf.first_rx_ns) < 5_000_000_000);
    }
}
