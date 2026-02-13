// M13 HUB — NETWORK MODULE
// Re-exports from datapath.rs (AF_XDP, FixedSlab, Engine, Telemetry, BPF).
// datapath.rs is preserved as-is — it contains the working zero-copy I/O layer.

pub mod af_xdp {
    //! AF_XDP traits and types used by the scheduler and VPP graph.
    //! Concrete implementations live in datapath.rs (Engine, ZeroCopyTx).

    /// Trait for TX path operations. Implemented by ZeroCopyTx in datapath.rs.
    pub trait TxPath {
        fn available_slots(&mut self) -> u32;
        fn stage_tx_addr(&mut self, addr: u64, len: u32);
        fn commit_tx(&mut self);
        fn kick_tx(&mut self);
    }
}

pub mod telemetry {
    //! Telemetry re-exports.
    pub use crate::datapath::Telemetry;
}
