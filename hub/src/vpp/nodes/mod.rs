// M13 HUB — VPP GRAPH NODES
//
// Each node processes a PacketVector and fills a Disposition array.
// Nodes are pure functions of their input — no hidden state mutation.
// All state is passed explicitly via the node context.

pub mod rx_parse;
pub mod aead_ops;
pub mod classify;
pub mod tun_io;
pub mod tx_enqueue;
