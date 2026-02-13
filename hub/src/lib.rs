// M13 HUB — CRATE ROOT (LIBRARY)
// World-class VPP architecture: genuine graph-based vector packet processing.
//
// Module hierarchy:
//   protocol/  — Wire format, peer table, fragmentation
//   crypto/    — AES-256-GCM AEAD, PQC handshake (ML-KEM + ML-DSA)
//   engine/    — TSC clock, scheduler, jitter buffer, RX state
//   tunnel/    — TUN interface, raw UDP framing, IP checksum
//   vpp/       — Graph framework: PacketVector, nodes, scatter, executor
//
// NOTE: datapath.rs is declared by main.rs (binary crate), not here.
// The lib.rs modules are the VPP architecture layer that sits above the
// existing datapath engine. main.rs orchestrates both.

#![allow(unused_imports)]
#![allow(dead_code)]

pub mod protocol;
pub mod crypto;
pub mod engine;
pub mod tunnel;
pub mod vpp;
