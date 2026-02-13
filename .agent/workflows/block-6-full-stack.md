---
description: Block 6 Full-Stack Roadmap — Node Datapath, PQC, Crypto, Identity, RLNC, Verification (6.1–6.14)
---

# BLOCK 6 — REORGANIZED ROADMAP (NEW ORDER)

## Ground Truth

The correct new order acknowledges this reality:

| New Sprint | Content |
|---|---|
| 6.1 | Node Datapath + Dual Transport + Frag + Hexdump |
| 6.2 | PQC Key Exchange (ML-KEM-1024) + AEAD + Anti-Replay |
| 6.3 | PQC Auth (ML-DSA-87) + Plane Separation + BPF Hardening |
| 6.4 | Distributed Trust (PKI, DAG, CBOR MicroCerts) |
| 6.5 | Automated Lifecycle (Peer FSM) |
| 6.6 | Persistence (redb, ACID, Merkle) |
| 6.7 | Multi-path + AVX-512 Batch Classification |
| 6.8 | RLNC Codec (Handshake FEC) + AVX-512 GF(2^8) |
| 6.9 | BBRv3 Reconfigure + Busy-Poll + PGO + Bench |
| 6.10 | Timing Verification (Jitter + Latency) |
| 6.11 | Health Vector (Operational Dashboard) |
| 6.12 | Flood Test + Enemy Verification |
| 6.13 | E2E Hub ↔ Node Integration |
| 6.14 | CSE Contested Environment (Final Proof) |

## Dependency Chain

```
6.1 (Node + Transport + Frag + Hexdump) ← DONE
  └→ 6.2 (PQC ML-KEM-1024 Key Exchange + ChaCha20-Poly1305 AEAD + Anti-Replay)
       └→ 6.3 (PQC ML-DSA-87 Mutual Auth + Plane Separation + BPF Hardening)
            └→ 6.4 (PKI, DAG Chain, CBOR MicroCerts, ML-DSA-87)
                 └→ 6.5 (Peer FSM: 6-state lifecycle, heartbeat, quarantine, ban)
                      └→ 6.6 (redb persistence, ACID, Merkle integrity)
                           └→ 6.7 (Multi-path + AVX-512 Classify)
                                └→ 6.8 (RLNC + AVX-512 GF)
                                     └→ 6.9 (BBRv3 recal + PGO + Bench) ← Code shape FINAL
                                          └→ 6.10 (Timing Verification)
                                               └→ 6.11 (Health Dashboard)
                                                    └→ 6.12 (Flood Test)
                                                         └→ 6.13 (E2E Integration)
                                                              └→ 6.14 (CSE Final Proof)
```

Each sprint only activates after the previous one is bilateral-proven. No skipping.

---

## SPRINT 6.1 — Node Datapath + Dual Transport + Fragmentation + Hexdump (DONE — needs 4 fixes)

### 6.1.1 — ARCHITECTURAL MOTIVATION

Block 5 produced a fully functional Hub: AF_XDP engine, BBRv3, jitter buffer, DWRR scheduler (~1,800 lines in hub/src/main.rs, ~541 lines in hub/src/datapath.rs). But the Node was:

```
fn main() {}
```

No bilateral testing is possible without a Node. No crypto, no PKI, no FSM can be validated without a second endpoint. Therefore the Node engine is Sprint 6.1 — the foundation everything else builds on.

**Key architectural decision: Dual Transport.**

The Hub runs on Hetzner (Germany). The Node runs on a local laptop. They are NOT on the same L2 segment. AF_XDP requires same-L2 (raw Ethernet). Therefore the Node must support TWO transport modes:

```
MODE 1 — AF_XDP (same-L2, WiFi 7 in production):
  sudo ./m13-node --iface wlp91s0 [--echo] [--hexdump]
  Uses: UMEM, zero-copy, BPF steersman, isolated core, IRQ fencing

MODE 2 — UDP (cross-internet, testbed):
  ./m13-node --hub-ip <hub_ip:port> [--echo] [--hexdump]
  Uses: kernel UDP socket, non-privileged, no BPF required
```

Both modes share: FSM, echo logic, hexdump, telemetry, M13 wire format, fragmentation.
Both modes use M13's full wire format (ETH header + M13Header) even over UDP to keep the protocol identical.

The Hub gains a UDP listener that accepts UDP connections from Nodes and participates in the same M13 protocol over a UDP socket in parallel with AF_XDP.

### 6.1.2 — ZERO-SHARE ARCHITECTURE

Hub and Node have **completely independent** datapath.rs files. No shared crate, no `#[cfg]` gymnastics.

| Component | Hub datapath.rs | Node datapath.rs |
|---|---|---|
| UMEM size | 1 GB | 512 MB |
| Slab depth | 8192 frames | 4096 frames |
| Workers | Up to 4 | 1 (single WiFi NIC) |
| Telemetry fields | Full (BBR, jitter, scheduler) | Subset (TX/RX/drops/cycles/state) |
| Diagnostic prefix | `[M13-HUB FATAL 0x__]` | `[M13-NODE FATAL 0x__]` |

### 6.1.3 — NODE STATE FSM (3-STATE)

```rust
enum NodeState { Disconnected, Registering, Established }
```

Transitions: Boot → Disconnected → Registering → Established. Link-loss (5s) → Disconnected.

### 6.1.4 — FRAGMENTATION ENGINE (BILATERAL)

Fragment sub-header (8 bytes). Max 16 fragments per message. Max reassembled: 23,104 bytes. Assembly timeout: 5 seconds.

### 6.1.5 — HEXDUMP ENGINE (4 CAPTURE POINTS)

Rate-limited to 10 dumps/sec. Annotated: ETH, MAGIC, CRYPT, MAC, NONCE, SEQ, FLAGS, PLEN, PAD.

### 6.1.6 — HUB UDP LISTENER

Default port 443 (blends with QUIC). Always-on. BBR-derived pacing for UDP TX.

### 6.1.7 — AUTO-DETECT MAC

Both Hub and Node read MAC from `/sys/class/net/<iface>/address`. Fallback: random LAA.

### 6.1.8 — AUTO-CLEANUP (Hub)

Pre-flight: kill stale processes (excludes self), detach XDP, allocate hugepages (dynamically calculated).

### 6.1.9 — KNOWN GAPS (TO FIX BEFORE 6.2)

| Gap | Fix |
|---|---|
| AF_XDP frag stub | Port UDP path logic into AF_XDP path |
| Hexdump detail | Node missing CRYPT/MAC/NONCE/PLEN annotations |
| Hot-loop syscall | Replace clock_ns() with rdtsc_ns() |
| Bilateral proof | Run 1000-frame echo test + strace verification |

### 6.1.10 — COMMANDS

```bash
# Hub (on server)
sudo RUST_LOG=debug ./target/release/m13-hub --hexdump -i eno2

# Node (on laptop)
sudo RUST_LOG=debug ./target/release/m13-node --hub-ip <hub_ip>:443 --hexdump
```

---

## SPRINT 6.2 — PQC Authenticated Key Exchange + AEAD + Anti-Replay

**Depends on:** 6.1 (bilateral cleartext transport)
**Pure PQC:** ML-KEM-1024 (FIPS 203, Level 5) + ChaCha20-Poly1305 (256-bit, 128-bit PQ) + HKDF-SHA-512

### 6.2.1 — THREAT MODEL

Zero crypto = forgery, replay, eavesdropping, bit-flip, reflection, MITM, harvest-now-decrypt-later (HNDL). HNDL is the PQC-specific threat: adversary records traffic today, decrypts when CRQC available. Only PQC key exchange prevents HNDL.

### 6.2.2 — PQC ALGORITHM SELECTION

| Role | Algorithm | Standard | Level | Key Sizes |
|------|-----------|----------|-------|-----------|
| Key Exchange | ML-KEM-1024 | FIPS 203 | **5** | ek=1568B, dk=3168B, ct=1568B, ss=32B |
| Digital Signature | ML-DSA-87 | FIPS 204 | **5** | pk=2592B, sk=4896B, sig=4627B |
| Symmetric AEAD | ChaCha20-Poly1305 | RFC 8439 | 128-bit PQ | key=32B, nonce=12B, tag=16B |
| KDF | HKDF-SHA-512 | RFC 5869 | 256-bit PQ | N/A |

No X25519. No Ed25519. No RSA. No hybrid. Rust crates: `ml-kem`, `ml-dsa`, `sha2`, `hkdf` (cold-path only).

### 6.2.3 — PQC HANDSHAKE (3-MESSAGE)

Establishes session key via ephemeral ML-KEM-1024 + ML-DSA-87 mutual auth.

```
Msg 1: ClientHello (Node→Hub) = session_nonce(32) + ek_node(1568) + pk_node(2592) = 4193B → 3 frags
Msg 2: ServerHello (Hub→Node) = ct(1568) + pk_hub(2592) + sig_hub(4627)           = 8788B → 7 frags
Msg 3: Finished    (Node→Hub) = sig_node(4627)                                    = 4628B → 4 frags
```

Session key: `HKDF-SHA-512(salt=session_nonce, IKM=ML-KEM-ss, info="M13-PQC-SESSION-KEY-v1", L=32)`.
Forward secrecy: ephemeral ML-KEM keypair destroyed after decaps. Total: ~1.8ms, 17.6KB, 14 frags.

### 6.2.4 — WIRE FORMAT v2 (POST-CRYPTO)

```
signature[0]     = 0xD1 (magic) — CLEARTEXT
signature[1]     = 0x01 (version) — CLEARTEXT
signature[2]     = crypto_version (0x00=cleartext, 0x01=ChaCha20-Poly1305-PQC)
signature[3]     = reserved
signature[4..20] = Poly1305 MAC (16 bytes)
signature[20..32]= nonce (12 bytes: seq_id[8] + direction[1] + pad[3])
```

AAD = signature[0..4]. Encrypted = seq_id + flags + payload_len + padding + Payload.
Direction binding: nonce[8] = 0x00 (hub→node), 0x01 (node→hub).

### 6.2.5 — ChaCha20-Poly1305 AEAD (inline, ~150 lines)

ChaCha20 RFC 8439: ~356 cycles/block. Poly1305: ~35 cycles/16B. Total: ~977 cycles ≈ 264ns/frame (scalar).
Key is PQC-derived session key (NOT PSK). Update DEADLINE_NS to 120,000 post-6.2.

### 6.2.6 — ANTI-REPLAY WINDOW

2048-bit sliding window. `AntiReplay.check(seq)`. Common case: ~8 cycles ≈ 2.2ns.

### 6.2.7 — LONG-TERM IDENTITY KEYS

Each peer: persistent ML-DSA-87 keypair (pk=2592B, sk=4896B). Phase 1: generated at first boot, TOFU. Phase 2 (Sprint 6.4): MicroCerts with ML-DSA-87 signatures.

### 6.2.8 — SESSION LIFECYCLE

Boot → DISCONNECTED → (gen ephemeral ML-KEM) → HANDSHAKING → (3-msg PQC handshake) → ESTABLISHED.
Link loss 5s → zeroize session_key → DISCONNECTED. Rekey → new ephemeral ML-KEM → HANDSHAKING.

### 6.2.9 — TELEMETRY: `auth_fail`, `replay_drops`, `handshake_ok`, `handshake_fail`

### 6.2.10 — K26 FPGA ACCELERATION PATH

| Primitive | Software (x86_64) | FPGA Target |
|-----------|-------------------|-----------|
| ChaCha20-Poly1305 | ~264ns/frame | <50ns/frame |
| ML-KEM-1024 NTT | ~100µs | <10µs |
| ML-DSA-87 NTT | ~120µs verify | <15µs |

---

## SPRINT 6.3 — PQC Handshake/Data Plane Separation + BPF Hardening

**Depends on:** 6.2 (PQC key exchange + AEAD operational)

### 6.3.1 — BPF 4-LAYER DEFENSE

1. Source MAC Allowlist (BPF_MAP_TYPE_HASH, 256 entries)
2. Frame Size Bounds (62..1514)
3. Magic Byte Validation (0xD1, 0x01)
4. Control Frame Rate Limiting (**100/sec** token bucket per source — PQC-aware: ML-DSA-87 verify = ~120µs)

### 6.3.2 — FLAG ALLOCATION

`FLAG_HANDSHAKE = 0x02` (Bit 1). Handshake frames: `FLAGS = FLAG_CONTROL | FLAG_HANDSHAKE = 0x82`.
Handshake sub-types (payload[0]): HS_CLIENT_HELLO=0x01, HS_SERVER_HELLO=0x02, HS_FINISHED=0x03.

### 6.3.3 — HANDSHAKE QUEUE ROUTING

Same AF_XDP socket, dedicated in-userspace queue (64 entries). Copy from UMEM (PQC ops are ~1.8ms — must not hold UMEM frame). Processed AFTER each hot-loop batch.

### 6.3.4 — CLASSIFY ORDER UPDATE

FLAG_FEEDBACK → FLAG_CRITICAL → **FLAG_HANDSHAKE (new)** → FLAG_CONTROL → data.

### 6.3.5 — UDP MODE REPLICATION

Same 4 layers in userspace: size bounds, magic check, MAC check, rate-limit counter.

---

## SPRINT 6.4 — Distributed Trust (PKI, DAG, CBOR MicroCerts)

**Depends on:** 6.2, 6.3

### 6.4.1 — Trust Hierarchy

Root of Trust (offline) → Intermediate CA (field operator) → Leaf (Hub/Node).

### 6.4.2 — MicroCert (~7,400 bytes raw, ~7,600 bytes CBOR)

ML-DSA-87 pub_key(2592) + issuer_key_hash(8) + subject_id(8) + permissions(2) + timestamps + ML-DSA-87 signature(4627).

### 6.4.3 — CBOR Encoder/Decoder (inline, ~140 lines, zero crates)

### 6.4.4 — ML-DSA-87 via `ml-dsa` crate (cold path only, ~120µs verify)

### 6.4.5 — DAG Chain Validation

`verify_cert_chain()` with expiry, issuer binding, revocation (Bloom), permission narrowing, signature verify.

### 6.4.6 — SHA-256 (inline, ~90 lines)

### 6.4.7 — Gossip-Based Revocation (Bloom filter, 1024 bits, k=7, <1% FPR)

---

## SPRINT 6.5 — Automated Lifecycle (Peer FSM)

**Depends on:** 6.4

### 6.5.1 — 6-STATE FSM

Unknown → Quarantined → Active → Rekeying → Zombie → Banned. Default-deny transition table.

### 6.5.2 — Heartbeat Protocol (5s interval, 15s timeout, carries Bloom filter)

### 6.5.3 — Ban Enforcement (BPF MAC removal + userspace defense-in-depth)

### 6.5.4 — Quarantine Isolation (handshake frames only, no data)

---

## SPRINT 6.6 — Persistence (redb, ACID, Merkle)

**Depends on:** 6.5

### 6.6.1 — Architecture: Hot path reads in-memory, control path writes redb, background fsync.

### 6.6.2 — Schema: PEER_TABLE, CERT_TABLE, REVOCATION_TABLE, CONFIG_TABLE, MERKLE_TABLE.

### 6.6.3 — Boot Recovery: Active+stale→Unknown, Zombie/Rekeying→Unknown, Banned→preserved.

### 6.6.4 — Merkle Integrity Tree (SHA-256). Tamper detection → fatal exit.

### 6.6.5 — Write Scheduling: Soft (1s interval) + Hard (immediate for CertValid/Banned/Rekey).

---

## SPRINT 6.7 — Multi-Path Routing + AVX-512 Batch Classification

**Depends on:** 6.5, 6.6

### 6.7.1 — Path Abstraction (AfXdp/Udp, Probing/Active/Degraded/Down)

### 6.7.2 — PathTable (8 paths per peer, DWRR weighted by btlbw)

### 6.7.3 — Critical → primary path. Data → weighted across active paths.

### 6.7.4 — AVX-512 vpshufb classify: 256 frames < 500ns. Scalar fallback: < 2µs.

---

## SPRINT 6.8 — RLNC Codec (Handshake FEC) + AVX-512 GF(2^8)

**Depends on:** 6.4, 6.7

### 6.8.1 — RLNC for HANDSHAKE ONLY (not data plane)

K=4 source, N=12 coded. 50% loss → 93% first-round success.

### 6.8.2 — GF(2^8) with precomputed exp/log tables. ~3.2ns per gf_mul (scalar).

### 6.8.3 — Progressive Gaussian elimination decoder. O(K²) per insertion.

### 6.8.4 — AVX-512 vpshufb split-nibble: 64 bytes/cycle. 370× speedup over scalar.

---

## SPRINT 6.9 — BBRv3 Reconfigure + Busy-Poll + PGO + Bench

**Depends on:** 6.7, 6.8 (code shape FINAL)

### 6.9.1 — Multi-path BBR (uncoupled: loss on path A doesn't affect path B)

### 6.9.2 — SO_PREFER_BUSY_POLL + SO_BUSY_POLL_BUDGET

### 6.9.3 — PGO 4-step: profile-generate → workload → merge → profile-use

### 6.9.4 — Inline benchmark harness (`--bench`): 12+ primitives measured via rdtsc.

---

## SPRINT 6.10 — Timing Verification (Jitter + Latency)

**Depends on:** 6.9 (PGO binary finalized)

### 6.10.1 — Jitter Probe: 1000 TC_CRITICAL frames at 1kHz. Measure inter-arrival variance.

### 6.10.2 — Latency Probe: 10,000 frames with tx_timestamp. Measure RTT distribution (p50/p95/p99/p999).

### 6.10.3 — Pass criteria: σ² < 500µs², p99 < 5ms.

---

## SPRINT 6.11 — Health Vector (Operational Dashboard)

**Depends on:** 6.10

Operational monitoring: health vector aggregation, per-path metrics, per-peer lifecycle state, alerting thresholds.

---

## SPRINT 6.12 — Flood Test + Enemy Verification

**Depends on:** 6.11

Adversarial testing: replay floods, forged frames, rate-limited control floods, Bloom filter poisoning, cert chain attacks.

---

## SPRINT 6.13 — E2E Hub ↔ Node Integration

**Depends on:** 6.12

Full bilateral integration test: handshake → data plane → rekey → link loss → recovery → graceful shutdown.

---

## SPRINT 6.14 — CSE Contested Environment (Final Proof)

**Depends on:** 6.13

Contested Spectrum Environment proof: jamming, rain fade, multi-path failover, RLNC reconnect burst, timing under stress.
