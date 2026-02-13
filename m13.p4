/* * M13 PHASE I - SILICON DEFINITION (REVISION 5.14)
 * FILE: m13.p4
 * TARGET: Hybrid (Vitis HW / BMv2 Sim)
 * STRICTNESS: Fiduciary
 * Sprint 5.14: Raw Ethernet (EtherType 0x88B5), magic+version validation.
 *   Removed UDP encapsulation — Hub sends raw L2 frames.
 */

#include <core.p4>

#if defined(SIMULATION)
    #include <v1model.p4>
#endif

// ---------------------------------------------------------------------------
// 1. CONSTANTS & HEADERS
// ---------------------------------------------------------------------------
const bit<16> ETHERTYPE_M13  = 0x88B5; // IEEE 802.1 Local Experimental
const bit<16> ETHERTYPE_IPV4 = 0x0800; // Retained for dual-stack future
const bit<8>  M13_MAGIC      = 0xD1;   // Wire protocol magic (signature[0])
const bit<8>  M13_VERSION_1  = 0x01;   // Phase 1 wire protocol version

header ethernet_h {
    bit<48> dstAddr; bit<48> srcAddr; bit<16> etherType;
}

// Retained for future dual-stack support — not parsed in Phase 1
header ipv4_h {
    bit<4> version; bit<4> ihl; bit<8> diffserv; bit<16> totalLen;
    bit<16> identification; bit<3> flags; bit<13> fragOffset;
    bit<8> ttl; bit<8> protocol; bit<16> hdrChecksum;
    bit<32> srcAddr; bit<32> dstAddr;
}

header udp_h {
    bit<16> srcPort; bit<16> dstPort; bit<16> length; bit<16> checksum;
}

// THE M13 WIRE STRUCT (Aligned to 64-bit Memory Bus)
// signature[0] = magic (0xD1), signature[1] = version (0x01)
// signature[2..31] = reserved (Phase 2: cryptographic signature)
header m13_h {
    bit<8>   magic;        // 1B — must be 0xD1
    bit<8>   version;      // 1B — must be 0x01 for Phase 1
    bit<240> sig_reserved; // 30B — reserved for Phase 2 crypto
    bit<64>  sequence_id;  // 8B
    bit<8>   flags;        // 1B
    bit<32>  payload_len;  // 4B
    bit<24>  _padding;     // 3B (Alignment Fuse)
}

struct headers {
    ethernet_h ethernet;
    ipv4_h     ipv4;
    udp_h      udp;
    m13_h      m13;
}

struct metadata { bit<1> drop_flag; }

// ---------------------------------------------------------------------------
// 2. PARSER (The State Machine)
// Sprint 5.14: Parse raw Ethernet → M13 directly on EtherType 0x88B5.
//   IPv4/UDP path retained but not used in Phase 1.
// ---------------------------------------------------------------------------
parser M13Parser(packet_in packet,
                 out headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {

    state start { transition parse_ethernet; }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_M13  : parse_m13;
            ETHERTYPE_IPV4 : parse_ipv4;
            default        : accept;
        }
    }

    // Primary path: raw Ethernet → M13 header
    state parse_m13 {
        packet.extract(hdr.m13);
        transition accept;
    }

    // Legacy path: retained for future dual-stack / tunneled mode
    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }
}

// ---------------------------------------------------------------------------
// 3. PIPELINE (The Firewall)
// ---------------------------------------------------------------------------

control M13VerifyChecksum(inout headers h, inout metadata m) {
    apply { }
}

control M13Ingress(inout headers hdr, inout metadata meta, inout standard_metadata_t sm) {
    apply {
        if (hdr.m13.isValid()) {
            // Wire protocol validation: magic + version must match
            if (hdr.m13.magic != M13_MAGIC || hdr.m13.version != M13_VERSION_1) {
                mark_to_drop(sm);
            }
            // Valid M13 packet — proceed to DMA / forwarding
        } else {
            // Non-M13 EtherType reached ingress — drop (fail-secure)
            mark_to_drop(sm);
        }
    }
}

control M13Egress(inout headers h, inout metadata m, inout standard_metadata_t sm) {
    apply { }
}

control M13ComputeChecksum(inout headers h, inout metadata m) {
    apply { }
}

control M13Deparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.m13);
    }
}

// ---------------------------------------------------------------------------
// 4. ARCHITECTURE MAPPING
// ---------------------------------------------------------------------------
#if defined(SIMULATION)
    V1Switch(
        M13Parser(),
        M13VerifyChecksum(),
        M13Ingress(),
        M13Egress(),
        M13ComputeChecksum(),
        M13Deparser()
    ) main;
#endif
