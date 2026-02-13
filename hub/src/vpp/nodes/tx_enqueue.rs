// M13 HUB — VPP NODE: TX ENQUEUE
// Final node in the TX graph. Takes encrypted, framed packets and submits
// them to the AF_XDP TX ring via the Scheduler.
//
// This node bridges the VPP graph to the AF_XDP zero-copy TX path.
// It operates in bulk: all packets in the vector are enqueued to the
// scheduler, then a single commit+kick flushes the batch to hardware.

use crate::vpp::{PacketVector, Disposition, NextNode};
use crate::engine::scheduler::Scheduler;

/// Enqueue a vector of packets to the TX scheduler.
/// All packets go to bulk queue (data). Critical frames (feedback) are
/// enqueued separately by the feedback processing node.
#[inline]
pub fn tx_enqueue_vector(
    input: &PacketVector,
    disp: &mut Disposition,
    scheduler: &mut Scheduler,
) {
    let n = input.len;
    for i in 0..n {
        let desc = &input.descs[i];
        scheduler.enqueue_bulk(desc.addr, desc.len);
        disp.next[i] = NextNode::Consumed;
    }
}
