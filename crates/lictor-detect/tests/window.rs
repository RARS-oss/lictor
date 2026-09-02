// SPDX-License-Identifier: MIT
//! K-of-N window and position-trail tests against hand-computed values.

use lictor_detect::window::{Trail, Window, TRAIL_W};

#[test]
fn k_of_n_sequence_table() {
    // (hit, ring after the push, popcount) for N = 5; K = 3 gives the predictive column.
    let table: [(bool, u64, u8, bool); 10] = [
        (true, 0b00001, 1, false),
        (false, 0b00010, 1, false),
        (true, 0b00101, 2, false),
        (true, 0b01011, 3, true),
        (false, 0b10110, 3, true),
        (true, 0b01101, 3, true),
        (true, 0b11011, 4, true),
        (true, 0b10111, 4, true),
        (false, 0b01110, 3, true),
        (false, 0b11100, 3, true),
    ];
    let mask = (1u64 << 5) - 1;
    let k = 3u8;
    let mut w = Window::default();
    assert_eq!(w.bits, 0);
    assert_eq!(w.hits(), 0);
    for (i, (hit, bits, hits, predictive)) in table.iter().enumerate() {
        w.push(*hit, mask);
        assert_eq!(w.bits, *bits, "ring after step {i}");
        assert_eq!(w.hits(), *hits, "popcount after step {i}");
        assert_eq!(w.hits() >= k, *predictive, "predictive after step {i}");
    }
    w.clear();
    assert_eq!(w.bits, 0);
    assert_eq!(w.hits(), 0);
}

#[test]
fn mask_wraps_old_hits_out() {
    let mask = (1u64 << 5) - 1;
    let mut w = Window::default();
    for _ in 0..5 {
        w.push(true, mask);
    }
    assert_eq!(w.bits, mask);
    assert_eq!(w.hits(), 5);
    // Five clean ticks push every old hit off the ring.
    for i in 0..5 {
        w.push(false, mask);
        assert_eq!(w.hits(), 4 - i, "after clean tick {i}");
    }
    assert_eq!(w.bits, 0);

    // n = 1: the ring is the last tick only.
    let mut w1 = Window::default();
    w1.push(true, 1);
    assert_eq!(w1.bits, 1);
    w1.push(true, 1);
    assert_eq!(w1.bits, 1);
    assert_eq!(w1.hits(), 1);
    w1.push(false, 1);
    assert_eq!(w1.bits, 0);

    // n = 63: the widest ring the freeze allows; 64 hits saturate at 63 and never touch bit 63.
    let mask63 = (1u64 << 63) - 1;
    let mut w63 = Window::default();
    for _ in 0..64 {
        w63.push(true, mask63);
    }
    assert_eq!(w63.bits, mask63);
    assert_eq!(w63.hits(), 63);
    assert_eq!(w63.bits >> 63, 0);
}

#[test]
fn trail_net_and_path_hand_values() {
    let mut t = Trail::default();
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert_eq!(t.net(2, 1), 0.0);
    assert_eq!(t.path(2, 2), 0.0);

    let pts = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [6.0, 8.0], [9.0, 12.0]];
    for p in &pts {
        t.push(p, 2);
    }
    assert!(!t.is_empty());
    assert_eq!(t.len(), 5);
    // Straight line with one repeated sample: net == path == 15 over the full window.
    assert_eq!(t.net(2, 5), 15.0);
    assert_eq!(t.path(2, 5), 15.0);
    // Last three samples: (6,8) (6,8) (9,12).
    assert_eq!(t.net(2, 3), 5.0);
    assert_eq!(t.path(2, 3), 5.0);
    // Last four: (3,4) .. (9,12).
    assert_eq!(t.net(2, 4), 10.0);
    assert_eq!(t.path(2, 4), 10.0);
    // w == 1: no displacement, no path.
    assert_eq!(t.net(2, 1), 0.0);
    assert_eq!(t.path(2, 1), 0.0);
    // Not enough samples: 0.0, never a partial window.
    assert_eq!(t.net(2, 6), 0.0);
    assert_eq!(t.path(2, 6), 0.0);
    // w beyond the ring capacity is also 0.0.
    assert_eq!(t.net(2, TRAIL_W + 1), 0.0);
    assert_eq!(t.path(2, TRAIL_W + 1), 0.0);
    // Only the first d coordinates count.
    assert_eq!(t.net(1, 5), 9.0);
    assert_eq!(t.path(1, 5), 9.0);
}

#[test]
fn trail_zigzag_separates_net_from_path() {
    let mut t = Trail::new();
    for p in [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]] {
        t.push(&p, 2);
    }
    assert_eq!(t.net(2, 4), 1.0);
    assert_eq!(t.path(2, 4), 3.0);
    assert_eq!(t.net(2, 3), 0.0);
    assert_eq!(t.path(2, 3), 2.0);
    t.push(&[0.0, 0.0], 2);
    assert_eq!(t.net(2, 5), 0.0);
    assert_eq!(t.path(2, 5), 4.0);
}

#[test]
fn trail_ring_wraps_and_clears() {
    let mut t = Trail::new();
    for i in 0..(TRAIL_W + 8) {
        t.push(&[i as f64, 0.0], 2);
    }
    assert_eq!(t.len(), TRAIL_W);
    // Newest sample is 39; the (TRAIL_W - 1)-th older one is 8.
    assert_eq!(t.net(2, TRAIL_W), (TRAIL_W - 1) as f64);
    assert_eq!(t.path(2, TRAIL_W), (TRAIL_W - 1) as f64);
    assert_eq!(t.net(2, 2), 1.0);
    t.clear();
    assert!(t.is_empty());
    assert_eq!(t.net(2, 2), 0.0);
    t.push(&[5.0, 5.0], 2);
    t.push(&[8.0, 9.0], 2);
    assert_eq!(t.net(2, 2), 5.0);
}

#[test]
fn trail_push_clamps_dimension_to_the_slice() {
    let mut t = Trail::new();
    // d larger than the slice: only the available coordinates are stored (never a panic).
    t.push(&[1.0, 2.0], 4);
    t.push(&[4.0, 6.0], 4);
    assert_eq!(t.net(2, 2), 5.0);
    assert_eq!(t.net(4, 2), 5.0);
}
