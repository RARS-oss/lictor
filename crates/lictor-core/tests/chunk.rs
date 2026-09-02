// SPDX-License-Identifier: MIT
//! The chunk IR: fill/view/row_mut, every `ChunkError`, index clamping and `all_finite`.

use lictor_core::{ChunkBuf, ChunkError, ChunkView, MAX_D, MAX_H};

fn rows(h: usize, d: usize) -> Vec<f64> {
    (0..h * d).map(|i| i as f64).collect()
}

fn pusht() -> ChunkBuf {
    let mut b = ChunkBuf::new();
    b.fill(3, 24, 15, 2, 8, &rows(15, 2)).unwrap();
    b
}

#[test]
fn fill_and_view() {
    let b = pusht();
    assert!(b.is_filled());
    let v = b.view().unwrap();
    assert_eq!((v.seq, v.t_emit, v.horizon, v.dim, v.exec_steps), (3, 24, 15, 2, 8));
    assert_eq!(v.data.len(), 30);
    assert_eq!(v.action(0), &[0.0, 1.0]);
    assert_eq!(v.action(14), &[28.0, 29.0]);
    assert_eq!(v.overlap_len(), 7);
    assert!(v.all_finite());
}

#[test]
fn every_chunk_error() {
    let cases: [(u16, u16, u16, usize, ChunkError); 6] = [
        (0, 2, 0, 0, ChunkError::Empty),
        (15, 0, 0, 0, ChunkError::Empty),
        (65, 2, 8, 130, ChunkError::TooLong),
        (15, 33, 8, 495, ChunkError::TooWide),
        (15, 2, 16, 30, ChunkError::ExecExceedsHorizon),
        (15, 2, 8, 29, ChunkError::LenMismatch),
    ];
    for (h, d, exec, len, expected) in cases {
        let mut b = ChunkBuf::default();
        let src = vec![0.5; len];
        assert_eq!(b.fill(0, 0, h, d, exec, &src), Err(expected), "h={h} d={d} exec={exec} len={len}");
        assert!(!b.is_filled(), "a rejected fill must leave the buffer unfilled");
        assert!(b.view().is_none());
    }
}

#[test]
fn a_rejected_fill_keeps_the_previous_chunk() {
    let mut b = pusht();
    assert_eq!(b.fill(4, 32, 15, 2, 8, &[1.0; 29]), Err(ChunkError::LenMismatch));
    let v = b.view().unwrap();
    assert_eq!(v.seq, 3);
    assert_eq!(v.action(1), &[2.0, 3.0]);
}

#[test]
fn max_sized_chunk_fits_exactly() {
    let mut b = ChunkBuf::new();
    b.fill(0, 0, MAX_H as u16, MAX_D as u16, MAX_H as u16, &rows(MAX_H, MAX_D)).unwrap();
    let v = b.view().unwrap();
    assert_eq!(v.data.len(), MAX_H * MAX_D);
    assert_eq!(v.action(MAX_H - 1)[MAX_D - 1], (MAX_H * MAX_D - 1) as f64);
    assert_eq!(v.overlap_len(), 0);
    assert!(std::mem::size_of::<ChunkBuf>() >= MAX_H * MAX_D * 8);
}

#[test]
fn row_mut_edits_are_visible_in_the_view() {
    let mut b = pusht();
    {
        let r = b.row_mut(7);
        assert_eq!(r.len(), 2);
        r[0] = -1.0;
        r[1] = f64::INFINITY;
    }
    let v = b.view().unwrap();
    assert_eq!(v.action(7), &[-1.0, f64::INFINITY]);
    assert!(!v.all_finite());
    assert_eq!(b.row_mut(0).len(), 2);
}

#[test]
fn row_mut_on_an_empty_buffer_is_an_empty_slice() {
    let mut b = ChunkBuf::new();
    assert!(b.row_mut(0).is_empty());
}

#[cfg(not(debug_assertions))]
#[test]
fn indices_clamp_to_the_last_row_in_release() {
    let mut b = pusht();
    assert_eq!(b.row_mut(15).len(), 2);
    b.row_mut(100)[0] = 77.0;
    let v = b.view().unwrap();
    assert_eq!(v.action(14), &[77.0, 29.0]);
    assert_eq!(v.action(15), v.action(14));
    assert_eq!(v.action(usize::MAX), v.action(14));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "action index 15 >= horizon 15")]
fn action_index_past_the_horizon_asserts_in_debug() {
    let b = pusht();
    let v = b.view().unwrap();
    let _ = v.action(15);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "row_mut index 15 >= horizon 15")]
fn row_mut_past_the_horizon_asserts_in_debug() {
    let mut b = pusht();
    let _ = b.row_mut(15);
}

#[test]
fn all_finite_detects_nan_and_inf() {
    let mut b = ChunkBuf::new();
    b.fill(0, 0, 2, 2, 1, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(b.view().unwrap().all_finite());
    b.fill(0, 0, 2, 2, 1, &[1.0, f64::NAN, 3.0, 4.0]).unwrap();
    assert!(!b.view().unwrap().all_finite(), "fill accepts NaN; decide() catches it");
    b.fill(0, 0, 2, 2, 1, &[1.0, 2.0, f64::NEG_INFINITY, 4.0]).unwrap();
    assert!(!b.view().unwrap().all_finite());
}

#[test]
fn copy_from_and_clear() {
    let a = pusht();
    let mut b = ChunkBuf::new();
    b.copy_from(a.view().unwrap());
    assert!(b.is_filled());
    let (va, vb) = (a.view().unwrap(), b.view().unwrap());
    assert_eq!(va.data, vb.data);
    assert_eq!(
        (va.seq, va.t_emit, va.horizon, va.dim, va.exec_steps),
        (vb.seq, vb.t_emit, vb.horizon, vb.dim, vb.exec_steps)
    );
    b.clear();
    assert!(!b.is_filled());
    assert!(b.view().is_none());
    b.copy_from(a.view().unwrap());
    assert!(b.is_filled());
}

#[test]
fn hand_built_views_never_panic() {
    let data = [1.0, 2.0];
    let short = ChunkView { seq: 0, t_emit: 0, horizon: 3, dim: 2, exec_steps: 1, data: &data };
    assert_eq!(short.action(0), &[1.0, 2.0]);
    assert!(short.action(1).is_empty());
    let empty = ChunkView { seq: 0, t_emit: 0, horizon: 0, dim: 0, exec_steps: 0, data: &[] };
    assert!(empty.action(0).is_empty());
    assert!(empty.all_finite());
    assert_eq!(empty.overlap_len(), 0);
}
