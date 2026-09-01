// SPDX-License-Identifier: MIT
//! OWNER: WP-10. Stub written by WP-0 (todo!()-free: the allocator must be in place from minute one).
//! The ONLY unsafe in the binary: a `GlobalAlloc` wrapper over `System` that counts allocations and deallocations
//! into atomics ONLY while `COUNTING` is set. Always installed (a global allocator cannot be installed at run time);
//! cost while not counting is one relaxed atomic load per allocation, off the measured path anyway.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations counted since the last `reset()` (while the gate was set).
pub fn allocations() -> u64 {
    ALLOCS.load(Ordering::Relaxed)
}

/// Deallocations counted since the last `reset()` (while the gate was set).
pub fn deallocations() -> u64 {
    DEALLOCS.load(Ordering::Relaxed)
}

pub fn reset() {
    ALLOCS.store(0, Ordering::Relaxed);
    DEALLOCS.store(0, Ordering::Relaxed);
}

/// reset + open the gate.
pub fn start() {
    reset();
    COUNTING.store(true, Ordering::SeqCst);
}

/// close the gate; returns (allocations, deallocations) counted while it was open.
pub fn stop() -> (u64, u64) {
    COUNTING.store(false, Ordering::SeqCst);
    (allocations(), deallocations())
}
