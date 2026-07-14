//! A pop-or-append slot allocator with stable indices — the classic free-list.
//!
//! `SlotFreeList<T>` hands out stable integer indices into a growing backing store, and
//! recycles the indices of freed elements. Each allocation either **pops** a recycled index
//! (overwriting that slot's payload) or, if none is free, **appends** a new slot at the
//! current high-water mark. An index, once returned, never moves and never aliases a live
//! slot; the backing store only ever grows (a freed slot keeps its now-dead payload until
//! reallocated), so the high-water length is monotonic. This is the textbook free-list of
//! the dynamic-storage-allocation literature specialised to fixed-size slots: the free set
//! is a plain list of reusable indices, allocation is O(1) amortised, and the payloads live
//! in one contiguous vector addressable by slot.
//!
//! ## Reuse order is deterministic ascending (load-bearing)
//!
//! The free set is kept **sorted ascending and deduplicated** after every free, and
//! allocation pops from its end. This makes the sequence of indices a given series of
//! allocate/free operations produces a deterministic function of that series alone — two
//! runs that free the same slots reuse them in the same order. A consumer whose output must
//! be reproducible across an incremental path and a rebuilt-from-scratch path (the two
//! agreeing only up to slot RENUMBERING) relies on this determinism, so the sort+dedup is a
//! contract, not an incidental tidiness.
//!
//! Cite: Wilson, Johnstone, Neely & Boles, *Dynamic Storage Allocation: A Survey and
//! Critical Review* (1995) — the free-list family and reuse policies; Knuth, TAOCP vol. 1
//! §2.5 (dynamic storage allocation, the available-space list). Deviation: fixed-size slots
//! (so no coalescing or size classes) and a total-order reuse policy (sorted free set) for
//! reproducibility rather than allocation speed.

use std::ops::Index;

/// A stable-index slot allocator over payloads of type `T`. Slot indices are `u32`; the
/// backing store grows monotonically and freed indices are recycled in deterministic
/// ascending order.
#[derive(Debug, Clone)]
pub struct SlotFreeList<T> {
    /// Payloads indexed by slot. A freed slot's entry is retained (dead) until the slot is
    /// reallocated, so `slots.len()` is the high-water mark, never the live count.
    slots: Vec<T>,
    /// The recycled (reusable) slot indices, kept sorted ascending and deduplicated; a new
    /// allocation pops from the end before growing `slots`.
    free_indices: Vec<u32>,
}

impl<T> Default for SlotFreeList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SlotFreeList<T> {
    /// An empty allocator (no slots, no free indices).
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
        }
    }

    /// An allocator seeded with `slots` all considered LIVE (an empty free set): slot `i`
    /// holds `slots[i]`, and the next allocation appends at `slots.len()`. The dense-seed
    /// entry for a consumer that already holds a packed `0..count` payload vector.
    pub fn from_slots(slots: Vec<T>) -> Self {
        Self {
            slots,
            free_indices: Vec::new(),
        }
    }

    /// The high-water slot count (live + freed holes) — the length of the backing store,
    /// i.e. the number of distinct indices ever allocated and not yet reused past.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the backing store is empty (no slot has ever been allocated).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The backing payloads in slot order (freed slots included, holding their dead
    /// payloads) — the contiguous view a bulk consumer scatters.
    pub fn as_slice(&self) -> &[T] {
        &self.slots
    }

    /// Allocate a slot for `payload`: reuse a freed index if one is available (keeping the
    /// high-water mark — and thus the backing store — from growing needlessly), else append
    /// a new slot. Reuse pops the LARGEST free index (the free set is sorted ascending), so
    /// the reuse order is a deterministic function of the free/allocate sequence.
    pub fn allocate(&mut self, payload: T) -> u32 {
        match self.free_indices.pop() {
            Some(slot) => {
                self.slots[slot as usize] = payload;
                slot
            }
            None => {
                let slot = self.slots.len() as u32;
                self.slots.push(payload);
                slot
            }
        }
    }

    /// Return `indices` to the free set, then re-sort+dedup the WHOLE set so reuse stays in
    /// deterministic ascending order (and a doubly-freed index cannot appear twice). The
    /// freed slots' payloads are left in place (dead until reallocated); freeing only marks
    /// the indices reusable.
    pub fn free<I: IntoIterator<Item = u32>>(&mut self, indices: I) {
        self.free_indices.extend(indices);
        self.free_indices.sort_unstable();
        self.free_indices.dedup();
    }
}

impl<T> Index<u32> for SlotFreeList<T> {
    type Output = T;

    fn index(&self, slot: u32) -> &T {
        &self.slots[slot as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh allocations hand out ascending indices `0..n` and read back their payloads.
    #[test]
    fn fresh_allocations_are_dense_ascending() {
        let mut list: SlotFreeList<char> = SlotFreeList::new();
        assert!(list.is_empty());
        let slots: Vec<u32> = ['a', 'b', 'c', 'd'].into_iter().map(|c| list.allocate(c)).collect();
        assert_eq!(slots, vec![0, 1, 2, 3]);
        assert_eq!(list.len(), 4);
        assert_eq!(list[2], 'c');
        assert_eq!(list.as_slice(), &['a', 'b', 'c', 'd']);
    }

    /// Freed indices are reused in deterministic ascending order (largest-first pop of the
    /// sorted set), the high-water mark does not grow while free indices remain, and a
    /// double-free is deduplicated (never handed out twice).
    #[test]
    fn frees_reuse_in_deterministic_order_and_dedup() {
        let mut list: SlotFreeList<u32> = SlotFreeList::from_slots(vec![0, 1, 2, 3, 4]);
        assert_eq!(list.len(), 5);

        // Free 1 and 3 (out of order, plus a duplicate 3) — the set sorts to [1, 3].
        list.free([3, 1, 3]);

        // Reuse pops the largest first: 3, then 1. No growth while frees remain.
        assert_eq!(list.allocate(30), 3);
        assert_eq!(list.allocate(10), 1);
        assert_eq!(list.len(), 5);

        // Free set now empty — the next allocation appends at the high-water mark.
        assert_eq!(list.allocate(50), 5);
        assert_eq!(list.len(), 6);
        assert_eq!(list.as_slice(), &[0, 10, 2, 30, 4, 50]);
    }
}
