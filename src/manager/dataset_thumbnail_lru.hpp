#pragma once

#include <cstddef>
#include <vector>

// Per-cell status the Dataset Browser grid draws differently: a
// placeholder while Loading, the actual texture once Loaded, or a
// distinct failed indicator for Failed -- no automatic retry once
// Failed (see DatasetThumbnailWorker's doc comment).
enum class DatasetThumbnailStatus { Loading, Loaded, Failed };

// Pure function: given the current LRU order of every task id with a
// resident thumbnail texture (oldest-touched first, most-recently-touched
// last) and the set of task ids currently visible on screen (which must
// never be evicted no matter how stale their LRU position), returns
// which ids to evict so at most `residentCap` remain resident afterward.
// Evicts least-recently-touched ids first, skipping any in
// `currentlyVisible`; if `currentlyVisible` alone is at or over
// `residentCap`, every non-visible id is returned (visible ids are never
// evicted, even if that puts residency over the cap).
std::vector<int> idsToEvict(
    const std::vector<int>& lruOrder, const std::vector<int>& currentlyVisible, size_t residentCap);
