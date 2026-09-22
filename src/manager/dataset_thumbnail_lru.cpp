#include "manager/dataset_thumbnail_lru.hpp"

#include <algorithm>

std::vector<int> idsToEvict(
    const std::vector<int>& lruOrder, const std::vector<int>& currentlyVisible, size_t residentCap) {
    std::vector<int> toEvict;
    if (lruOrder.size() <= residentCap) {
        return toEvict;
    }

    size_t residentCount = lruOrder.size();
    for (const int id : lruOrder) {
        if (residentCount <= residentCap) {
            break;
        }
        if (std::find(currentlyVisible.begin(), currentlyVisible.end(), id) != currentlyVisible.end()) {
            continue; // never evict something on screen right now
        }
        toEvict.push_back(id);
        residentCount--;
    }

    return toEvict;
}
