#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <functional> // For std::less

// ==========================================================
// CACHE-FRIENDLY FLAT SET (With Custom Comparator)
// ==========================================================
template <typename T, typename Compare = std::less<T>>
class FlatSet {
private:
    std::vector<T> data;
    Compare comp; // The comparator instance

    // Restores order and uniqueness after bulk changes
    void finalize() {
        std::sort(data.begin(), data.end(), comp);
        // unique only works on adjacent elements, so we use our comparator
        // to define equality: !comp(a, b) && !comp(b, a)
        auto equal = [this](const T& a, const T& b) {
            return !comp(a, b) && !comp(b, a);
        };
        data.erase(std::unique(data.begin(), data.end(), equal), data.end());
    }

public:
    // Support for Range-based for (colon) loops
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    const_iterator begin() const { return data.begin(); }
    const_iterator end() const { return data.end(); }

    // --- Standard Operations ---
    void insert(const T& value) {
        auto it = std::lower_bound(data.begin(), data.end(), value, comp);
        // Check if element already exists (if it's not greater and not less)
        if (it == data.end() || comp(value, *it)) {
            data.insert(it, value);
        }
    }

    bool remove(const T& value) {
        auto it = std::lower_bound(data.begin(), data.end(), value, comp);
        if (it != data.end() && !comp(value, *it)) {
            data.erase(it);
            return true;
        }
        return false;
    }

    bool contains(const T& value) const {
        return std::binary_search(data.begin(), data.end(), value, comp);
    }

    // --- Bulk Operations ---
    template <typename Iter>
    void insert(Iter first, Iter last) {
        data.insert(data.end(), first, last);
        finalize();
    }

    void insert(std::initializer_list<T> list) {
        insert(list.begin(), list.end());
    }

    size_t size() const { return data.size(); }
};

// ==========================================================
// UNIT TESTS & DEMONSTRATION
// ==========================================================

// A custom comparator for descending order
struct DescendingInt {
    bool operator()(const int& a, const int& b) const {
        return a > b; 
    }
};

int main() {
    // 1. Test with default comparator (Ascending)
    FlatSet<int> ascendingSet;
    ascendingSet.insert({5, 1, 9, 3});
    
    std::cout << "Ascending (Default): ";
    for (int x : ascendingSet) std::cout << x << " "; // 1 3 5 9
    std::cout << "\n";
    assert(*ascendingSet.begin() == 1);

    // 2. Test with custom comparator (Descending)
    FlatSet<int, DescendingInt> descendingSet;
    descendingSet.insert({5, 1, 9, 3});

    std::cout << "Descending (Custom):  ";
    for (int x : descendingSet) std::cout << x << " "; // 9 5 3 1
    std::cout << "\n";
    assert(*descendingSet.begin() == 9);

    // 3. Verify uniqueness with custom comparator
    descendingSet.insert(5); 
    assert(descendingSet.size() == 4);

    std::cout << "✓ Custom comparator and colon loops verified!" << std::endl;
    return 0;
}
