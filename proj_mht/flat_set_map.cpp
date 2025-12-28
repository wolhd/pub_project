#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <string>

// ==========================================================
// CACHE-FRIENDLY FLAT MAP
// ==========================================================
template <typename K, typename V>
class FlatMap {
private:
    struct Pair {
        K key;
        V value;
        bool operator<(const Pair& other) const { return key < other.key; }
        friend bool operator<(const Pair& p, const K& k) { return p.key < k; }
    };
    std::vector<Pair> data;

public:
    void insert(K key, V value) {
        auto it = std::lower_bound(data.begin(), data.end(), key);
        if (it != data.end() && it->key == key) {
            it->value = value; 
        } else {
            data.insert(it, {key, value});
        }
    }

    V* find(const K& key) {
        auto it = std::lower_bound(data.begin(), data.end(), key);
        if (it != data.end() && it->key == key) {
            return &(it->value);
        }
        return nullptr;
    }

    size_t size() const { return data.size(); }
};

// ==========================================================
// CACHE-FRIENDLY FLAT SET (With Delete)
// ==========================================================
template <typename T>
class FlatSet {
private:
    std::vector<T> data;

public:
    void insert(const T& value) {
        auto it = std::lower_bound(data.begin(), data.end(), value);
        if (it == data.end() || *it != value) {
            data.insert(it, value);
        }
    }

    // New Delete Method
    bool remove(const T& value) {
        auto it = std::lower_bound(data.begin(), data.end(), value);
        if (it != data.end() && *it == value) {
            data.erase(it); // O(n) shift, but cache-friendly
            return true;
        }
        return false;
    }

    bool contains(const T& value) const {
        return std::binary_search(data.begin(), data.end(), value);
    }

    size_t size() const { return data.size(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

// ==========================================================
// UNIT TESTS
// ==========================================================
void runTests() {
    // FlatMap Tests
    FlatMap<int, std::string> map;
    map.insert(1, "Hello");
    assert(*map.find(1) == "Hello");
    std::cout << "✓ FlatMap tests passed.\n";

    // FlatSet Tests (Including Delete)
    FlatSet<int> set;
    set.insert(10);
    set.insert(20);
    set.insert(30);

    // Test Deletion
    assert(set.size() == 3);
    bool deleted = set.remove(20);
    assert(deleted == true);
    assert(set.size() == 2);
    assert(set.contains(20) == false);

    // Test Deleting Non-existent
    bool deleted_fake = set.remove(99);
    assert(deleted_fake == false);
    assert(set.size() == 2);

    // Verify ordering is maintained after delete
    auto it = set.begin();
    assert(*it == 10);
    assert(*(++it) == 30);

    std::cout << "✓ FlatSet tests (with delete) passed.\n";
}

int main() {
    runTests();
    std::cout << "\nAll tests passed successfully!" << std::endl;
    return 0;
}
