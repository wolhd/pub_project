#include <iostream>
#include <vector>
#include <cstdlib>   // for rand()
#include <ctime>     // for srand()

int partition(std::vector<int>& nums, int left, int right, int pivotIndex) {
    int pivotValue = nums[pivotIndex];
    std::swap(nums[pivotIndex], nums[right]); // Move pivot to end
    int storeIndex = left;

    for (int i = left; i < right; ++i) {
        if (nums[i] < pivotValue) {
            std::swap(nums[storeIndex], nums[i]);
            ++storeIndex;
        }
    }

    std::swap(nums[storeIndex], nums[right]); // Move pivot to final place
    return storeIndex;
}

int quickSelect(std::vector<int>& nums, int left, int right, int k) {
    if (left == right)
        return nums[left];

    int pivotIndex = left + rand() % (right - left + 1);
    pivotIndex = partition(nums, left, right, pivotIndex);

    if (k == pivotIndex)
        return nums[k];
    else if (k < pivotIndex)
        return quickSelect(nums, left, pivotIndex - 1, k);
    else
        return quickSelect(nums, pivotIndex + 1, right, k);
}

double findMedianQuickSelect(std::vector<int>& nums) {
    int n = nums.size();
    if (n == 0)
        throw std::invalid_argument("Array is empty");

    srand(static_cast<unsigned int>(time(nullptr))); // Seed for randomness

    if (n % 2 == 1) {
        return quickSelect(nums, 0, n - 1, n / 2);
    } else {
        int left = quickSelect(nums, 0, n - 1, n / 2 - 1);
        int right = quickSelect(nums, 0, n - 1, n / 2);
        return (left + right) / 2.0;
    }
}

int main() {
    std::vector<int> arr = {7, 1, 3, 5, 9, 2};

    try {
        double median = findMedianQuickSelect(arr);
        std::cout << "Median: " << median << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
