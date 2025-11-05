

#include <vector>
#include <algorithm>

std::vector<int> find_conflicts_with_index_gap_sparse(
    const std::vector<std::vector<int>>& jobToPeople,
    const std::vector<std::vector<int>>& conflicts,
    int jobIndex)
{
    const auto& people_j = jobToPeople[jobIndex];
    if (people_j.empty()) return {};

    int imax = *std::max_element(people_j.begin(), people_j.end());
    int cutoff = imax - 3;
    if (cutoff < 0) return {};

    std::vector<int> result;
    for (int k : conflicts[jobIndex]) {
        if (k == jobIndex) continue;

        // Does job k have a person <= cutoff?
        for (int p : jobToPeople[k]) {
            if (p <= cutoff) {
                result.push_back(k);
                break;
            }
        }
    }

    return result;
}
#include <iostream>

int main() {
    // jobToPeople[j] = list of people indices
    std::vector<std::vector<int>> jobToPeople = {
        {4, 6, 9},  // job 0
        {2, 5},     // job 1
        {7, 8},     // job 2
        {3, 6, 10}  // job 3
    };

    // conflict list
    std::vector<std::vector<int>> conflicts = {
        {1, 2, 3},  // job 0 conflicts with 1,2,3
        {0}, {0}, {0}
    };

    int jobIndex = 0;
    auto result = find_conflicts_with_index_gap_sparse(jobToPeople, conflicts, jobIndex);

    std::cout << "Qualified conflicting jobs for job " << jobIndex << ": ";
    for (int j : result)
        std::cout << j << " ";
    std::cout << "\n";
}


#include <Eigen/Sparse>
#include <vector>

using namespace Eigen;

std::vector<int> find_conflicts_sparse(const SparseMatrix<int>& A, int jobIndex)
{
    SparseMatrix<int> At = A.transpose();

    // compute column of (AᵀA) corresponding to jobIndex efficiently
    VectorXi overlap = At * A.col(jobIndex);

    std::vector<int> conflicts;
    for (int j = 0; j < overlap.size(); ++j) {
        if (j != jobIndex && overlap[j] > 0)
            conflicts.push_back(j);
    }
    return conflicts;
}
#include <Eigen/Sparse>
#include <iostream>

using namespace Eigen;

int main() {
    const int numPeople = 5;
    const int numJobs = 4;

    // Triplets for sparse matrix construction
    std::vector<Triplet<int>> triplets = {
        {0,0,1}, {2,0,1},  // job 0
        {1,1,1},           // job 1
        {0,2,1}, {1,2,1},  // job 2
        {2,3,1}            // job 3
    };

    SparseMatrix<int> A(numPeople, numJobs);
    A.setFromTriplets(triplets.begin(), triplets.end());

    int jobIndex = 0;
    VectorXi overlap = A.transpose() * A.col(jobIndex);

    std::cout << "Conflicts for job " << jobIndex << ": ";
    for (int j = 0; j < overlap.size(); ++j) {
        if (j != jobIndex && overlap[j] > 0)
            std::cout << j << " ";
    }
    std::cout << "\n";
}
