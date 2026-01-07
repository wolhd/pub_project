#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

using namespace std;
using vecvec = vector<vector<int>>;
using veci = vector<int>;
const int MAX_HYP = 4000;

std::string vecvecStr(const std::vector<std::vector<int>>& vec2D) {
    std::stringstream ss;
    ss << "["; // Outer bracket

    for (size_t i = 0; i < vec2D.size(); ++i) {
        ss << "["; // Inner bracket
        for (size_t j = 0; j < vec2D[i].size(); ++j) {
            ss << vec2D[i][j];
            if (j < vec2D[i].size() - 1) {
                ss << ", "; // Separator between numbers
            }
        }
        ss << "]"; // Close inner bracket
        if (i < vec2D.size() - 1) {
            ss << ", "; // Separator between inner vectors
        }
        ss << std::endl;
    }
    ss << "]"; // Close outer bracket

    return ss.str();
}
int rand(int min_val, int max_val) {

    // 1. Seed the random number engine
    // Using the system clock provides a unique seed each time the program runs
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed); // Standard mersenne_twister_engine

    // 2. Define the distribution
    // This defines the range [min_val, max_val] inclusive
    std::uniform_int_distribution<int> dist(min_val, max_val);

    // 3. Generate the random number
    int random_num = dist(gen);
    return random_num;
}
vector<int> nRand(int N, int min, int max) {
    veci res;
    for (int i = 0; i < N; ++i) {
        res.push_back(rand(min, max));
    }
    return res;
}

vecvec genRepRows(int nr, int nMin, int nMax) {
    vecvec res;
    for( int i=0; i<nr; i++) {
        int nhyp = rand(nMin, nMax);
        res.push_back( nRand(nhyp, 0, MAX_HYP) );
    }
    return res;
}

vecvec generate() {
    vector<vector<int>> all;

    int numReps = 500, numHypMin = 2, numHypMax = 200;
    vecvec repRows = genRepRows(numReps, numHypMin, numHypMax);
    all.insert(all.begin(), repRows.begin(), repRows.end());

    // std::cout << vecvecStr(all); 
    // cout << std::endl;
    return all;
}
unordered_map<int,unordered_set<int>> adjListSet(vecvec repsHyps) {
    unordered_map<int,unordered_set<int>> mapSet;
    for(auto repVec : repsHyps) {
        for(int i=0; i < repVec.size(); i++) {
            int h = repVec[i];
            mapSet[h].insert(repVec.begin(), repVec.end());
        }
    }
    return mapSet;
}
vecvec adjListVec(vecvec repsHyps) {
    vecvec vecVec(MAX_HYP+1);
    for (veci v : vecVec) {
        v.reserve(100);
    }
    for(auto repVec : repsHyps) {
        for(int i=0; i < repVec.size()-1; i++) {
            int h1 = repVec[i];
            for(int j=1; j < repVec.size(); j++) {
                int h2 = repVec[j];
                vecVec[h1].push_back(h2);
                vecVec[h2].push_back(h1);
            }
        }
    }
    return vecVec;
}
int main() {
    int N = 4;
    for (int i=0; i < N; i++) {
        
        auto start = std::chrono::high_resolution_clock::now();
        adjListSet(generate());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time elapsed: " << duration.count() << " milliseconds" << std::endl;
    }
    std::cout << "adjListVec" << endl;
    for (int i=0; i < N; i++) {
        
        auto start = std::chrono::high_resolution_clock::now();
        adjListVec(generate());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time elapsed: " << duration.count() << " milliseconds" << std::endl;
    }
    return 0;
}
