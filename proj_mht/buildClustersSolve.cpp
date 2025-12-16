// To build clusters of non-interacting track hypotheses:
// Create an interaction graph
//   Edge if hypotheses share measurements (or have any conflict).
// Find connected components
//   Each component is a cluster.

// build index meas to hyp
unordered_map<int, vector<int>> measToHyps;

for (int h = 0; h < H; h++)
    for (int m : hyps[h].measurements)
        measToHyps[m].push_back(h);

// build conflicts
vector<vector<int>> conflicts(H);

// Measurement conflicts
for (auto &kv : measToHyps) {
    auto &v = kv.second;
    for (int i = 0; i < v.size(); i++)
        for (int j = i+1; j < v.size(); j++) {
            conflicts[v[i]].push_back(v[j]);
            conflicts[v[j]].push_back(v[i]);
        }
}

// Same-track conflicts (mutually exclusive subtree leaves)
for (auto &kv : trackToHyps) {
    auto &v = kv.second;
    for (int i = 0; i < v.size(); i++)
        for (int j = i+1; j < v.size(); j++) {
            conflicts[v[i]].push_back(v[j]);
            conflicts[v[j]].push_back(v[i]);
        }
}

// Solve each cluster
vector<int> selectCluster(const vector<int> &cluster,
                          const vector<TrackHyp> &hyps,
                          const vector<vector<int>> &conflicts)
{
    int n = cluster.size();
    MWIS solver(n);

    unordered_map<int,int> map;
    for (int i = 0; i < n; i++) {
        map[cluster[i]] = i;
        solver.set_weight(i, hyps[cluster[i]].weight);
    }

    for (int i = 0; i < n; i++) {
        int hi = cluster[i];
        for (int hj : conflicts[hi]) {
            if (map.count(hj))
                solver.add_edge(i, map[hj]);
        }
    }

    auto res = solver.solve();
    vector<int> selected;
    for (int idx : res.second)
        selected.push_back(cluster[idx]);

    return selected;
}

// Global selection loop
vector<int> globalSelection(const vector<TrackHyp> &hyps)
{
    auto conflicts = buildConflicts(hyps);
    auto clusters = buildClusters(hyps.size(), conflicts);

    vector<int> chosen;
    for (auto &c : clusters) {
        auto sel = selectCluster(c, hyps, conflicts);
        chosen.insert(chosen.end(), sel.begin(), sel.end());
    }
    return chosen;
}
