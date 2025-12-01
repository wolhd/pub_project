vector<vector<int>> buildClusters(int n, vector<vector<int>> &conflicts)
{
    vector<vector<int>> clusters;
    vector<bool> visited(n, false);

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            vector<int> comp;
            stack<int> st;
            st.push(i);

            while (!st.empty()) {
                int u = st.top(); st.pop();
                if (visited[u]) continue;
                visited[u] = true;
                comp.push_back(u);

                for (int v : conflicts[u]) {
                    if (!visited[v]) st.push(v);
                }
            }

            clusters.push_back(comp);
        }
    }

    return clusters;
}

To fill the conflicts adjacency list, use measurement sharing:

Method A — Based on shared measurement IDs (most common)

For each measurement m:

Find all hypotheses that use measurement m

Connect all of them

Pseudocode:
unordered_map<int, vector<int>> measToHyps;

for (int h = 0; h < N; h++)
    for (int m : hypothesis[h].measurements)
        measToHyps[m].push_back(h);

vector<vector<int>> conflicts(N);

for (auto &kv : measToHyps) {
    auto &vec = kv.second;
    for (int i = 0; i < vec.size(); i++)
        for (int j = i+1; j < vec.size(); j++) {
            int a = vec[i], b = vec[j];
            conflicts[a].push_back(b);
            conflicts[b].push_back(a);
        }
}
