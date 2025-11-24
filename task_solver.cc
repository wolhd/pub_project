// task_solver.cc
// Build with OR-Tools C++ (see compile instructions below).

#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <ortools/sat/cp_model.h>

using namespace operations_research;
using namespace sat;

struct Task {
  int cost;
  std::vector<int> workers;  // indices in [0..N-1]
};

// ---------- Utility: prune dominated tasks ----------
// Remove task i if there exists task j such that:
//   W_j ⊆ W_i  AND cost_j <= cost_i   (i is dominated by j)
static void PruneDominatedTasks(const std::vector<Task>& tasks,
                                std::vector<Task>& out_tasks,
                                std::vector<int>& out_orig_indices) {
  int T = tasks.size();
  std::vector<bool> keep(T, true);

  // Convert worker vectors to sets for subset checks
  std::vector<std::unordered_set<int>> wsets(T);
  for (int i = 0; i < T; ++i) {
    wsets[i].reserve(tasks[i].workers.size()*2 + 1);
    for (int w : tasks[i].workers) wsets[i].insert(w);
  }

  for (int i = 0; i < T; ++i) {
    if (!keep[i]) continue;
    for (int j = 0; j < T; ++j) {
      if (i == j || !keep[j]) continue;
      // if W_j subset of W_i and cost_j <= cost_i then i is dominated
      if (wsets[i].size() >= wsets[j].size() && tasks[j].cost <= tasks[i].cost) {
        bool j_subset_i = true;
        for (int w : wsets[j]) {
          if (wsets[i].find(w) == wsets[i].end()) { j_subset_i = false; break; }
        }
        if (j_subset_i) {
          // if j is strictly cheaper or equal and subset, drop i
          if (tasks[j].cost <= tasks[i].cost) {
            keep[i] = false;
            break;
          }
        }
      }
    }
  }

  // Build outputs and mapping to original indices
  out_tasks.clear();
  out_orig_indices.clear();
  for (int i = 0; i < T; ++i) {
    if (keep[i]) {
      out_tasks.push_back(tasks[i]);
      out_orig_indices.push_back(i);
    }
  }
}

// ---------- Build conflict graph and components ----------
// Two tasks conflict if their worker sets intersect (share >=1 worker).
static void BuildComponents(const std::vector<Task>& tasks,
                            std::vector<std::vector<int>>& components) {
  int T = tasks.size();
  // Map worker -> list of tasks that use it
  std::unordered_map<int, std::vector<int>> worker_to_tasks;
  for (int t = 0; t < T; ++t) {
    for (int w : tasks[t].workers) {
      worker_to_tasks[w].push_back(t);
    }
  }

  // Build adjacency list
  std::vector<std::vector<int>> adj(T);
  for (const auto& kv : worker_to_tasks) {
    const std::vector<int>& task_list = kv.second;
    for (size_t i = 0; i < task_list.size(); ++i) {
      for (size_t j = i + 1; j < task_list.size(); ++j) {
        int a = task_list[i], b = task_list[j];
        adj[a].push_back(b);
        adj[b].push_back(a);
      }
    }
  }

  // BFS for connected components
  std::vector<char> seen(T, 0);
  components.clear();
  for (int i = 0; i < T; ++i) {
    if (seen[i]) continue;
    std::vector<int> comp;
    std::queue<int> q;
    q.push(i);
    seen[i] = 1;
    while (!q.empty()) {
      int u = q.front(); q.pop();
      comp.push_back(u);
      for (int v : adj[u]) {
        if (!seen[v]) { seen[v] = 1; q.push(v); }
      }
    }
    components.push_back(std::move(comp));
  }
}

// ---------- Solve one component with OR-Tools CP-SAT ----------
// tasks: pruned tasks; comp: list of indices into tasks for this component
// N: number of workers (global)
static std::pair<std::vector<int>, int> SolveComponent(const std::vector<Task>& tasks,
                                                       const std::vector<int>& comp,
                                                       int N) {
  CpModelBuilder model;

  int Tloc = static_cast<int>(comp.size());
  std::vector<BoolVar> x(Tloc);
  for (int i = 0; i < Tloc; ++i) {
    x[i] = model.NewBoolVar();
  }

  // Per-worker constraints: a worker appears in at most one selected task
  for (int w = 0; w < N; ++w) {
    std::vector<BoolVar> involved;
    involved.reserve(4);
    for (int local_i = 0; local_i < Tloc; ++local_i) {
      int task_idx = comp[local_i];
      // check if worker w in tasks[task_idx]
      // linear scan is OK here (components are usually small)
      for (int tw : tasks[task_idx].workers) {
        if (tw == w) { involved.push_back(x[local_i]); break; }
      }
    }
    if (!involved.empty()) {
      model.AddLessOrEqual(LinearExpr::Sum(involved), 1);
    }
  }

  // Objective: minimize sum(cost * x)
  LinearExpr objective(0);
  for (int local_i = 0; local_i < Tloc; ++local_i) {
    objective += tasks[comp[local_i]].cost * x[local_i];
  }
  model.Minimize(objective);

  // Solve model
  const CpModelProto proto = model.Build();
  // Use default solver settings (fast for small problems).
  const CpSolverResponse response = Solve(proto);

  std::vector<int> chosen_local;
  int total_cost = 0;
  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    for (int local_i = 0; local_i < Tloc; ++local_i) {
      if (SolutionBooleanValue(response, x[local_i])) {
        chosen_local.push_back(comp[local_i]);  // store index into pruned tasks
        total_cost += tasks[comp[local_i]].cost;
      }
    }
  } else {
    // No feasible solution for this component (shouldn't happen normally).
    std::cerr << "Warning: no feasible solution for a component\n";
  }

  return {chosen_local, total_cost};
}

// ---------- Top-level solve with pruning + decomposition ----------
std::pair<std::vector<int>, int> SolveTasksWithDecomposition(int N,
                                                             const std::vector<Task>& tasks) {
  // 1) Prune dominated tasks
  std::vector<Task> pruned_tasks;
  std::vector<int> pruned_to_orig;  // pruned index -> original index
  PruneDominatedTasks(tasks, pruned_tasks, pruned_to_orig);

  // 2) Build components on pruned tasks
  std::vector<std::vector<int>> components;
  BuildComponents(pruned_tasks, components);

  // 3) Solve each component and collect chosen pruned indices
  std::vector<int> chosen_pruned;
  int total_cost = 0;
  for (const auto& comp : components) {
    if (comp.empty()) continue;
    auto res = SolveComponent(pruned_tasks, comp, N);
    for (int pidx : res.first) chosen_pruned.push_back(pidx);
    total_cost += res.second;
  }

  // 4) Map pruned indices back to original indices
  std::vector<int> chosen_original;
  chosen_original.reserve(chosen_pruned.size());
  for (int p : chosen_pruned) {
    chosen_original.push_back(pruned_to_orig[p]);
  }
  // Sort chosen indices for nicer output (optional)
  std::sort(chosen_original.begin(), chosen_original.end());

  return {chosen_original, total_cost};
}

// ---------- Example usage ----------
int main() {
  int N = 100;

  // Example: replace with your real data
  std::vector<Task> tasks;
  tasks.push_back({5, {0,1}});
  tasks.push_back({3, {2}});
  tasks.push_back({7, {1,3}});
  tasks.push_back({10, {50,51,2}});
  // Add some dominated example:
  tasks.push_back({6, {0,1,2}});  // might be dominated by a cheaper subset if present

  auto result = SolveTasksWithDecomposition(N, tasks);
  const std::vector<int>& chosen = result.first;
  int total_cost = result.second;

  std::cout << "Chosen tasks (original indices):";
  for (int t : chosen) std::cout << " " << t;
  std::cout << "\nTotal cost: " << total_cost << "\n";

  return 0;
}
