// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <map>

#define USE_PYTHON_SCIPY 0

#if USE_PYTHON_SCIPY > 0
#include <fstream>
#include <cstdio>
#include <array>
#include <memory>
#include <stdexcept>
#endif

//================================================================================
// Welford's online algorithm is used here to compute mean and variance in one
// pass. This is more efficient and numerically stable than the standard two-pass
// approach (first pass for mean, second for sum of squared differences).
//================================================================================
struct SampleStats {
    size_t n = 0;
    double M1 = 0.0; // Mean
    double M2 = 0.0; // Sum of squares of differences from the current mean

    // Welford's online algorithm to update stats with a new value
    void update(double x) {
        n++;
        double delta = x - M1;
        M1 += delta / n;
        double delta2 = x - M1;
        M2 += delta * delta2;
    }

    // Static factory method to compute stats for a whole vector
    static SampleStats compute(const std::vector<double>& v) {
        SampleStats s;
        for (double x : v) {
            s.update(x);
        }
        return s;
    }

    double mean() const {
        return M1;
    }

    double variance() const {
        if (n < 2) return 0.0;
        return M2 / (n - 1); // Sample variance
    }

    double stdev() const {
        return std::sqrt(variance());
    }
};

// Mean
double mean(const std::vector<double>& v) {
    if (v.empty()) return NAN;
    // Note: Can still be computed directly or via SampleStats
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Standard deviation
double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0; // Stdev is 0 for a single point, undefined for empty.
    
    // Using the more efficient single-pass method.
    return SampleStats::compute(v).stdev();
}

// Median
double median(std::vector<double> v) { // Pass by value is correct, as we modify it
    if (v.empty()) {
        return NAN;
    }
    size_t n = v.size();
    auto nth = v.begin() + n / 2;
    std::nth_element(v.begin(), nth, v.end());
    double med = *nth;

    if (n % 2 == 0) {
        auto max_it = std::max_element(v.begin(), nth);
        med = (*max_it + med) / 2.0;
    }
    return med;
}

// Percentile (0–100)
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    if (p < 0.0 || p > 100.0) return NAN;

    std::sort(v.begin(), v.end());
    if (p == 100.0) {
        return v.back();
    }
    double idx = (p/100.0) * (v.size()-1);
    size_t i = static_cast<size_t>(idx);
    double frac = idx - i;
    if (i + 1 < v.size())
        return v[i] * (1.0 - frac) + v[i+1] * frac;
    return v[i];
}

// Mann-Whitney U test (two-sided, approximate normal p-value)
double mann_whitney_pvalue(const std::vector<double>& A, const std::vector<double>& B) {
    if (A.empty() || B.empty()) return NAN;

    std::vector<std::pair<double, int>> combined;
    combined.reserve(A.size() + B.size());
    for (double x : A) combined.push_back({x, 0});
    for (double x : B) combined.push_back({x, 1});
    std::sort(combined.begin(), combined.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    //============================================================================
    // Handling Tied Ranks
    // All tied elements must be assigned the average of the ranks they would have
    // occupied. The variance also needs a correction term for ties.
    //============================================================================
    double R1 = 0.0;
    double tie_correction_term = 0.0;
    for (size_t i = 0; i < combined.size(); ) {
        size_t j = i;
        while (j < combined.size() && combined[j].first == combined[i].first) {
            j++;
        }
        double num_ties = j - i;
        double avg_rank = i + (num_ties + 1.0) / 2.0;

        if (num_ties > 1) {
            tie_correction_term += (num_ties * num_ties * num_ties - num_ties);
        }

        for (size_t k = i; k < j; k++) {
            if (combined[k].second == 0) { // Belongs to sample A
                R1 += avg_rank;
            }
        }
        i = j;
    }

    double n1 = A.size(), n2 = B.size();
    double U1 = R1 - n1 * (n1 + 1.0) / 2.0;
    double mu = n1 * n2 / 2.0;
    
    double N = n1 + n2;
    double sigma_sq_denom = N * (N - 1.0);
    if (sigma_sq_denom == 0) return 1.0; // Avoid division by zero if N < 2
    
    double sigma_sq = (n1 * n2 / 12.0) * (N + 1.0 - tie_correction_term / sigma_sq_denom);
    if (sigma_sq <= 0) return 1.0; // No variance, cannot compute z-score
    double sigma = std::sqrt(sigma_sq);

    // Apply continuity correction
    double z = (U1 - mu - 0.5 * std::copysign(1.0, U1 - mu)) / sigma;

    // The two-sided p-value is mathematically equivalent to erfc(|z|/sqrt(2)).
    // Using erfc() is more numerically stable for very small p-values.
    return std::erfc(std::fabs(z) / std::sqrt(2.0));
}
// Cohen’s d
double cohens_d(const SampleStats& statsA, const SampleStats& statsB) {
    size_t n1 = statsA.n, n2 = statsB.n;
    if (n1 + n2 <= 2) return NAN;

    double s1_sq = statsA.variance(), s2_sq = statsB.variance();
    double pooled_var = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2);
    if (pooled_var <= 0) return NAN;
    
    return (statsA.mean() - statsB.mean()) / std::sqrt(pooled_var);
}

// Overload for convenience
double cohens_d(const std::vector<double>& A, const std::vector<double>& B) {
    return cohens_d(SampleStats::compute(A), SampleStats::compute(B));
}

// Bootstrap confidence interval for mean difference
std::pair<double, double> bootstrap_CI(const std::vector<double>& A, const std::vector<double>& B,
                                     int n_bootstrap=10000, int seed=42) {
    if (A.empty() || B.empty()) return {NAN, NAN};
    
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distA(0, A.size() - 1);
    std::uniform_int_distribution<> distB(0, B.size() - 1);
    
    std::vector<double> diffs;
    diffs.reserve(n_bootstrap);

    for (int i = 0; i < n_bootstrap; i++) {
        double sumA = 0.0, sumB = 0.0;
        for (size_t j = 0; j < A.size(); j++) sumA += A[distA(gen)];
        for (size_t j = 0; j < B.size(); j++) sumB += B[distB(gen)];
        diffs.push_back(sumA / A.size() - sumB / B.size());
    }
    
    std::sort(diffs.begin(), diffs.end());
    
    size_t low_idx = static_cast<size_t>(n_bootstrap * 0.025);
    size_t high_idx = static_cast<size_t>(n_bootstrap * 0.975 -1); // -1 as it's an upper bound
    if (high_idx >= diffs.size()) high_idx = diffs.size() - 1;

    return {diffs[low_idx], diffs[high_idx]};
}

// TODO: remove this. It is used to verify C++ implementation only.
#if USE_PYTHON_SCIPY > 0
double call_python_mannwhitney(const std::vector<double>& A,
                               const std::vector<double>& B) {
    // Create a temporary Python script
    std::string filename = "tmp_mwu.py";
    std::ofstream pyfile(filename);
    if (!pyfile.is_open()) {
        throw std::runtime_error("Failed to create temporary Python file");
    }

    pyfile << "import scipy.stats as stats\n";
    pyfile << "A = " << "[";
    for (size_t i = 0; i < A.size(); ++i) {
        pyfile << A[i];
        if (i + 1 < A.size()) pyfile << ", ";
    }
    pyfile << "]\n";

    pyfile << "B = " << "[";
    for (size_t i = 0; i < B.size(); ++i) {
        pyfile << B[i];
        if (i + 1 < B.size()) pyfile << ", ";
    }
    pyfile << "]\n";

    pyfile << "print(stats.mannwhitneyu(A, B, use_continuity=True, alternative=\"two-sided\").pvalue)\n";
    pyfile.close();

    // Run Python script
    std::string cmd = "python3 " + filename;
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Optionally remove temporary file
    std::remove(filename.c_str());

    return std::stod(result);
}
#endif

void compare_statistics(const std::vector<double>& A, const std::vector<double>& B, std::string& name_A, std::string& name_B, bool is_small_better=true) {
    SampleStats statsA = SampleStats::compute(A);
    SampleStats statsB = SampleStats::compute(B);
    std::cout << "\n" << "----" << "\n";
    std::cout << "A:" << name_A << "\n";
    std::cout << "B:" << name_B << "\n";

    std::cout << "A mean: " << statsA.mean()
    << " std: " << statsA.stdev()
    << " median: " << median(A)
    << " p95: " << percentile(A, 95) << "\n";

    std::cout << "B mean: " << statsB.mean()
    << " std: " << statsB.stdev()
    << " median: " << median(B)
    << " p95: " << percentile(B, 95) << "\n";

    double mean_diff = statsA.mean() - statsB.mean();
    std::cout << "Mean diff (A-B): " << mean_diff << "\n";
    std::cout << "Relative improvement: " << (statsB.mean() - statsA.mean()) / statsB.mean() << "\n";

    std::cout << "Cohen's d=" << cohens_d(statsA, statsB) << "\n";

    auto [ci_low, ci_high] = bootstrap_CI(A, B);
    std::cout << "Bootstrap 95% CI for mean diff: ["
                << ci_low << ", " << ci_high << "]\n";

#if USE_PYTHON_SCIPY > 0
    double p_value = call_python_mannwhitney(A, B);
#else
    double p_value = mann_whitney_pvalue(A, B);
#endif

    std::cout << "Mann-Whitney pvalue=" << p_value << "\n";   
    if (p_value < 0.05f) {
        char winner = ((mean_diff < 0) ? (is_small_better? 'A' : 'B') : (is_small_better? 'B' : 'A'));
        std::cout << "Winner:" << (winner == 'A' ? name_A : name_B) << "\n";
    }
}