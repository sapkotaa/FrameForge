#include "quality/bdrate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ff {
namespace {

struct Fit {
    std::vector<double> c;  // polynomial in (x - x0): c[0] + c[1] t + c[2] t^2 + ...
    double x0 = 0;
    double xmin = 0, xmax = 0;
};

// Least squares via normal equations with partial-pivot Gaussian elimination.
Fit fit_poly(const std::vector<RdPoint>& pts) {
    if (pts.size() < 2) throw std::invalid_argument("bd_rate: need at least 2 points per curve");
    const size_t deg = std::min<size_t>(3, pts.size() - 1);
    const size_t n = deg + 1;

    Fit f;
    f.xmin = f.xmax = pts[0].quality;
    for (const auto& p : pts) {
        f.xmin = std::min(f.xmin, p.quality);
        f.xmax = std::max(f.xmax, p.quality);
        f.x0 += p.quality;
    }
    f.x0 /= static_cast<double>(pts.size());  // centre x for conditioning

    std::vector<std::vector<double>> m(n, std::vector<double>(n + 1, 0.0));
    for (const auto& p : pts) {
        const double t = p.quality - f.x0;
        const double y = std::log10(p.kbps);
        std::vector<double> pw(2 * n, 1.0);
        for (size_t i = 1; i < 2 * n; ++i) pw[i] = pw[i - 1] * t;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) m[i][j] += pw[i + j];
            m[i][n] += pw[i] * y;
        }
    }
    for (size_t col = 0; col < n; ++col) {
        size_t piv = col;
        for (size_t r = col + 1; r < n; ++r)
            if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) piv = r;
        std::swap(m[col], m[piv]);
        if (std::fabs(m[col][col]) < 1e-12) throw std::runtime_error("bd_rate: degenerate curve (duplicate quality values?)");
        for (size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double k = m[r][col] / m[col][col];
            for (size_t c = col; c <= n; ++c) m[r][c] -= k * m[col][c];
        }
    }
    f.c.resize(n);
    for (size_t i = 0; i < n; ++i) f.c[i] = m[i][n] / m[i][i];
    return f;
}

double integrate(const Fit& f, double lo, double hi) {
    auto antiderivative = [&](double x) {
        const double t = x - f.x0;
        double acc = 0, pw = t;
        for (size_t i = 0; i < f.c.size(); ++i) {
            acc += f.c[i] * pw / static_cast<double>(i + 1);
            pw *= t;
        }
        return acc;
    };
    return antiderivative(hi) - antiderivative(lo);
}

}  // namespace

double bd_rate(const std::vector<RdPoint>& anchor, const std::vector<RdPoint>& test) {
    const Fit a = fit_poly(anchor);
    const Fit t = fit_poly(test);
    const double lo = std::max(a.xmin, t.xmin);
    const double hi = std::min(a.xmax, t.xmax);
    if (hi <= lo) return std::numeric_limits<double>::quiet_NaN();
    const double avg_diff = (integrate(t, lo, hi) - integrate(a, lo, hi)) / (hi - lo);
    return (std::pow(10.0, avg_diff) - 1.0) * 100.0;
}

}  // namespace ff
