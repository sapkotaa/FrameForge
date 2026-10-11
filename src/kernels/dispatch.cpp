#include "kernels/dispatch.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "impl.hpp"

namespace ff {

const char* to_string(Isa isa) noexcept {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Neon: return "neon";
        case Isa::Avx2: return "avx2";
    }
    return "?";
}

bool isa_supported(Isa isa) noexcept {
    switch (isa) {
        case Isa::Scalar: return true;
        case Isa::Neon:
#if defined(FF_HAVE_NEON)
            return true;
#else
            return false;
#endif
        case Isa::Avx2:
#if defined(FF_HAVE_AVX2) && defined(__x86_64__)
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") != 0;
#else
            return false;
#endif
    }
    return false;
}

std::vector<Isa> supported_isas() {
    std::vector<Isa> out;
    for (Isa i : {Isa::Scalar, Isa::Neon, Isa::Avx2})
        if (isa_supported(i)) out.push_back(i);
    return out;
}

Isa best_isa() noexcept {
    if (isa_supported(Isa::Avx2)) return Isa::Avx2;
    if (isa_supported(Isa::Neon)) return Isa::Neon;
    return Isa::Scalar;
}

const KernelTable& kernels_for(Isa isa) {
    if (!isa_supported(isa)) throw std::invalid_argument(std::string("ISA not supported here: ") + to_string(isa));
    switch (isa) {
        case Isa::Scalar: return detail::scalar_table();
#if defined(FF_HAVE_NEON)
        case Isa::Neon: return detail::neon_table();
#endif
#if defined(FF_HAVE_AVX2)
        case Isa::Avx2: return detail::avx2_table();
#endif
        default: break;
    }
    throw std::invalid_argument("ISA not compiled in");
}

namespace {
std::atomic<const KernelTable*> g_active{nullptr};

const KernelTable& initial_table() {
    if (const char* env = std::getenv("FRAMEFORGE_ISA")) {
        for (Isa i : {Isa::Scalar, Isa::Neon, Isa::Avx2})
            if (std::strcmp(env, to_string(i)) == 0 && isa_supported(i)) return kernels_for(i);
    }
    return kernels_for(best_isa());
}
}  // namespace

const KernelTable& kernels() noexcept {
    const KernelTable* t = g_active.load(std::memory_order_acquire);
    if (!t) {
        t = &initial_table();  // idempotent: racing threads compute the same answer
        g_active.store(t, std::memory_order_release);
    }
    return *t;
}

void set_active_isa(Isa isa) {
    g_active.store(&kernels_for(isa), std::memory_order_release);
}

}  // namespace ff
