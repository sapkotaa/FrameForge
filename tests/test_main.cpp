#include <cstdio>
#include <cstring>
#include <exception>

#include "test_util.hpp"

int main(int argc, char** argv) {
    int ran = 0, failed_cases = 0;
    for (const auto& c : fftest::registry()) {
        if (argc > 1 && !std::strstr(c.name, argv[1])) continue;
        const int before = fftest::failures();
        std::printf("[ RUN  ] %s\n", c.name);
        std::fflush(stdout);
        try {
            c.fn();
        } catch (const fftest::Abort&) {
        } catch (const std::exception& e) {
            std::fprintf(stderr, "    unexpected exception: %s\n", e.what());
            ++fftest::failures();
        }
        ++ran;
        if (fftest::failures() != before) {
            ++failed_cases;
            std::printf("[ FAIL ] %s\n", c.name);
        } else {
            std::printf("[  OK  ] %s\n", c.name);
        }
    }
    std::printf("%d test(s), %d failed\n", ran, failed_cases);
    return failed_cases == 0 ? 0 : 1;
}


