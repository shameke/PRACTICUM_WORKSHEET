/*
 * collatz_par.c — Phase 3: Multi-Thread Scaling
 *
 * Compile: gcc -O2 -fopenmp collatz_par.c -o collatz_par
 * Run:     ./collatz_par <N> <threads>
 *
 * For each k in {1, 2, 4, 8, 16 (if supported)}:
 *   run 3 times, discard Run 1, average Run 2 & Run 3 -> T_k
 *   S_emp(k) = T_seq / T_k   (T_seq from collatz_seq.c)
 *
 * Then derive p from k=2:  p = 2 * (1 - 1/S_emp(2))
 * and compute S_theo(k) = 1 / ((1-p) + p/k) for the other k values yourself
 * (e.g. in a spreadsheet or small script) — fill Table 1.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "collatz_common.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <threads>\n", argv[0]);
        return 1;
    }

    uint64_t N = strtoull(argv[1], NULL, 10);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    double t0 = omp_get_wtime();

    uint32_t max_steps = 0;
    uint64_t checksum = 0;

    #pragma omp parallel for schedule(static) reduction(max:max_steps) reduction(+:checksum)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) max_steps = s;
        checksum = (checksum + s) % MOD;
    }

    checksum %= MOD; // combine step: partial per-thread sums are each < MOD,
                      // but OpenMP's reduction(+:) adds them with plain +,
                      // so we fold back into MOD once at the end (this still
                      // matches the sequential checksum bit-for-bit, since
                      // modular addition is associative/commutative).

    double t1 = omp_get_wtime();

    printf("N               = %llu\n", (unsigned long long)N);
    printf("threads         = %d\n", threads);
    printf("max_steps       = %u\n", max_steps);
    printf("checksum        = %llu\n", (unsigned long long)checksum);
    printf("wall_time_sec   = %.6f\n", t1 - t0);

    return 0;
}
