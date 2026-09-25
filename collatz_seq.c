/*
 * collatz_seq.c — Phase 2: Sequential Baseline
 *
 * Compile: gcc -O2 -fopenmp collatz_seq.c -o collatz_seq
 * Run:     ./collatz_seq <N>
 *
 * N must be computed from YOUR student ID:
 *   N = 10,000,000 + (last 4 digits of Student ID * 1,000)
 *
 * Benchmarking protocol (per worksheet Step 2.2):
 *   - Run the binary 3 times.
 *   - Discard Run 1 (cold cache / OS page allocation warmup).
 *   - T_seq = (Run 2 + Run 3) / 2.0   <-- compute this yourself from
 *     the three real wall-clock numbers this program prints.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "collatz_common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    uint64_t N = strtoull(argv[1], NULL, 10);

    double t0 = omp_get_wtime();

    uint32_t max_steps = 0;
    uint64_t max_i = 0;
    uint64_t checksum = 0;

    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) {
            max_steps = s;
            max_i = i;
        }
        checksum = (checksum + s) % MOD;
    }

    double t1 = omp_get_wtime();

    printf("N               = %llu\n", (unsigned long long)N);
    printf("max_steps       = %u (at i = %llu)\n", max_steps, (unsigned long long)max_i);
    printf("checksum        = %llu\n", (unsigned long long)checksum);
    printf("wall_time_sec   = %.6f\n", t1 - t0);

    return 0;
}
