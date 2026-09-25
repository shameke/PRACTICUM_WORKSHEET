/*
 * scheduling.c — Phase 4, Experiment B
 *
 * Uses schedule(runtime), so the actual clause+chunk is chosen at run
 * time via the OMP_SCHEDULE environment variable -- no recompiling
 * needed for each row of Table 3.
 *
 * Compile: gcc -O2 -fopenmp scheduling.c -o scheduling
 * Run examples (use your own max physical thread count for <threads>):
 *
 *   OMP_SCHEDULE="static"        ./scheduling <N> <threads>
 *   OMP_SCHEDULE="static,1000"   ./scheduling <N> <threads>
 *   OMP_SCHEDULE="dynamic,100"   ./scheduling <N> <threads>
 *   OMP_SCHEDULE="dynamic,10000" ./scheduling <N> <threads>
 *   OMP_SCHEDULE="guided"        ./scheduling <N> <threads>
 *
 * Run each 3x (discard Run 1), fill Table 3 with real times.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "collatz_common.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: OMP_SCHEDULE=<clause> %s <N> <threads>\n", argv[0]);
        return 1;
    }

    uint64_t N = strtoull(argv[1], NULL, 10);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    double t0 = omp_get_wtime();

    uint32_t max_steps = 0;
    uint64_t checksum = 0;

    #pragma omp parallel for schedule(runtime) reduction(max:max_steps) reduction(+:checksum)
    for (uint64_t i = 1; i <= N; i++) {
        uint32_t s = collatz_steps(i);
        if (s > max_steps) max_steps = s;
        checksum = (checksum + s) % MOD;
    }
    checksum %= MOD;

    double t1 = omp_get_wtime();

    const char *sched_env = getenv("OMP_SCHEDULE");
    printf("OMP_SCHEDULE    = %s\n", sched_env ? sched_env : "(default/static)");
    printf("N               = %llu\n", (unsigned long long)N);
    printf("threads         = %d\n", threads);
    printf("max_steps       = %u\n", max_steps);
    printf("checksum        = %llu\n", (unsigned long long)checksum);
    printf("wall_time_sec   = %.6f\n", t1 - t0);

    return 0;
}
