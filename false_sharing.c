/*
 * false_sharing.c — Phase 4, Experiment A
 *
 * Counts how many i in [1, N] have collatz_steps(i) > 100.
 *
 * Compile: gcc -O2 -fopenmp false_sharing.c -o false_sharing
 * Run:     ./false_sharing <N> <threads> <variant>
 *            variant = 1  -> naive hit_count[tid]++ (triggers false sharing)
 *            variant = 2  -> mitigated (padded struct, one cache line/thread)
 *
 * Run both variants at your CPU's max physical thread count, 3x each
 * (discard Run 1), and fill Table 2 with real times.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "collatz_common.h"

#define MAX_THREADS 128
#define CACHE_LINE 64

// Variant 2 helper: pad each counter out to a full cache line so no two
// threads' counters ever share a MESI cache line.
typedef struct {
    int count;
    char pad[CACHE_LINE - sizeof(int)];
} PaddedCounter;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <N> <threads> <variant:1|2>\n", argv[0]);
        return 1;
    }

    uint64_t N = strtoull(argv[1], NULL, 10);
    int threads = atoi(argv[2]);
    int variant = atoi(argv[3]);

    if (threads > MAX_THREADS) {
        fprintf(stderr, "threads > MAX_THREADS (%d)\n", MAX_THREADS);
        return 1;
    }
    omp_set_num_threads(threads);

    long total_hits = 0;
    double t0, t1;

    if (variant == 1) {
        // --- Naive array: adjacent ints packed into the same 64B line ---
        int hit_count[MAX_THREADS];
        memset(hit_count, 0, sizeof(hit_count));

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > 100) {
                hit_count[omp_get_thread_num()]++;
            }
        }
        t1 = omp_get_wtime();

        for (int t = 0; t < threads; t++) total_hits += hit_count[t];

    } else if (variant == 2) {
        // --- Mitigated: padded struct, one cache line per thread ---
        PaddedCounter hit_count[MAX_THREADS];
        memset(hit_count, 0, sizeof(hit_count));

        t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > 100) {
                hit_count[omp_get_thread_num()].count++;
            }
        }
        t1 = omp_get_wtime();

        for (int t = 0; t < threads; t++) total_hits += hit_count[t].count;

        // Alternative mitigation (uncomment to try instead of padding):
        // long reduction_hits = 0;
        // t0 = omp_get_wtime();
        // #pragma omp parallel for schedule(static) reduction(+:reduction_hits)
        // for (uint64_t i = 1; i <= N; i++)
        //     if (collatz_steps(i) > 100) reduction_hits++;
        // t1 = omp_get_wtime();
        // total_hits = reduction_hits;

    } else {
        fprintf(stderr, "variant must be 1 or 2\n");
        return 1;
    }

    double elapsed = t1 - t0;
    double throughput = (double)N / elapsed;

    printf("variant         = %d (%s)\n", variant, variant == 1 ? "naive/false-sharing" : "padded/mitigated");
    printf("N               = %llu\n", (unsigned long long)N);
    printf("threads         = %d\n", threads);
    printf("total_hits      = %ld\n", total_hits);
    printf("wall_time_sec   = %.6f\n", elapsed);
    printf("throughput_ips  = %.2f\n", throughput);

    return 0;
}
