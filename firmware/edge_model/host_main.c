/*
 * host_main.c — host-side driver for the SAME edge_infer.c that runs on the ESP32.
 *
 * tests/test_edge_parity.py compiles edge_infer.c together with this file using
 * `cc`, feeds it the windows the test also pushes through PyTorch, and compares
 * the logits.  The point is that this is not a reimplementation of the on-device
 * code being tested — it is the on-device code.
 *
 * Protocol (stdin):  <n_windows>\n  then n_windows * EDGE_WINDOW * EDGE_N_CHANNELS floats
 * Protocol (stdout): n_windows * EDGE_N_CLASSES logits, plus one timing line
 */

#include "edge_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int edge_infer(const float *window, float *logits);

int main(void)
{
    int n, w, i;
    float *buf = NULL;
    float logits[EDGE_N_CLASSES];

    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "bad window count\n");
        return 2;
    }
    buf = (float *)malloc(sizeof(float) * EDGE_WINDOW * EDGE_N_CHANNELS);
    if (!buf) {
        fprintf(stderr, "oom\n");
        return 2;
    }

    for (w = 0; w < n; ++w) {
        for (i = 0; i < EDGE_WINDOW * EDGE_N_CHANNELS; ++i) {
            if (scanf("%f", &buf[i]) != 1) {
                fprintf(stderr, "short input at window %d\n", w);
                free(buf);
                return 2;
            }
        }
        edge_infer(buf, logits);
        for (i = 0; i < EDGE_N_CLASSES; ++i) {
            printf("%.6f%c", logits[i], (i + 1 == EDGE_N_CLASSES) ? '\n' : ' ');
        }
    }

    /* one repeat pass, timed — reported separately from the parity output */
    {
        clock_t t0 = clock();
        int reps = 20;
        for (w = 0; w < reps; ++w) {
            edge_infer(buf, logits);
        }
        t0 = clock() - t0;
        fprintf(stderr, "HOST_MS_PER_INFER %.4f\n",
                (double)t0 * 1000.0 / (double)CLOCKS_PER_SEC / (double)reps);
    }

    free(buf);
    return 0;
}
