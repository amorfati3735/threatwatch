/*
 * edge_infer.c — int8 forward pass for the ThreatWatch CNN, no dependencies.
 *
 * Compiles both for the ESP32 (Arduino) and on the host (cc) so that
 * tests/test_edge_parity.py can prove the on-device implementation computes the
 * same function as the PyTorch model it was exported from.
 *
 * Numerics
 * --------
 * Weights are int8 with a per-output-channel symmetric scale:
 *
 *      W_float[oc]  ~=  W_int8[oc] * scale[oc]
 *
 * so a convolution is
 *
 *      y[oc][t] = bias[oc] + scale[oc] * SUM_{ic,k} W_int8[oc][ic][k] * x[ic][t + k*d - p]
 *
 * and the int8 dot product is allowed to accumulate in float32.  Activations stay
 * float32 because the ESP32 has a hardware single-precision FPU: this removes all
 * fixed-point requantisation error while still cutting weight storage ~4x.  On an
 * AVR-class part (2 KB SRAM) the arena alone would not fit, which is why this
 * project's Uno sketch does sensing only.
 *
 * BatchNorm is folded into the weights and bias at export time, so the loops below
 * have no separate normalisation stage.
 */

#include "edge_model.h"

#include <math.h>
#include <string.h>

#if EDGE_N_CONV != 3
#error "edge_infer.c implements exactly 3 convolutional stages; regenerate edge_model.h"
#endif

/* One arena for every intermediate tensor.  Static, so it never lands on the
 * (much smaller) task stack. */
static float g_arena[EDGE_ARENA_FLOATS];
static float g_input[EDGE_INPUT_FLOATS];

#define BUF_A (g_arena)
#define BUF_B (g_arena + EDGE_A_FLOATS)
#define BUF_C (g_arena + EDGE_A_FLOATS + EDGE_B_FLOATS)

/* ------------------------------------------------------------------ */
/* primitives                                                          */
/* ------------------------------------------------------------------ */
static void relu_inplace(float *x, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* MaxPool1d(2), stride 2, non-overlapping, written back COMPACTED.
 *
 * The compaction matters: after pooling, channel c must live at offset
 * c * (len/2), because that is the stride the next convolution reads with.  An
 * earlier version pooled into the front of each channel's *original* len-stride
 * block, which left the tensor strided incorrectly and silently produced wrong
 * logits on every window.  tests/test_edge_parity.py caught it.
 *
 * Safe in place, in increasing channel order: destination channel c occupies
 * [c*out_len, (c+1)*out_len) and c*len is its source, and (c+1)*out_len <= c*len
 * for every c >= 1, so a destination never overlaps an unread source.  For c = 0
 * the regions coincide and the forward order is already safe because output j
 * reads inputs 2j and 2j+1, both at indices >= j. */
static void maxpool2_compact(float *x, int ch, int len)
{
    int c, j;
    int out_len = len / 2;
    for (c = 0; c < ch; ++c) {
        float *src = x + (size_t)c * len;
        float *dst = x + (size_t)c * out_len;
        for (j = 0; j < out_len; ++j) {
            float a = src[2 * j];
            float b = src[2 * j + 1];
            dst[j] = (a > b) ? a : b;
        }
    }
}

/* y[oc][t] = bias[oc] + scale[oc] * sum_{ic,k} W[oc][ic][k] * x[ic][t + k*d - p] */
static void conv1d_bn(const conv_spec *s, const float *x, int in_len, float *y)
{
    const int K = s->kernel;
    const int P = s->pad;
    const int D = s->dil;
    const int OC = s->out_ch;
    const int IC = s->in_ch;
    int oc, ic, k, t;

    for (oc = 0; oc < OC; ++oc) {
        const int8_t *w_oc = s->w + (size_t)oc * IC * K;
        const float scale = s->scale[oc];
        const float bias = s->bias[oc];
        float *y_oc = y + (size_t)oc * s->out_len;

        for (t = 0; t < s->out_len; ++t) {
            float acc = 0.0f;
            for (ic = 0; ic < IC; ++ic) {
                const float *x_ic = x + (size_t)ic * in_len;
                const int8_t *w_ic = w_oc + (size_t)ic * K;
                for (k = 0; k < K; ++k) {
                    int idx = t + k * D - P;
                    if (idx >= 0 && idx < in_len) {
                        acc += (float)w_ic[k] * x_ic[idx];
                    }
                }
            }
            y_oc[t] = bias + scale * acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* normalisation                                                       */
/* ------------------------------------------------------------------ */
/* Input arrives row-major as (T, C); we transpose into channel-major (C, T) and
 * normalise with exactly the mode the checkpoint was trained with. */
static void prepare_input(const float *window)
{
    int t, c;

#if EDGE_NORM_MODE == 1
    /* per-window z-score: no stored statistics, so this is exactly reproducible
     * on the device and cancels each wearer's personal offset. */
    for (c = 0; c < EDGE_N_CHANNELS; ++c) {
        float mean = 0.0f, var = 0.0f, sd;
        for (t = 0; t < EDGE_WINDOW; ++t) {
            mean += window[(size_t)t * EDGE_N_CHANNELS + c];
        }
        mean /= (float)EDGE_WINDOW;
        for (t = 0; t < EDGE_WINDOW; ++t) {
            float d = window[(size_t)t * EDGE_N_CHANNELS + c] - mean;
            var += d * d;
        }
        var /= (float)EDGE_WINDOW;
        sd = sqrtf(var);
        if (sd < 1e-6f) sd = 1.0f;
        for (t = 0; t < EDGE_WINDOW; ++t) {
            g_input[(size_t)c * EDGE_WINDOW + t] =
                (window[(size_t)t * EDGE_N_CHANNELS + c] - mean) / sd;
        }
    }
#else
    for (c = 0; c < EDGE_N_CHANNELS; ++c) {
        float mu = EDGE_NORM_MEAN[c];
        float sd = EDGE_NORM_STD[c];
        if (sd < 1e-6f) sd = 1.0f;
        for (t = 0; t < EDGE_WINDOW; ++t) {
            g_input[(size_t)c * EDGE_WINDOW + t] =
                (window[(size_t)t * EDGE_N_CHANNELS + c] - mu) / sd;
        }
    }
#endif
}

/* ------------------------------------------------------------------ */
/* forward pass                                                        */
/* ------------------------------------------------------------------ */
int edge_infer(const float *window, float *logits)
{
    int c, i, t;
    float feat[EDGE_C2_OUT_CH];
    float *last;

    prepare_input(window);

    /* stage 0: input -> A */
    conv1d_bn(&EDGE_CONVS[0], g_input, EDGE_WINDOW, BUF_A);
    relu_inplace(BUF_A, (size_t)EDGE_C0_OUT_CH * EDGE_C0_LEN);
    maxpool2_compact(BUF_A, EDGE_C0_OUT_CH, EDGE_C0_LEN);

    /* stage 1: A -> B   (in_len is the pooled length of stage 0) */
    conv1d_bn(&EDGE_CONVS[1], BUF_A, EDGE_C0_LEN / 2, BUF_B);
    relu_inplace(BUF_B, (size_t)EDGE_C1_OUT_CH * EDGE_C1_LEN);
    maxpool2_compact(BUF_B, EDGE_C1_OUT_CH, EDGE_C1_LEN);

    /* stage 2: B -> C */
    conv1d_bn(&EDGE_CONVS[2], BUF_B, EDGE_C1_LEN / 2, BUF_C);
    relu_inplace(BUF_C, (size_t)EDGE_C2_OUT_CH * EDGE_C2_LEN);
    last = BUF_C;

    /* global average pool over time  (AdaptiveAvgPool1d(1)) */
    for (c = 0; c < EDGE_C2_OUT_CH; ++c) {
        const float *p = last + (size_t)c * EDGE_C2_LEN;
        float acc = 0.0f;
        for (t = 0; t < EDGE_C2_LEN; ++t) {
            acc += p[t];
        }
        feat[c] = acc / (float)EDGE_C2_LEN;
    }

    /* dense head, also int8 with per-output-channel scale */
    for (i = 0; i < EDGE_N_CLASSES; ++i) {
        const int8_t *w = EDGE_HEAD_W + (size_t)i * EDGE_HEAD_IN;
        float acc = 0.0f;
        for (c = 0; c < EDGE_HEAD_IN; ++c) {
            acc += (float)w[c] * feat[c];
        }
        logits[i] = EDGE_HEAD_BIAS[i] + EDGE_HEAD_SCALE[i] * acc;
    }
    return 0;
}

void edge_softmax(float *logits)
{
    int i;
    float mx = logits[0];
    float sum = 0.0f;
    for (i = 1; i < EDGE_N_CLASSES; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    for (i = 0; i < EDGE_N_CLASSES; ++i) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    for (i = 0; i < EDGE_N_CLASSES; ++i) {
        logits[i] /= sum;
    }
}

int edge_static_bytes(void)
{
    return (int)(sizeof(g_arena) + sizeof(g_input));
}

/* ------------------------------------------------------------------ */
/* bring-up hooks                                                      */
/* ------------------------------------------------------------------ */
/* Compiled out of the firmware build.  They exist so this exact source can be
 * instrumented during bring-up (and in the host parity test) without editing it,
 * which would mean the tested code was no longer the deployed code. */
#ifdef EDGE_INFER_DEBUG
const float *edge_debug_input(void) { return g_input; }
const float *edge_debug_arena(void) { return g_arena; }

/* Normalise + run stage 0 only, leaving the UNPOOLED conv output in BUF_A.
 * Bring-up aid: lets a caller inspect the first layer in isolation. */
void edge_debug_conv0(const float *window)
{
    prepare_input(window);
    conv1d_bn(&EDGE_CONVS[0], g_input, EDGE_WINDOW, BUF_A);
    relu_inplace(BUF_A, (size_t)EDGE_C0_OUT_CH * EDGE_C0_LEN);
}
int edge_debug_conv0_len(void) { return EDGE_C0_LEN; }
int edge_debug_conv0_ch(void) { return EDGE_C0_OUT_CH; }
int edge_debug_offset(const char *which)
{
    if (which[0] == 'A') return 0;
    if (which[0] == 'B') return EDGE_A_FLOATS;
    return EDGE_A_FLOATS + EDGE_B_FLOATS;
}
#endif
