/*
 * edge_infer.h — dependency-free int8 forward pass for the ThreatWatch CNN.
 *
 * This file is hand-written and stable.  Everything it consumes that is specific
 * to a trained network lives in the *generated* edge_model.h, which includes this
 * header.  Nothing here allocates, uses the heap, or depends on any library beyond
 * <math.h> and <stdint.h>, so the same source compiles unchanged for
 *   - the ESP32 sketch,   and
 *   - the host-side parity test (tests/test_edge_parity.py, compiled with cc).
 *
 * That shared-source property is the point: the test proves the code that runs on
 * the board computes the same function as PyTorch, rather than testing a
 * re-implementation of it.
 */

#ifndef EDGE_INFER_H
#define EDGE_INFER_H

#include <stdint.h>

/* The Arduino build compiles the .ino as C++ and edge_infer.c as C, so without
 * these guards the sketch gets mangled names ("undefined reference to
 * 'edge_infer(float const*, float*)'") while the C symbols stay unmangled.
 * Inert on the host, where tests/test_edge_parity.py compiles this as C99. */
#ifdef __cplusplus
extern "C" {
#endif

/* One convolution stage: weights are int8 with a per-output-channel scale.
 * Bias is float32 and already includes any folded BatchNorm offset. */
typedef struct {
    int in_ch;
    int out_ch;
    int kernel;
    int pad;
    int dil;
    int pool;        /* 0 = no pooling, 2 = MaxPool1d(2) applied after ReLU */
    int out_len;     /* output length for this stage, fixed by EDGE_WINDOW   */
    const int8_t *w; /* [out_ch][in_ch][kernel]                             */
    const float *scale;
    const float *bias;
} conv_spec;

/**
 * Run the network on one window.
 *
 * @param window  input, row-major (EDGE_WINDOW x EDGE_N_CHANNELS), raw units
 *                (i.e. before normalisation — this function normalises internally
 *                using exactly the mode the checkpoint was trained with).
 * @param logits  output, EDGE_N_CLASSES raw scores (apply softmax for
 *                probabilities; the argmax is unchanged either way).
 * @return 0 on success.
 */
int edge_infer(const float *window, float *logits);

/** Softmax an EDGE_N_CLASSES logit vector in place (host convenience). */
void edge_softmax(float *logits);

/** Static RAM this module occupies, in bytes (arena + input scratch). */
int edge_static_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* EDGE_INFER_H */
