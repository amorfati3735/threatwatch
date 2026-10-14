/*
 * edge_probe.ino — build-only probe.
 *
 * Links the REAL firmware/edge_model sources into an ESP32 binary so the
 * linker's own "Global variables use N bytes ... leaving M bytes" line gives
 * an honest, measured activation-arena + weight footprint on the board.
 * Nothing here is shipped; it exists so docs/EDGE_DEPLOYMENT.md can quote a
 * measurement instead of an estimate.
 */
#include <Arduino.h>
/* edge_model.h is the generated header: it defines EDGE_WINDOW, EDGE_ARENA_FLOATS
 * etc. and pulls in edge_infer.h (the hand-written declarations). */
#include "edge_model.h"

static float g_window[EDGE_WINDOW * EDGE_N_CHANNELS];
static float g_logits[EDGE_N_CLASSES];

void setup() {
  Serial.begin(115200);
  delay(200);

  for (int i = 0; i < EDGE_WINDOW * EDGE_N_CHANNELS; ++i) g_window[i] = 0.0f;

  unsigned long t0 = micros();
  int rc = edge_infer(g_window, g_logits);
  unsigned long dt = micros() - t0;

  Serial.println();
  Serial.printf("PROBE static_bytes   = %d\n", edge_static_bytes());
  Serial.printf("PROBE window         = %d x %d\n", EDGE_WINDOW, EDGE_N_CHANNELS);
  Serial.printf("PROBE arena_floats   = %d\n", EDGE_ARENA_FLOATS);
  Serial.printf("PROBE rc             = %d\n", rc);
  Serial.printf("PROBE first_logits   = %.4f %.4f %.4f\n",
                g_logits[0], g_logits[1], g_logits[2]);
  Serial.printf("PROBE infer_us       = %lu  (ESP32 @ 240 MHz)\n", dt);
}

void loop() {}
