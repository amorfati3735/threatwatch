/*
 * heap_probe.ino — build-only probe (no edge model linked).
 *
 * Purpose: report how much contiguous heap the board really has, so
 * docs/EDGE_DEPLOYMENT.md can state whether a heap-backed activation arena
 * would fit — measured, not assumed.  Nothing is leaked: every probe
 * allocation is freed before the report is printed.
 */
#include <Arduino.h>

static size_t largestBlock() {
  for (size_t n = 512u * 1024u; n >= 4096u; n >>= 1) {
    void *p = malloc(n);
    if (p) { free(p); return n; }
  }
  return 0;
}

void setup() {
  Serial.begin(115200);
  delay(300);

  const size_t best = largestBlock();

  Serial.println();
  Serial.printf("HEAP largest_block = %u bytes\n", (unsigned)best);
  Serial.printf("HEAP free_heap     = %u bytes\n", (unsigned)ESP.getFreeHeap());
  Serial.printf("HEAP min_free      = %u bytes\n", (unsigned)ESP.getMinFreeHeap());
  Serial.printf("HEAP max_alloc     = %u bytes\n", (unsigned)ESP.getMaxAllocHeap());
}

void loop() {}
