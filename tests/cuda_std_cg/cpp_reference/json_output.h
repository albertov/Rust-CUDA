#ifndef JSON_OUTPUT_H
#define JSON_OUTPUT_H

#include <stdio.h>

// Helper functions for JSON output formatting

inline void json_print_array_int(const char *name, const int *arr, int size) {
  printf("  \"%s\": [", name);
  for (int i = 0; i < size; i++) {
    printf("%d%s", arr[i], (i < size - 1) ? ", " : "");
  }
  printf("]");
}

inline void json_print_field_int(const char *name, int value,
                                 bool last = false) {
  printf("  \"%s\": %d%s\n", name, value, last ? "" : ",");
}

inline void json_print_field_string(const char *name, const char *value,
                                    bool last = false) {
  printf("  \"%s\": \"%s\"%s\n", name, value, last ? "" : ",");
}

inline void json_print_field_bool(const char *name, bool value,
                                  bool last = false) {
  printf("  \"%s\": %s%s\n", name, value ? "true" : "false", last ? "" : ",");
}

#endif // JSON_OUTPUT_H
