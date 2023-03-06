#include <chrono>
#include <cstdlib>

#include "stopwatch.hpp"

/// Démarre un chronomètre
void Stopwatch::start() { m_start = high_resolution_clock::now(); }

/// Arrête un chronomètre
void Stopwatch::stop() { m_end = high_resolution_clock::now(); }

/// Renvoie la durée mesurée par le chronomètre
high_resolution_clock::duration Stopwatch::elapsed() const {
    return m_end - m_start;
}

// Affiche un temps en ms/us/s en fonction de sa valeur
void Stopwatch::print_human_readable() const {
    float elapsed_ms =
        duration_cast<duration<float, milliseconds::period>>(
            elapsed())
            .count();
    Stopwatch::print_elapsed_ms(elapsed_ms);
}

void Stopwatch::print_elapsed_ms(float elapsed_ms) {
    if (elapsed_ms >= 1000.0) { // Seconds
        elapsed_ms /= 1000.0f;
        printf("%.2fs", elapsed_ms);
    } else if (elapsed_ms < 1.0) { // Microseconds
        elapsed_ms *= 1000.0f;
        printf("%.2fµs", elapsed_ms);
    } else { // Milliseconds
        printf("%.2fms", elapsed_ms);
    }
}