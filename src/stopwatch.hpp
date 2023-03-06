#ifndef __STOPWATCH_HPP__
#define __STOPWATCH_HPP__

#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

class Stopwatch {
  protected:
    high_resolution_clock::time_point m_start, m_end;

  public:
    void start();
    void stop();
    high_resolution_clock::duration elapsed() const;
    void print_human_readable() const;
    static void print_elapsed_ms(float elapsed_ms);
};

#endif // __STOPWATCH_HPP__