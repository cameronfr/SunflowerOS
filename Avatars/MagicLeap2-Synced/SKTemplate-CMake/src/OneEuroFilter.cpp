#include <torch/script.h>
#include "debug.h"

// One Euro Filter in c++, transcribed (by Copilot, lol) from the following python code
// https://github.com/HoBeom/OneEuroFilter-Numpy

torch::Tensor smoothing_factor(torch::Tensor t_e, torch::Tensor cutoff) {
  torch::Tensor r = 2 * M_PI * cutoff * t_e;
  return r / (r + 1);
}

torch::Tensor exponential_smoothing(torch::Tensor a, torch::Tensor x, torch::Tensor x_prev) {
  return a * x + (1 - a) * x_prev;
}

class OneEuroFilter {
  public:
    OneEuroFilter() {}

    void Initialize(long long t0, torch::Tensor x0, torch::Tensor dx0, float min_cutoff, float beta, float d_cutoff) {
      // The parameters.
      this->min_cutoff = torch::ones_like(x0) * min_cutoff;
      this->beta = torch::ones_like(x0) * beta;
      this->d_cutoff = torch::ones_like(x0) * d_cutoff;
      // Previous values.
      this->x_prev = x0;
      this->dx_prev = dx0;
      this->t_prev = t0;

      if (x_prev.sizes() != dx_prev.sizes()) {
        throw std::invalid_argument("x_prev and dx_prev must have the same shape");
      }

      initialized = true;
    }

    void ChangeParams(float min_cutoff, float beta, float d_cutoff) {
      this->min_cutoff = torch::ones_like(x_prev) * min_cutoff;
      this->beta = torch::ones_like(x_prev) * beta;
      this->d_cutoff = torch::ones_like(x_prev) * d_cutoff;
    }

    void ResetHistory() {
      this->x_prev = torch::zeros_like(x_prev);
      this->dx_prev = torch::zeros_like(dx_prev);
      this->t_prev = 0;
    }

    torch::Tensor filter(long long t, torch::Tensor x) {
      if (x.sizes() != this->x_prev.sizes()) {
        throw std::invalid_argument("x must have the same shape as x_prev");
      }
      if (!initialized) {
        throw std::invalid_argument("OneEuroFilter must be initialized before filtering");
      }

      // Compute the filtered signal.
      float t_e_float = (t - this->t_prev) / 1e6; // Convert ns to seconds.
      // std::cout << "t_e_float: " << t_e_float << std::endl;
      torch::Tensor t_e = torch::ones_like(x) * t_e_float;
      // std::cout << "t_e avg: " << torch::mean(t_e) << std::endl;

      // The filtered derivative of the signal.
      torch::Tensor a_d = smoothing_factor(t_e, this->d_cutoff);
      // std::cout << "a_d avg: " << torch::mean(a_d) << std::endl;
      torch::Tensor dx = (x - this->x_prev) / t_e;
      // std::cout << "dx avg: " << torch::mean(dx) << std::endl;
      torch::Tensor dx_hat = exponential_smoothing(a_d, dx, this->dx_prev);
      // std::cout << "dx_hat avg: " << torch::mean(dx_hat) << std::endl;

      // The filtered signal.
      torch::Tensor cutoff = this->min_cutoff + (this->beta * torch::abs(dx_hat));
      // std::cout << "cutoff avg: " << torch::mean(cutoff) << std::endl;
      torch::Tensor a = smoothing_factor(t_e, cutoff);
      // std::cout << "a avg: " << torch::mean(a) << std::endl;
      torch::Tensor x_hat = exponential_smoothing(a, x, this->x_prev);
      // std::cout << "x_hat avg: " << torch::mean(x_hat) << std::endl;

      // Memorize the previous values.
      this->x_prev = x_hat;
      this->dx_prev = dx_hat;
      this->t_prev = t;

      return x_hat;
    }

  private:
    bool initialized = false;
    torch::IntArrayRef data_shape;
    torch::Tensor min_cutoff;
    torch::Tensor beta;
    torch::Tensor d_cutoff;
    torch::Tensor x_prev;
    torch::Tensor dx_prev;
    long long t_prev;
};