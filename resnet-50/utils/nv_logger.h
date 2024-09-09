#ifndef _NV_LOGGER_H
#define _NV_LOGGER_H

#include <iostream>
#include <NvInfer.h>

namespace trt_sample {

class Logger : public nvinfer1::ILogger {
public:
  explicit Logger(Severity severity = Severity::kINFO) : mReportableSeverity(severity) {}
  nvinfer1::ILogger& getTRTLogger() { return *this; }
  Severity getReportableSeverity() const { return mReportableSeverity; }
  void log(Severity severity, const char* msg) noexcept override {
    if (severity > mReportableSeverity) {
      return;
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        std::cerr << msg;
      case Severity::kWARNING:
        std::cout << msg;
      default:
        std::cout << msg;
    }
  }
  Severity mReportableSeverity;
};

extern Logger gLogger;

}   // namespace trt_sample

#endif  // _NV_LOGGER_H