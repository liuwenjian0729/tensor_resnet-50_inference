#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <sstream>
#include <memory>
#include "utils/nv_logger.h"
#include "utils/nv_utils.h"

namespace trt_sample {

Logger gLogger;

std::string dims2str(const nvinfer1::Dims &dims) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << dims.d[i];
  }
  ss << "]";
  return ss.str();
}

bool onnx2trtEngine(const std::string& trt_file, const std::string& onnx_file) {
    // 1. create builder
    std::unique_ptr<nvinfer1::IBuilder> builder_ptr(nvinfer1::createInferBuilder(gLogger));
    auto builder = builder_ptr.get();
    if (builder == nullptr) {
        return false;
    }
    
    // 2. create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network_ptr(builder->createNetworkV2(explicitBatch));
    auto network = network_ptr.get();
    if (network == nullptr) {
        return false;
    }

    // 3. create parser
    std::unique_ptr<nvonnxparser::IParser> parser_ptr(nvonnxparser::createParser(*network, gLogger));
    auto parser = parser_ptr.get();
    if (parser == nullptr) {
        return false;
    }
    parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // 4. set config
    builder->setMaxBatchSize(1);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

    // 5. build engine
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ptr(builder->buildEngineWithConfig(*network, *config));
    auto engine = engine_ptr.get();
    if (engine == nullptr) {
        return false;
    }

    // 6. serialize engine
    std::unique_ptr<nvinfer1::IHostMemory> serialize_mem(engine->serialize());
    auto serializedModel = serialize_mem.get();
    std::ofstream ofs(trt_file, std::ios::binary);
    if (ofs) {
        ofs.write(static_cast<char*>(serializedModel->data()), serializedModel->size());
        ofs.close();
    }
    
    return true;
}

nvinfer1::ICudaEngine* loadEngine(const std::string& trt_file, nvinfer1::IRuntime* runtime) {
    std::ifstream ifs(trt_file, std::ios::binary);
    if (!ifs.good()) {
        std::cerr << "read " << trt_file << " error!" << std::endl;
        return nullptr;
    }

    // get file size
    ifs.seekg(0, ifs.end);
    long size = ifs.tellg();
    ifs.seekg(0, ifs.beg);

    // read data
    std::vector<char> buffer(size);
    ifs.read(buffer.data(), size);
    ifs.close();

    // create runtime
    return runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
}

} // namespace name
