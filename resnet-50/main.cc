#include <iostream>
#include <sstream>
#include <memory>
#include "engine/trt_engine.h"

using namespace trt_sample;

int main(int argc, char *argv[]) {

    if (argc <= 1) {
        std::cout<<"input model config file path" << std::endl;
        return -1;
    }

    const std::string config_file(argv[1]);

    // 1. create engine
    std::unique_ptr<TrtEngine> engine_ptr(new TrtEngine());

    if (!engine_ptr->init(config_file)) {
        std::cerr << "Failed to create TrtEngine object" << std::endl;
        return -1;
    }

    // 2. run engine
    engine_ptr->run();

    return 0;
}

