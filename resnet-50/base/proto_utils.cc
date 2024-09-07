#include "utils/proto_utils.h"

namespace trt_sample {

bool LoadProtoFromTextFile(const std::string &path, google::protobuf::Message *msg) {
    int fd = open(path.c_str(), O_RDONLY);
    google::protobuf::io::FileInputStream file_in(fd);
    bool success = google::protobuf::TextFormat::Parse(&file_in, msg);
    close(fd);
    return success;
}

}   // namespace trt_sample

