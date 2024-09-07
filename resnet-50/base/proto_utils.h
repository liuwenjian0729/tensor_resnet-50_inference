#ifndef _PROTO_UTILS_H
#define _PROTO_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace trt_sample {

bool LoadProtoFromTextFile(const std::string &path, google::protobuf::Message *msg);

}   // namespace trt_sample

#endif  // _PROTO_UTILS_H
