# Copyright 2023 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

# Docs for Mediapipe Tasks Interactive Segmenter
# TODO: add doc link.
cc_library(
    name = "interactive_segmenter",
    srcs = ["interactive_segmenter.cc"],
    hdrs = ["interactive_segmenter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":interactive_segmenter_graph",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/cc/components/containers:keypoint",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/vision/core:base_vision_task_api",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/core:vision_task_api_factory",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_result",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "interactive_segmenter_graph",
    srcs = ["interactive_segmenter_graph.cc"],
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:set_alpha_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator_cc_proto",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",
        "//mediapipe/tasks/cc/core:model_task_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter:image_segmenter_graph",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_cc_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
            "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        ],
    }),
    alwayslink = 1,
)

cc_binary(
    name = "interactive_segmenter_cc_demo",
    srcs = ["interactive_segmenter_cpp.cc"],
    deps = [
        ":interactive_segmenter",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:set_alpha_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator_cc_proto",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_log",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//mediapipe/framework/deps:file_path",
    ],
)

cc_test(
    name = "interactive_segmenter_cc_test",
    srcs = ["interactive_segmenter_test.cc"],
    # hdrs = ["interactive_segmenter.h"],
    # data = [
    #     "//mediapipe/tasks/testdata/vision:bert_text_classifier_models",
    #     "//mediapipe/tasks/testdata/text:text_classifier_models",
    # ],
    deps = [
        ":interactive_segmenter",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/tool:test_util",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:set_alpha_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator",
        "//mediapipe/calculators/util:flat_color_image_calculator_cc_proto",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_log",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:graph_builder_utils",
        "//mediapipe/util:label_map_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:test_util",
        "//mediapipe/framework/deps:file_path",
    ],
)
