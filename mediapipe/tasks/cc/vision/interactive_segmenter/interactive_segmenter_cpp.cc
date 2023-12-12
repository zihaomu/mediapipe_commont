// This file contain the demo of interactive_segmenter

#include <memory>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "interactive_segmenter.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"


namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {

using ::mediapipe::Image;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::containers::NormalizedKeypoint;
using ::mediapipe::tasks::components::containers::RectF;
using ::mediapipe::tasks::vision::core::ImageProcessingOptions;

constexpr absl::string_view kTestDataDirectory{
    "/Users/mzh/work/github/mediapipe_commont/mediapipe/tasks/testdata/vision/"};
// constexpr absl::string_view kPtmModel{"magic_touch.tflite"};
constexpr absl::string_view kPtmModel{"ptm_512_hdt_ptm_woid.tflite"};

constexpr absl::string_view kCatsAndDogsJpg{"cats_and_dogs.jpg"};
// Golden mask for the dogs in cats_and_dogs.jpg.
constexpr absl::string_view kCatsAndDogsMaskDog1{"cats_and_dogs_mask_dog1.png"};
constexpr absl::string_view kCatsAndDogsMaskDog2{"cats_and_dogs_mask_dog2.png"};
constexpr absl::string_view kPenguinsLarge{"penguins_large.jpg"};
constexpr absl::string_view kPenguinsSmall{"penguins_small.jpg"};
constexpr absl::string_view kPenguinsSmallMask{"penguins_small_mask.png"};
constexpr absl::string_view kPenguinsLargeMask{"penguins_large_mask.png"};

constexpr float kGoldenMaskSimilarity = 0.97;

// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Since interactive segmenter has only 2 categories,
// the golden mask uses 0 or 255 for each pixel. // mask 只有两个数值
constexpr int kGoldenMaskMagnificationFactor = 255;

void test()
{
//   Image image = DecodeImageFromFile(JoinPath(kTestDataDirectory, kCatsAndDogsJpg)).value();
  Image image = DecodeImageFromFile("/Users/mzh/work/opencv_dev/mediapipe_reproduce/data/body_image/test3.jpeg").value();
  RegionOfInterest interaction_roi;
  interaction_roi.format = RegionOfInterest::Format::kKeyPoint;
//   interaction_roi.keypoint = NormalizedKeypoint{0.66, 0.66};
  interaction_roi.keypoint = NormalizedKeypoint{0.397, 0.412};
auto options = std::make_unique<InteractiveSegmenterOptions>();
   options->base_options.model_asset_path = JoinPath(kTestDataDirectory, kPtmModel);

  std::unique_ptr<InteractiveSegmenter> segmenter = InteractiveSegmenter::Create(std::move(options)).value();
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = 0;
  auto result = segmenter->Segment(image, interaction_roi, image_processing_options).value();
     cv::Mat actual_mask = mediapipe::formats::MatView(
       result.confidence_masks->at(0).GetImageFrameSharedPtr().get());

    cv::imshow("actual_mask", actual_mask);
    cv::waitKey(0);
}

void test2()
{
  Image image = DecodeImageFromFile(JoinPath(kTestDataDirectory, kCatsAndDogsJpg)).value();
  RegionOfInterest interaction_roi;
  interaction_roi.format = RegionOfInterest::Format::kKeyPoint;
  interaction_roi.keypoint = NormalizedKeypoint{0.66, 0.66};
auto options = std::make_unique<InteractiveSegmenterOptions>();
   options->base_options.model_asset_path = JoinPath(kTestDataDirectory, kPtmModel);

  std::unique_ptr<InteractiveSegmenter> segmenter = InteractiveSegmenter::Create(std::move(options)).value();
  ImageProcessingOptions image_processing_options;
  image_processing_options.rotation_degrees = 0;
  auto result = segmenter->Segment(image, interaction_roi, image_processing_options).value();
     cv::Mat actual_mask = mediapipe::formats::MatView(
       result.confidence_masks->at(0).GetImageFrameSharedPtr().get());

    cv::imshow("actual_mask", actual_mask);
    cv::waitKey(0);
}

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe

int main()
{
    mediapipe::tasks::vision::interactive_segmenter::test();
    return 0;
}