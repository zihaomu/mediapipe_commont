#include <memory>
#include <optional>
#include <string>

#include "image_generator.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"


namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_generator {

void test()
{
    auto options = absl::make_unique<ImageGeneratorOptions>();
    options->text2image_model_directory = "/Users/mzh/work/models/sd_1.5/v1-5-pruned-emaonly.bin";
    std::unique_ptr<ImageGenerator> imageGenerator = ImageGenerator::Create(std::move(options)).value();
    auto result = imageGenerator->Generate("cat", 1, 0);
    std::shared_ptr<cv::Mat> generatedImage = mediapipe::formats::MatView(&result->generated_image);

    cv::imshow("generatedImage", *generatedImage);
    cv::waitKey(0);
}

}  // namespace image_generator
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe


int main()
{
    mediapipe::tasks::vision::image_generator::test();
    std::cout<<"print count "<<std::endl;
    return 0;
}