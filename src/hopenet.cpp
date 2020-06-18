#include "ncnn_config.h"
#ifdef GPU_SUPPORT
  #include "gpu.h"
  #include "gpu_support.h"
#endif
#include "ncnn_hopenet.h"

ncnnHopeNet engine;
HeadPose    head;

int main(int argc, char** argv)
{
  int gpu_device = 0;

  if (argc != 2)
  {
      fprintf(stderr, "Usage: %s [imagepath] optional: [gpudevice]\n", argv[0]);
      return -1;
  }

  const char* imagepath = argv[1];
  if (argc >2 ) gpu_device = atoi(argv[2]);

  cv::Mat image = cv::imread(imagepath, 1);
  if (image.empty())
  {
      fprintf(stderr, "cv::imread %s failed\n", imagepath);
      return -1;
  }

#ifndef GPU_SUPPORT
  std::cout << " running on CPU\r\n";
#endif
#ifdef GPU_SUPPORT
  std::cout << " with GPU_SUPPORT, selected gpu_device: " << gpu_device << "\r\n";
  g_vkdev = ncnn::get_gpu_device(selectGPU(gpu_device));
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  engine.neuralnet.opt.use_vulkan_compute = true;
  engine.neuralnet.set_vulkan_device(g_vkdev);
#endif

std::string path = "../model/";
engine.neuralnet.load_param((path+"hopenet.param").c_str());
engine.neuralnet.load_model((path+"hopenet.bin").c_str());

#ifdef GPU_SUPPORT
  ncnn::create_gpu_instance();
#endif

engine.initialize();

cv::Rect roi; // taking the whole image ( assuming it's cropped to BB )
roi.x = 1;
roi.y = 1;
roi.width  = image.cols-1;
roi.height = image.rows-1;

engine.detect(image, roi, head);

std::cout << "Head roll : \t" << std::fixed << std::setw( 6 ) << std::setprecision( 3 ) << head.roll << "\r\n";
std::cout << "Head pitch: \t" << std::fixed << std::setw( 6 ) << std::setprecision( 3 ) << head.pitch << "\r\n";
std::cout << "Head yaw  : \t" << std::fixed << std::setw( 6 ) << std::setprecision( 3 ) << head.yaw << "\r\n";

engine.draw();

#ifdef GPU_SUPPORT
  ncnn::destroy_gpu_instance();
#endif

  return 0;
}
