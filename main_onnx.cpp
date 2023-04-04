// #include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/dnn.hpp>
// #include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

// Camera Pipeline

int main() {
    /* Initialize model */
    // Load the ONNX model
    std::string rektnet_modelPath = "../pretrained_kpt.onnx";
    Ort::SessionOptions session_options;
    Ort::Session session(session_options, model_path);

    /* Input */
    // get image
    std::string image_filepath = "../test_kpt.png";
    cv::Mat original_image = cv::imread(image_filepath);
    cv::Size original_image_size(original_image.cols, original_image.rows);
    // print size of image
    std::cout << "Original image size: " << original_image.rows << "x" << original_image.cols << std::endl;
    // preprocess image
    cv::Mat image;
    cv::Size image_size(80, 80);
    cv::resize(original_image, image, image_size);
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cv::Mat input = cv::dnn::blobFromImage(image);
    // Create an input tensor from the input data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image.data(), image.size(), image.data(), image.size());

    /* Inference */
    // Run the inference
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);
    std::vector<const char*> input_names = {input_name};
    std::vector<Ort::Value> input_values = {input_tensor};
    std::vector<const char*> output_names = {output_name};
    std::vector<Ort::Value> output_values = session.Run(Ort::RunOptions(), input_names, input_values, output_names);
    // Get the output data
    std::vector<float> output_data = output_values.front().GetTensorMutableData<float>();

    // Print the output data
    std::cout << "Output data:";
    for (float output : output_data) {
        std::cout << " " << output;
    }
    std::cout << std::endl;

    return 0;
}
