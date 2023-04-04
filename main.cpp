#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <stdio.h>

int main(int argc, char** argv){

    // define RektNet
    std::string rektnet_modelPath = "../pretrained_kpt.onnx";
    auto model = cv::dnn::readNetFromONNX(rektnet_modelPath);
    if (model.empty()) {
        std::cerr << "Failed to load network!" << "\n" << std::endl;
        return -1;
    } else {
        std::cout << "Network loaded successfully." << "\n" << std::endl;
    }

    // get image
    std::string image_filepath = "../test_kpt.png";
    cv::Mat image = cv::imread(image_filepath);
    cv::Size image_size(image.cols, image.rows);

    // print size of original image
    std::cout << "Original image size: " << image.size << "\n" << std::endl;

    // convert to input blob (1x3x80x80)
    cv::Mat input = cv::dnn::blobFromImage(image, 1/255.0, cv::Size(80, 80), cv::Scalar(0,0,0), true, false);

    // Input blob (1x3x80x80)
    std::cout << "Input size: " << input.size << "\n" << std::endl;

    // set input to neural network
    model.setInput(input);

    // inference
    cv::Mat output = model.forward();

    // output blob (1x7x80x80)
    std::cout << "Output size: " << output.size << "\n" << std::endl;

    // loop over the keypoints and draw circles at their locations
    for (int i = 0; i < output.size[1]; i++) {
        // extract the x and y coordinates of the keypoint
        int x = static_cast<int>(output.at<float>(0, i, 0) * image.cols);
        int y = static_cast<int>(output.at<float>(0, i, 1) * image.rows);

        std::cout << "Keypoint " << i+1 << ": (" << x << ", " << y << ") "<< std::endl;

        // draw a circle at the keypoint location
        cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }

    // display the image with the keypoints
    cv::imshow("Output", image);
    
    // interpret output
    cv::Mat outputMat(output.size[2], output.size[3], CV_8UC3, output.ptr<float>());
    cv::normalize(outputMat, outputMat, 0, 1, cv::NORM_MINMAX);

    // Check the size of the output matrix
    std::cout << "outputMat size: " << outputMat.size << "\n" << std::endl;

    // Apply a heatmap colormap to the outputMat matrix
    cv::Mat heatmap;
    cv::applyColorMap(outputMat*255, heatmap, cv::COLORMAP_JET);

    cv::imshow("Heatmap", heatmap);
    cv::waitKey(0);

    // save image in outputs folder
    // std::string output_path = "test_kpt_output.jpg";
    // cv::imwrite(output_path, image_with_heatmap * 255);

    return 0; 
}