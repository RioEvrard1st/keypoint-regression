#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <stdio.h>

// #include "RektNet/RektNet.hpp"

// Camera Pipeline

int main(int argc, char** argv){

    // printf("Torch Test \n");
    // torch::Tensor tensor = torch::rand({2, 3});
    // std::cout << tensor << std::endl;

    // printf("OpenCV Test \n");
    // std::string image_path = "../test_kpt.png";
    // cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    // if ( !image.data )
    // {
    //     printf("No image data \n");
    //     return -1;
    // }
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", image);
    // cv::waitKey(0);
    // return 0;

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
    cv::Mat original_image = cv::imread(image_filepath);
    cv::Size original_image_size(original_image.cols, original_image.rows);

    // print size of original image
    std::cout << "Original image size: " << original_image.rows << "x" << original_image.cols << "\n" << std::endl;

    // tensor
    torch::Tensor image_tensor = torch::from_blob(original_image.data, {1, 3, 80, 80}, torch::kFloat) / 255.0;
    
    std::cout << "Tensor size: " << image_tensor.sizes() << "\n" << std::endl;

    // convert to blob
    cv::Mat input(image_tensor.sizes()[2], image_tensor.sizes()[3], CV_32FC3, image_tensor.data_ptr<float>());

    std::cout << "Input size: " << input.cols << "x" << input.rows << std::endl;
    std::cout << "Input has " << input.channels() << " channels." << std::endl;
    std::cout << "Data type of input: " << input.type() << std::endl;
    std::cout << "Input size: " << input.size() << "\n" << std::endl;

    // set input to neural network
    model.setInput(input);

    // inference
    cv::Mat output = model.forward();

    std::cout << "Output size: " << output.rows << "x" << output.cols << std::endl;
    std::cout << "Output has " << output.channels() << " channels." << std::endl;
    std::cout << "Data type of Output: " << output.type() << std::endl;
    std::cout << "Output size: " << output.size() << "\n" << std::endl;

    // interpret output
    cv::Mat outputMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    
    //outputMat = outputMat.reshape((outputMat.size[0], 7, outputMat.size[2], outputMat.size[3]));
    
    // Check the size of the output matrix
    if (outputMat.size[0] == 1 && outputMat.size[1] == 7 && outputMat.size[2] == 80 && outputMat.size[3] == 80 && outputMat.type() == CV_32F) {
        std::cout << "outputMat has the correct dimensions and type" << std::endl;
    } else {
        std::cout << "outputMat does not have the correct dimensions adn/or type" << std::endl;
    }

    std::cout << "outputMat size: " << outputMat.rows << "x" << outputMat.cols << std::endl;
    std::cout << "outputMat has " << outputMat.channels() << " channels." << std::endl;
    std::cout << "Data type of outputMat: " << outputMat.type() << "\n" << std::endl;

    // Normalize the outputMat values to the range [0, 1]
    //cv::normalize(outputMat, outputMat, 0, 1, cv::NORM_MINMAX);
    outputMat.convertTo(outputMat, CV_8UC3);

    // Apply a heatmap colormap to the outputMat matrix
    cv::Mat heatmap;
    cv::applyColorMap(outputMat * 255, heatmap, cv::COLORMAP_JET);

    // Display the heatmap image
    cv::imshow("Heatmap", heatmap);
    cv::waitKey(0);

    /*

    // create heatmap by iterating over points
    cv::Mat heatmap = cv::Mat::zeros(image.size(), CV_32FC3);
    int num_kpts = 7;
    for (int i = 0; i < num_kpts; i++) {
        int x = outputMat.at<float>(0, i);
        int y = outputMat.at<float>(1, i);
        std::cout << "x: " << x << "  y: " << y << std::endl;
        cv::circle(heatmap, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
    }

    // print heatmap info (should be 80x80)
    std::cout << "Heatmap size: " << heatmap.rows << "x" << heatmap.cols << std::endl;
    std::cout << "Heatmap has " << heatmap.channels() << " channels." << std::endl;
    std::cout << "Data type of heatmap: " << heatmap.type() << std::endl;

    // superimpose heatmap on original image
    double alpha = 0.5;
    cv::Mat image_with_heatmap = image.clone();
    cv::addWeighted(image, alpha, heatmap, 1-alpha, 0, image_with_heatmap);

    // resize to original size
    cv::resize(image_with_heatmap, image_with_heatmap, original_image_size);

    // save heatmap in outputs folder
    std::string output_path = "test_kpt_output.jpg";
    cv::imwrite(output_path, image_with_heatmap * 255);

    // show image
    cv::imshow("Heatmap", image_with_heatmap);
    cv::waitKey(0);

    */

    return 0; 
}
