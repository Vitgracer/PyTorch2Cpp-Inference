#include <torch/script.h>
#include <torch/data/transforms.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "params.h"

torch::Tensor get_image_tensor(std::string image_path) {   
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    std::cout << "Image loaded successfully.\n";

    // Resize to match training
    cv::resize(img, img, cv::Size(28, 28));

    // Convert to float and normalize to [0, 1]
    img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

    torch::Tensor tensor_image = torch::from_blob(
        img.data, { 1, 1, 28, 28 }, torch::kFloat32).clone();

    tensor_image = torch::data::transforms::Normalize<>({ 0.5 }, { 0.5 })(tensor_image);
    return tensor_image;
}

int main() {
    try {
        torch::Tensor tensor_image = get_image_tensor(IMAGE_PATH);

        torch::jit::script::Module model = torch::jit::load(MODEL_PATH, torch::kCPU);
        model.eval();  // Set to inference mode

        std::cout << "Model loaded successfully.\n";

        // Run inference
        std::cout << "Input shape: " << tensor_image.sizes() << std::endl;
        torch::NoGradGuard no_grad;
        auto output = model.forward({ tensor_image }).toTensor();

        // Process results
        auto prediction = output.argmax(1);
        int predicted_class = prediction.item<int>();

        std::cout << "Predicted class: " << predicted_class << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        return -1;
    }

    return 0;
}