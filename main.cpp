#include <iostream>
#include "Network.h"
#include "Utils.h"
#include <chrono>
#include <fstream>
#include <fstream>
#include "cmake-build-debug/data.pb.h"
#include <opencv2/opencv.hpp>

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

int main(int argl, const char **argc) {


    /* cv::Mat mat(28,28,CV_8UC1,cv::Scalar(0));
     cv::imshow("Test",mat);
     cv::waitKey(0);

 */

    Train::Data data;
    auto *point = data.add_data();
    point->add_input(1);
    point->add_input(1);
    point->add_output(0);
    //
    point = data.add_data();
    point->add_input(1);
    point->add_input(0);
    point->add_output(1);
    //
    point = data.add_data();
    point->add_input(0);
    point->add_input(1);
    point->add_output(1);
    //
    point = data.add_data();
    point->add_input(0);
    point->add_input(0);
    point->add_output(0);
    //




    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> input(2, 0);
    input[0] = 1;
    input[1] = 0;
    Network net(2);
    net.add_layer<Utils::Relu>(6);
    net.add_layer<Utils::Relu>(6);
    net.add_layer<Utils::Sigmoid>(1);

    net.train<Utils::SquareLoss>(data, 0.5f, 4);

    net.forward_pass(&input[0]);




    auto t2 = std::chrono::high_resolution_clock::now();
    auto durr = t2 - t1;

    std::cout << "Time needed: " << durr.count() / 1000000 << std::endl;
    std::cout << "Output" << std::endl;
    std::cout<<std::endl;
    float output = net.get_output_layer().post_activ.get()[0];

    std::cout<<output<<std::endl;


    std::cout<<std::endl;


    return 0;
}
