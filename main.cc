#include<iostream>
#include<algorithm>
#include<fstream>
#include <string>
#include<fast_optical_flow_front_end.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void drawFeaturePoints(cv::Mat& image, const std::vector<cv::Point2f>& points, const vector<uint32_t>& ids) {
    for (size_t i = 0; i < points.size(); i++) {
        //绘制特征点位置
        cv::circle(image, points[i], 2, cv::Scalar(0, 255, 0), -1);
        //绘制特征点id
        std::stringstream ss;
        ss << ids[i];
        cv::putText(image, ss.str(), cv::Point(points[i].x + 5, points[i].y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << endl << "Usage:./mono_euroc path_to_sequence_folder path_to_times_file_1" << endl;
        return 1;
    }
    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    int nImages;

    cout << "Loading images ..." << endl;
    LoadImages(string(argv[1]) + "/mav0/cam0/data", string(argv[2]), vstrImageFilenames, vTimestampsCam);
    cout << "LOADED!" << endl;

    nImages = vstrImageFilenames.size();

    cout << endl << "---------" << endl;
    cout.precision(17);

    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++) {
        //Read image from file
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()) {
            cerr << endl << "Fail to load image at : " << vstrImageFilenames[ni] << endl;
            return 1;
        }
        //调用函数来获取特征点
        dm::Config config;
        dm::vio::FastOpticalFlowFrontEnd front(config, -1, -1);
        uint32_t camera_id = 0;
        bool success = front.AddImage(camera_id, im);
        std::vector<cv::Point2f> cur_im_pts;
        cur_im_pts = front.point_coordinates()[camera_id];//该帧图像上的特征点
        std::vector<uint32_t> cur_im_pts_id;
        cur_im_pts_id = front.point_ids()[camera_id];//特征点对应的id

        cv::Mat im_with_keypoints = im.clone();
        cv::cvtColor(im_with_keypoints, im_with_keypoints, cv::COLOR_GRAY2BGR);

        //绘制特征点及对应id
        drawFeaturePoints(im_with_keypoints, cur_im_pts, cur_im_pts_id);

        // cv::Scalar color(0, 255, 0);
        // int radius = 2;

        // for (const auto& point : cur_im_pts) {
        //     cv::circle(im_with_keypoints, point, radius, color, -1);//在新图像上绘制特征点
        // }

        //显示绘制后的图像
        cv::imshow("Image with KeyPoints", im_with_keypoints);
        cv::waitKey(0);//等待按键关闭窗口

        //将绘制后的图像保存到本地
        std::string output_path = "/home/hxh/frontend(copy)/output_pic/" + std::to_string(vTimestampsCam[ni])+".jpg";//保存路径和文件名
       // cout << output_path << endl;
        cv::imwrite(output_path, im_with_keypoints); // 保存图像
    }
    return 0;
}

void LoadImages(const string& strImagePath, const string &strPathTimes,
            vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if(!s.empty()) {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t * 1e-9);
        }
    }
}
