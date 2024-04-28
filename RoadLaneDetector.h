//원본 출처(설명) https://codingwell.tistory.com/60
//원본 깃허브주소 https://github.com/choijoohee213/OpenCV_Road_Lane_Detection
//자세한 내용 알고싶으면 원본 확인
#pragma once
#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class RoadLaneDetector
{
private:
	double img_size, img_center;
	double left_m, right_m;
	Point left_b, right_b;
	bool left_detect = false, right_detect = false;

public:
	Mat img_frame;
	vector<Point> lane;
	Mat filter_colors(Mat img_frame);
	Mat limit_region(Mat img_edges);
	vector<Vec4i> houghLines(Mat img_mask);
	vector<vector<Vec4i> > separateLine(Mat img_edges, vector<Vec4i> lines);
	vector<Point> regression(vector<vector<Vec4i> > separated_lines, Mat img_input);
	string predictDir();
	Mat drawLine(Mat img_input, vector<Point> lane, string dir);
	Mat img_input;
};
