#include<iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include <windows.h>
#include <wingdi.h>
#include <string.h>
#include <time.h>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>

//壁は未実装

using namespace cv;
using namespace std;

int zero[15][8] = {};
int zero_field[15][8] = {};
//int zero_infield[12][6] = {};
int field[15][8] = {};
int tetsu_field[15][8] = {};//鉄を除いたフィールド
int de[15][8] = {};
int de_col[15][8] = {};
int de_ojama[15][8] = {};
int field1[15][8] = {};//赤
int field2[15][8] = {};//緑
int field3[15][8] = {};//青
int field4[15][8] = {};//黄
int field5[15][8] = {};//紫
int field6[15][8] = {};//お邪魔
int field9[15][8] = {};//固ぷよ
int renketu1[15][8] = {};//赤
int renketu2[15][8] = {};//緑
int renketu3[15][8] = {};//青
int renketu4[15][8] = {};//黄
int renketu5[15][8] = {};//紫
int renketu6[15][8] = {};//お邪魔

						 //0空1赤2緑3青4黄5紫6邪7壁8鉄9固

int seed1[15][8] = {};
int seed2[15][8] = {};
int seed3[15][8] = {};
int seed4[15][8] = {};
int seed5[15][8] = {};

Mat re = imread("r.png");
Mat g = imread("g.png");
Mat b = imread("b.png");
Mat y = imread("y.png");
Mat p = imread("p.png");
Mat o = imread("o.png");
Mat n = imread("n.png");
Mat f = imread("f.png");
Mat k = imread("k.png");

//テンプレートマッチング結果
Mat result_img1;
Mat result_img2;
Mat result_img3;
Mat result_img4;
Mat result_img5;
Mat result_img6;
Mat result_img7;
Mat result_img8;

int jyouken;
int n_jyouken;
string color;
string url;

int chain;
int a = 0;

clock_t start;
clock_t end;

int tekazu;
int clear_flag = 0;
int clear_count = 0;
int tetsu_flag = 0;
int kata_flag = 0;
vector<pair<int, int>> haipuyo;
pair<int, int> place;
vector<int> top(6);

vector<int> puyonum;
vector<int> colorlist;

//画像認識用、今は使ってない
double compare(Mat img1, Mat img2) {
	Mat img1g;
	Mat img2g;
	cvtColor(img1, img1g, CV_BGR2GRAY);
	cvtColor(img2, img2g, CV_BGR2GRAY);
	// 画像のヒストグラムを計算する
	int imageCount = 1; // 入力画像の枚数
	int channelsToUse[] = { 0 }; // 0番目のチャネルを使う
	int dimention = 1; // ヒストグラムの次元数
	int binCount = 256; // ヒストグラムのビンの数
	int binCounts[] = { binCount };
	float range[] = { 0, 256 }; // データの範囲は0～255
	const float* histRange[] = { range };
	Mat histogram1;
	calcHist(&img1g, imageCount, channelsToUse, Mat(), histogram1, dimention, binCounts, histRange);
	Mat histogram2;
	calcHist(&img2g, imageCount, channelsToUse, Mat(), histogram2, dimention, binCounts, histRange);

	double correlation = compareHist(histogram1, histogram2, CV_COMP_CORREL);
	return correlation;
}

//画像認識からのフィールド生成、今は不要
void field_generation(Mat img)
{
	for (int j = 1; j < 14; j++) {
		for (int i = 1; i < 7; i++) {
			cv::Rect roi(16 + 16 * i, 63 + 16 * j, 16, 16);
			Mat puyo = img(roi);
			vector<double> x = {};
			x.push_back(compare(puyo, n));
			x.push_back(compare(puyo, re));
			x.push_back(compare(puyo, g));
			x.push_back(compare(puyo, b));
			x.push_back(compare(puyo, y));
			x.push_back(compare(puyo, p));
			x.push_back(compare(puyo, o));

			vector<double>::iterator iter = max_element(x.begin(), x.end());
			int index = distance(x.begin(), iter);
			field[j][i] = index;
		}
	}
}

//画像認識用、画面のキャプチャからなぞぷよを探す
Mat cap() {
	/* デスクトップのサイズ */
	HWND desktop = GetDesktopWindow();
	RECT rect;
	GetWindowRect(desktop, &rect);
	int width = rect.right;
	int height = rect.bottom;

	/* RGB用と反転用とリサイズ用のIplImageの作成 */
	IplImage *iplimage;
	iplimage = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *flipimage;
	flipimage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	/* DIBの情報を設定する */
	BITMAPINFO bmpInfo;
	bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmpInfo.bmiHeader.biWidth = width;
	bmpInfo.bmiHeader.biHeight = height;
	bmpInfo.bmiHeader.biPlanes = 1;
	bmpInfo.bmiHeader.biBitCount = 24;
	bmpInfo.bmiHeader.biCompression = BI_RGB;

	/* DIBSection作成 */
	LPDWORD lpPixel;
	HDC hDC = GetDC(desktop);
	HBITMAP hBitmap = CreateDIBSection(hDC, &bmpInfo, DIB_RGB_COLORS, (void**)&lpPixel, NULL, 0);
	HDC hMemDC = CreateCompatibleDC(hDC);
	SelectObject(hMemDC, hBitmap);

	/* IplImageヘッダにデータをセット */
	iplimage->imageData = (char *)lpPixel;

	/* ウィンドウ */

	/* デスクトップから取得 */
	BitBlt(hMemDC, 0, 0, width, height, hDC, 0, 0, SRCCOPY);

	/* 上下反転して，リサイズ */
	cvFlip(iplimage, flipimage);
	cv::Mat moto1 = cvarrToMat(flipimage);

	return moto1;

	/* 解放 */
	cvReleaseImageHeader(&iplimage);
	cvReleaseImage(&flipimage);
	ReleaseDC(desktop, hDC);
	DeleteDC(hMemDC);
	DeleteObject(hBitmap);
}

//画像認識からフィールド生成、別バージョン
void field_generation2() {
	//スクショ取得
	Mat capture = cap();

	cv::Mat tmp_img = cv::imread("w.png",1);
	cv::Mat syote = cv::imread("01.png",1);

	Mat sy;
	Mat result_img;
	//フィールド位置
	cv::matchTemplate(capture, tmp_img, result_img, CV_TM_CCOEFF_NORMED);
	//配ぷよ位置
	cv::matchTemplate(capture, syote, sy, CV_TM_CCOEFF_NORMED);

	vector<float> score;
	std::vector<Point> detected_point;
	int flag = 0;
	for (int y = 0; y < result_img.rows; y++) {
		if (flag == 1) { break; }
		for (int x = 0; x < result_img.cols; x++) {
			if (result_img.at<float>(y, x) > 0.95) {
				detected_point.push_back(Point(x, y));
				score.push_back(result_img.at<float>(y, x));
				flag = 1;
				break;
			}
		}
	}

	if (score.size() == 0) {
		cout << "なぞぷよが見つかりません" << endl;
		return;
	}

	//cout << detected_point[0].x << "," << detected_point[0].y << endl;
	Rect roi_rect(detected_point[0].x, detected_point[0].y, tmp_img.cols, tmp_img.rows);//左上の壁

	cv::Point max_pt_h;
	double maxVal_h;
	cv::minMaxLoc(sy, NULL, &maxVal_h, NULL, &max_pt_h);

	/*std::cout << "rectangle: (" << roi_rect.x << ", " <<
	roi_rect.y << ") size (" << roi_rect.width << ", " <<
	roi_rect.height << ")" << std::endl;*/

	//フィールド範囲
	Rect roi1(detected_point[0].x + 16, detected_point[0].y - 16, 16 * 6, 16 * 13);
	Mat s_roi = capture(roi1);

	for (int i = 0;; i++) {
		cv::Rect roi_jiku(max_pt_h.x + 26, max_pt_h.y + 4 + i * 23, 16, 16);
		cv::Rect roi_ko(max_pt_h.x + 26 + 16, max_pt_h.y + 4 + i * 23, 16, 16);
		Mat jikupuyo = capture(roi_jiku);
		Mat kopuyo = capture(roi_ko);
		vector<double> jiku = {};
		vector<double> ko = {};
		jiku.push_back(compare(jikupuyo, n));
		jiku.push_back(compare(jikupuyo, re));
		jiku.push_back(compare(jikupuyo, g));
		jiku.push_back(compare(jikupuyo, b));
		jiku.push_back(compare(jikupuyo, y));
		jiku.push_back(compare(jikupuyo, p));
		jiku.push_back(compare(jikupuyo, o));
		ko.push_back(compare(kopuyo, n));
		ko.push_back(compare(kopuyo, re));
		ko.push_back(compare(kopuyo, g));
		ko.push_back(compare(kopuyo, b));
		ko.push_back(compare(kopuyo, y));
		ko.push_back(compare(kopuyo, p));
		ko.push_back(compare(kopuyo, o));

		vector<double>::iterator iter_j = max_element(jiku.begin(), jiku.end());
		int index_j = distance(jiku.begin(), iter_j);
		vector<double>::iterator iter_k = max_element(ko.begin(), ko.end());
		int index_k = distance(ko.begin(), iter_k);

		if (index_j == 0) { break; }
		pair<int, int> p(index_j, index_k);
		haipuyo.push_back(p);

	}

	cout << "手数=" << haipuyo.size() << endl;
	cout << "ぷよ譜" << endl;
	for (int i = 0; i < haipuyo.size(); i++) {
		cout << haipuyo[i].first << haipuyo[i].second << endl;
	}

	//Mat img_gray;
	//cvtColor(s_roi, img_gray, CV_BGR2GRAY);//フィールド画像をグレイスケール化

	cv::matchTemplate(s_roi, re, result_img1, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, g, result_img2, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, b, result_img3, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, y, result_img4, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, p, result_img5, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, o, result_img6, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, f, result_img7, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(s_roi, k, result_img8, CV_TM_CCOEFF_NORMED);

	float threshold = 0.95f;
	std::vector<Point> detected_point1;
	std::vector<Point> detected_point2;
	std::vector<Point> detected_point3;
	std::vector<Point> detected_point4;
	std::vector<Point> detected_point5;
	std::vector<Point> detected_point6;
	std::vector<Point> detected_point7;//鉄
	std::vector<Point> detected_point8;//固ぷよ

	for (int y = 0; y < result_img1.rows; y++) {
		for (int x = 0; x < result_img1.cols; x++) {
			if (result_img1.at<float>(y, x) > threshold)
				detected_point1.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img2.rows; y++) {
		for (int x = 0; x < result_img2.cols; x++) {
			if (result_img2.at<float>(y, x) > threshold)
				detected_point2.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img3.rows; y++) {
		for (int x = 0; x < result_img3.cols; x++) {
			if (result_img3.at<float>(y, x) > threshold)
				detected_point3.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img4.rows; y++) {
		for (int x = 0; x < result_img4.cols; x++) {
			if (result_img4.at<float>(y, x) > threshold)
				detected_point4.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img5.rows; y++) {
		for (int x = 0; x < result_img5.cols; x++) {
			if (result_img5.at<float>(y, x) > threshold)
				detected_point5.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img6.rows; y++) {
		for (int x = 0; x < result_img6.cols; x++) {
			if (result_img6.at<float>(y, x) > threshold)
				detected_point6.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img7.rows; y++) {
		for (int x = 0; x < result_img7.cols; x++) {
			if (result_img7.at<float>(y, x) > threshold)
				detected_point7.push_back(Point(x, y));
		}
	}

	for (int y = 0; y < result_img8.rows; y++) {
		for (int x = 0; x < result_img8.cols; x++) {
			if (result_img8.at<float>(y, x) > threshold)
				detected_point8.push_back(Point(x, y));
		}
	}

	if (detected_point7.size() > 0) {
		tetsu_flag = 1;
	}

	for (int i = 0; i < detected_point1.size(); i++) {
		field[detected_point1[i].y / 16 + 1][detected_point1[i].x / 16 + 1] = 1;
	}
	for (int i = 0; i < detected_point2.size(); i++) {
		field[detected_point2[i].y / 16 + 1][detected_point2[i].x / 16 + 1] = 2;
	}
	for (int i = 0; i < detected_point3.size(); i++) {
		field[detected_point3[i].y / 16 + 1][detected_point3[i].x / 16 + 1] = 3;
	}
	for (int i = 0; i < detected_point4.size(); i++) {
		field[detected_point4[i].y / 16 + 1][detected_point4[i].x / 16 + 1] = 4;
	}
	for (int i = 0; i < detected_point5.size(); i++) {
		field[detected_point5[i].y / 16 + 1][detected_point5[i].x / 16 + 1] = 5;
	}
	for (int i = 0; i < detected_point6.size(); i++) {
		field[detected_point6[i].y / 16 + 1][detected_point6[i].x / 16 + 1] = 6;
	}
	for (int i = 0; i < detected_point7.size(); i++) {
		field[detected_point7[i].y / 16 + 1][detected_point7[i].x / 16 + 1] = 8;
	}
	for (int i = 0; i < detected_point8.size(); i++) {
		field[detected_point8[i].y / 16 + 1][detected_point8[i].x / 16 + 1] = 9;
	}

	cout << "ぷよ図" << endl;
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			cout << field[c][r];
		}
		cout << "" << endl;
	}

	cv::namedWindow("Image", 1);
	cv::imshow("Image", s_roi);
	waitKey(0);
	cvDestroyWindow("Image");

	return;
}

void screenshot() {
	int aaa;
	cin >> aaa;
	//スクショ取得
	Mat capture = cap();
	Mat cap_gray;
	cvtColor(capture, cap_gray, CV_BGR2GRAY);//フィールド画像をグレイスケール化

	cv::Mat tmp_img = cv::imread("w.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat syote = cv::imread("01.png", CV_LOAD_IMAGE_GRAYSCALE);

	Mat sy;
	Mat result_img;

	cv::namedWindow("Image");
	cv::imshow("Image", capture);
	waitKey(0);
	

	//フィールド位置
	cv::matchTemplate(cap_gray, tmp_img, result_img, CV_TM_CCOEFF_NORMED);
	////配ぷよ位置
	//cv::matchTemplate(capture, syote, sy, CV_TM_CCOEFF_NORMED);

	//vector<float> score;
	//std::vector<Point> detected_point;
	//int flag = 0;
	//for (int y = 0; y < result_img.rows; y++) {
	//	if (flag == 1) { break; }
	//	for (int x = 0; x < result_img.cols; x++) {
	//		if (result_img.at<float>(y, x) > 0.95) {
	//			detected_point.push_back(Point(x, y));
	//			score.push_back(result_img.at<float>(y, x));
	//			flag = 1;
	//			break;
	//		}
	//	}
	//}

	//if (score.size() == 0) {
	//	cout << "なぞぷよが見つかりません" << endl;
	//	return;
	//}

	////cout << detected_point[0].x << "," << detected_point[0].y << endl;
	//Rect roi_rect(detected_point[0].x, detected_point[0].y, tmp_img.cols, tmp_img.rows);//左上の壁

	//cv::Point max_pt_h;
	//double maxVal_h;
	//cv::minMaxLoc(sy, NULL, &maxVal_h, NULL, &max_pt_h);

	///*std::cout << "rectangle: (" << roi_rect.x << ", " <<
	//roi_rect.y << ") size (" << roi_rect.width << ", " <<
	//roi_rect.height << ")" << std::endl;*/

	////フィールド範囲
	//Rect roi1(detected_point[0].x + 16, detected_point[0].y - 16, 16 * 6, 16 * 13);
	//Mat s_roi = capture(roi1);

	//cv::namedWindow("Image", 1);
	//cv::imshow("Image", s_roi);
	//waitKey(0);
}

//urlからフィールドとツモ生成
void url_generation() {
	cout << "URLを入力してください" << endl;
	cin >> url;
	stringstream url(url);
	vector<string> url_bunkatu;
	string s;
	while (getline(url, s, '?')) {
		url_bunkatu.push_back(s);
	}

	stringstream url2(url_bunkatu[1]);
	while (getline(url2, s, '_')) {
		url_bunkatu.push_back(s);
	}

	if (url_bunkatu[2].find('~') == string::npos) {
		for (int i = 0; i < url_bunkatu[2].size(); i++) {
			switch (url_bunkatu[2][url_bunkatu[2].size() - i - 1]) {
			case '0':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case '1':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case '2':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case '3':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case '4':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case '5':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case '6':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case '7':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 0;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case '8':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case '9':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'a':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'b':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'c':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'd':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'e':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'f':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 1;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'g':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'h':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'i':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'j':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'k':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'l':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'm':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'n':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 2;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'o':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'p':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'q':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'r':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 's':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 't':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'u':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'v':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 3;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'w':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'x':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'y':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'z':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'A':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'B':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'C':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'D':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 4;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'E':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'F':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'G':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'H':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'I':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'J':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'K':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'L':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 5;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'M':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'N':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'O':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'P':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'Q':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'R':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case 'S':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case 'T':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 6;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			case 'U':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 0;
				break;
			case 'V':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 1;
				break;
			case 'W':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 2;
				break;
			case 'X':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 3;
				break;
			case 'Y':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 4;
				break;
			case 'Z':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 5;
				break;
			case '.':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 6;
				break;
			case '-':
				field[13 - int(i / 3)][5 - (i % 3) * 2] = 7;
				field[13 - int(i / 3)][6 - (i % 3) * 2] = 7;
				break;
			}
		}
	}

	else {
		if (url_bunkatu[2].find("8") != std::string::npos) {
			tetsu_flag = 1;
		}
		vector<string> url_bunkatu2;
		string s;
		stringstream url3(url_bunkatu[2]);
		while (getline(url3, s, '.')) {
			url_bunkatu2.push_back(s);
		}

		//auto it = url_bunkatu2.begin();
		int i = 0;
		for (auto it = url_bunkatu2.begin(); it < url_bunkatu2.end(); it++) {
			if (i == 0) {
				url_bunkatu2[i] = url_bunkatu2[i].substr(1);
			}
			if (url_bunkatu2[i].size() > 6) {
				it = url_bunkatu2.insert(it, url_bunkatu2[i].substr(0, 6));
				*(it + 1) = url_bunkatu2[i + 1].substr(6);
			}
			i++;
		}
		cout << url_bunkatu2.size() << endl;
		for (auto it = url_bunkatu2.begin(); it<url_bunkatu2.end(); it++) {
			cout << *it << endl;
		}
		for (int i = 0; i < url_bunkatu2.size(); i++) {
			for (int j = 0; j < url_bunkatu2[url_bunkatu2.size() - 1 - i].size(); j++) {
				field[13 - i][j + 1] = url_bunkatu2[url_bunkatu2.size() - 1 - i][j] - '0';
			}
		}
	}

	cout << "ぷよ図" << endl;
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			cout << field[c][r];
		}
		cout << "" << endl;
	}

	for (int i = 0; i < url_bunkatu[3].size() / 2; i++) {
		switch (url_bunkatu[3][i * 2]) {
		case '0': {
			pair<int, int> p(1, 1);
			haipuyo.push_back(p);
			break; }
		case '2': {
			pair<int, int> p(2, 1);
			haipuyo.push_back(p);
			break; }
		case '4': {
			pair<int, int> p(3, 1);
			haipuyo.push_back(p);
			break; }
		case '6': {
			pair<int, int> p(4, 1);
			haipuyo.push_back(p);
			break; }
		case '8': {
			pair<int, int> p(5, 1);
			haipuyo.push_back(p);
			break; }
		case 'c': {
			pair<int, int> p(1, 2);
			haipuyo.push_back(p);
			break; }
		case 'e': {
			pair<int, int> p(2, 2);
			haipuyo.push_back(p);
			break; }
		case 'g': {
			pair<int, int> p(3, 2);
			haipuyo.push_back(p);
			break; }
		case 'i': {
			pair<int, int> p(4, 2);
			haipuyo.push_back(p);
			break; }
		case 'k': {
			pair<int, int> p(5, 2);
			haipuyo.push_back(p);
			break; }
		case 'o': {
			pair<int, int> p(1, 3);
			haipuyo.push_back(p);
			break; }
		case 'q': {
			pair<int, int> p(2, 3);
			haipuyo.push_back(p);
			break; }
		case 's': {
			pair<int, int> p(3, 3);
			haipuyo.push_back(p);
			break; }
		case 'u': {
			pair<int, int> p(4, 3);
			haipuyo.push_back(p);
			break; }
		case 'w': {
			pair<int, int> p(5, 3);
			haipuyo.push_back(p);
			break; }
		case 'A': {
			pair<int, int> p(1, 4);
			haipuyo.push_back(p);
			break; }
		case 'C': {
			pair<int, int> p(2, 4);
			haipuyo.push_back(p);
			break; }
		case 'E': {
			pair<int, int> p(3, 4);
			haipuyo.push_back(p);
			break; }
		case 'G': {
			pair<int, int> p(4, 4);
			haipuyo.push_back(p);
			break; }
		case 'L': {
			pair<int, int> p(5, 4);
			haipuyo.push_back(p);
			break; }
		case 'M': {
			pair<int, int> p(1, 5);
			haipuyo.push_back(p);
			break; }
		case 'O': {
			pair<int, int> p(2, 5);
			haipuyo.push_back(p);
			break; }
		case 'Q': {
			pair<int, int> p(3, 5);
			haipuyo.push_back(p);
			break; }
		case 'S': {
			pair<int, int> p(4, 5);
			haipuyo.push_back(p);
			break; }
		case 'U': {
			pair<int, int> p(5, 5);
			haipuyo.push_back(p);
			break; }
		}
	}

	cout << "手数=" << haipuyo.size() << endl;
	cout << "ぷよ譜" << endl;
	for (int i = 0; i < haipuyo.size(); i++) {
		cout << haipuyo[i].first << haipuyo[i].second << endl;
	}

	switch (url_bunkatu[5][0]) {
	case '2':
		jyouken = 1;
		break;
	case 'a':
		jyouken = 2;
		break;
	case 'b':
		jyouken = 3;
		break;
	case 'c':
		jyouken = 4;
		break;
	case 'd':
		jyouken = 5;
		break;
	case 'u':
		jyouken = 6;
		break;
	case 'v':
		jyouken = 7;
		break;
	case 'w':
		jyouken = 8;
		break;
	case 'x':
		jyouken = 9;
		break;
	case 'E':
		jyouken = 10;
		break;
	case 'F':
		jyouken = 11;
		break;
	case 'G':
		jyouken = 12;
		break;
	case 'H':
		jyouken = 13;
		break;
	case 'I':
		jyouken = 14;
		break;
	case 'J':
		jyouken = 15;
		break;
	case 'Q':
		jyouken = 16;
		break;
	case 'R':
		jyouken = 17;
		break;
	}

	switch (url_bunkatu[5][1]) {
	case '0':
		color = "　";
		break;
	case '1':
		color = "赤";
		break;
	case '2':
		color = "緑";
		break;
	case '3':
		color = "青";
		break;
	case '4':
		color = "黄";
		break;
	case '5':
		color = "紫";
		break;
	case '6':
		color = "おじゃま";
		break;
	case '7':
		color = "色";
		break;
	}

	switch (url_bunkatu[5][2]) {
	case '0':
		n_jyouken = 0;
		break;
	case '1':
		n_jyouken = 1;
		break;
	case '2':
		n_jyouken = 2;
		break;
	case '3':
		n_jyouken = 3;
		break;
	case '4':
		n_jyouken = 4;
		break;
	case '5':
		n_jyouken = 5;
		break;
	case '6':
		n_jyouken = 6;
		break;
	case '7':
		n_jyouken = 7;
		break;
	case '8':
		n_jyouken = 8;
		break;
	case '9':
		n_jyouken = 9;
		break;
	case 'a':
		n_jyouken = 10;
		break;
	case 'b':
		n_jyouken = 11;
		break;
	case 'c':
		n_jyouken = 12;
		break;
	case 'd':
		n_jyouken = 13;
		break;
	case 'e':
		n_jyouken = 14;
		break;
	case 'f':
		n_jyouken = 15;
		break;
	case 'g':
		n_jyouken = 16;
		break;
	case 'h':
		n_jyouken = 17;
		break;
	case 'i':
		n_jyouken = 18;
		break;
	case 'j':
		n_jyouken = 19;
		break;
	case 'k':
		n_jyouken = 20;
		break;
	case 'l':
		n_jyouken = 21;
		break;
	case 'm':
		n_jyouken = 22;
		break;
	case 'n':
		n_jyouken = 23;
		break;
	case 'o':
		n_jyouken = 24;
		break;
	case 'p':
		n_jyouken = 25;
		break;
	case 'q':
		n_jyouken = 26;
		break;
	case 'r':
		n_jyouken = 27;
		break;
	case 's':
		n_jyouken = 28;
		break;
	case 't':
		n_jyouken = 29;
		break;
	case 'u':
		n_jyouken = 30;
		break;
	case 'v':
		n_jyouken = 31;
		break;
	case 'w':
		n_jyouken = 32;
		break;
	case 'x':
		n_jyouken = 33;
		break;
	case 'y':
		n_jyouken = 34;
		break;
	case 'z':
		n_jyouken = 35;
		break;
	case 'A':
		n_jyouken = 36;
		break;
	case 'B':
		n_jyouken = 37;
		break;
	case 'C':
		n_jyouken = 38;
		break;
	case 'D':
		n_jyouken = 39;
		break;
	case 'E':
		n_jyouken = 40;
		break;
	case 'F':
		n_jyouken = 41;
		break;
	case 'G':
		n_jyouken = 42;
		break;
	case 'H':
		n_jyouken = 43;
		break;
	case 'I':
		n_jyouken = 44;
		break;
	case 'J':
		n_jyouken = 45;
		break;
	case 'K':
		n_jyouken = 46;
		break;
	case 'L':
		n_jyouken = 47;
		break;
	case 'M':
		n_jyouken = 48;
		break;
	case 'N':
		n_jyouken = 49;
		break;
	case 'O':
		n_jyouken = 50;
		break;
	case 'P':
		n_jyouken = 51;
		break;
	case 'Q':
		n_jyouken = 52;
		break;
	case 'R':
		n_jyouken = 53;
		break;
	case 'S':
		n_jyouken = 54;
		break;
	case 'T':
		n_jyouken = 55;
		break;
	case 'U':
		n_jyouken = 56;
		break;
	case 'V':
		n_jyouken = 57;
		break;
	case 'W':
		n_jyouken = 58;
		break;
	case 'X':
		n_jyouken = 59;
		break;
	case 'Y':
		n_jyouken = 60;
		break;
	case 'Z':
		n_jyouken = 61;
		break;
	case '.':
		n_jyouken = 62;
		break;
	case '-':
		n_jyouken = 63;
		break;
	}
}

//ブラウザでurl開くやつ
LPWSTR stringtowidechar(std::string temp)
{
	int n;
	n = MultiByteToWideChar(CP_ACP, 0, temp.c_str(), temp.size(), NULL, 0);
	LPWSTR p = new WCHAR[n + 1];
	n = MultiByteToWideChar(CP_ACP, 0, temp.c_str(), temp.size(), p, n);
	*(p + n) = '\0';
	return p;
}

//問題からurlを出力
string url_export(int array[][8],vector<pair<int,int>> haipuyo) {
	string url;
	string url_field;
	url = "http://ips.karou.jp/simu/pn.html?";
	if (tetsu_flag + kata_flag == 0) {
		for (int i = 0; i < 39; i++) {
			pair<int, int> u(array[13 - int(i / 3)][5 - (i % 3) * 2], array[13 - int(i / 3)][6 - (i % 3) * 2]);
			if (u.first == 0 && u.second == 0) {
				url_field = "0" + url_field;
			}
			else if (u.first == 0 && u.second == 1) {
				url_field = "1" + url_field;
			}
			else if (u.first == 0 && u.second == 2) {
				url_field = "2" + url_field;
			}
			else if (u.first == 0 && u.second == 3) {
				url_field = "3" + url_field;
			}
			else if (u.first == 0 && u.second == 4) {
				url_field = "4" + url_field;
			}
			else if (u.first == 0 && u.second == 5) {
				url_field = "5" + url_field;
			}
			else if (u.first == 0 && u.second == 6) {
				url_field = "6" + url_field;
			}
			else if (u.first == 0 && u.second == 7) {
				url_field = "7" + url_field;
			}
			else if (u.first == 1 && u.second == 0) {
				url_field = "8" + url_field;
			}
			else if (u.first == 1 && u.second == 1) {
				url_field = "9" + url_field;
			}
			else if (u.first == 1 && u.second == 2) {
				url_field = "a" + url_field;
			}
			else if (u.first == 1 && u.second == 3) {
				url_field = "b" + url_field;
			}
			else if (u.first == 1 && u.second == 4) {
				url_field = "c" + url_field;
			}
			else if (u.first == 1 && u.second == 5) {
				url_field = "d" + url_field;
			}
			else if (u.first == 1 && u.second == 6) {
				url_field = "e" + url_field;
			}
			else if (u.first == 1 && u.second == 7) {
				url_field = "f" + url_field;
			}
			else if (u.first == 2 && u.second == 0) {
				url_field = "g" + url_field;
			}
			else if (u.first == 2 && u.second == 1) {
				url_field = "h" + url_field;
			}
			else if (u.first == 2 && u.second == 2) {
				url_field = "i" + url_field;
			}
			else if (u.first == 2 && u.second == 3) {
				url_field = "j" + url_field;
			}
			else if (u.first == 2 && u.second == 4) {
				url_field = "k" + url_field;
			}
			else if (u.first == 2 && u.second == 5) {
				url_field = "l" + url_field;
			}
			else if (u.first == 2 && u.second == 6) {
				url_field = "m" + url_field;
			}
			else if (u.first == 2 && u.second == 7) {
				url_field = "n" + url_field;
			}
			else if (u.first == 3 && u.second == 0) {
				url_field = "o" + url_field;
			}
			else if (u.first == 3 && u.second == 1) {
				url_field = "p" + url_field;
			}
			else if (u.first == 3 && u.second == 2) {
				url_field = "q" + url_field;
			}
			else if (u.first == 3 && u.second == 3) {
				url_field = "r" + url_field;
			}
			else if (u.first == 3 && u.second == 4) {
				url_field = "s" + url_field;
			}
			else if (u.first == 3 && u.second == 5) {
				url_field = "t" + url_field;
			}
			else if (u.first == 3 && u.second == 6) {
				url_field = "u" + url_field;
			}
			else if (u.first == 3 && u.second == 7) {
				url_field = "v" + url_field;
			}
			else if (u.first == 4 && u.second == 0) {
				url_field = "w" + url_field;
			}
			else if (u.first == 4 && u.second == 1) {
				url_field = "x" + url_field;
			}
			else if (u.first == 4 && u.second == 2) {
				url_field = "y" + url_field;
			}
			else if (u.first == 4 && u.second == 3) {
				url_field = "z" + url_field;
			}
			else if (u.first == 4 && u.second == 4) {
				url_field = "A" + url_field;
			}
			else if (u.first == 4 && u.second == 5) {
				url_field = "B" + url_field;
			}
			else if (u.first == 4 && u.second == 6) {
				url_field = "C" + url_field;
			}
			else if (u.first == 4 && u.second == 7) {
				url_field = "D" + url_field;
			}
			else if (u.first == 5 && u.second == 0) {
				url_field = "E" + url_field;
			}
			else if (u.first == 5 && u.second == 1) {
				url_field = "F" + url_field;
			}
			else if (u.first == 5 && u.second == 2) {
				url_field = "G" + url_field;
			}
			else if (u.first == 5 && u.second == 3) {
				url_field = "H" + url_field;
			}
			else if (u.first == 5 && u.second == 4) {
				url_field = "I" + url_field;
			}
			else if (u.first == 5 && u.second == 5) {
				url_field = "J" + url_field;
			}
			else if (u.first == 5 && u.second == 6) {
				url_field = "K" + url_field;
			}
			else if (u.first == 5 && u.second == 7) {
				url_field = "L" + url_field;
			}
			else if (u.first == 6 && u.second == 0) {
				url_field = "M" + url_field;
			}
			else if (u.first == 6 && u.second == 1) {
				url_field = "N" + url_field;
			}
			else if (u.first == 6 && u.second == 2) {
				url_field = "O" + url_field;
			}
			else if (u.first == 6 && u.second == 3) {
				url_field = "P" + url_field;
			}
			else if (u.first == 6 && u.second == 4) {
				url_field = "Q" + url_field;
			}
			else if (u.first == 6 && u.second == 5) {
				url_field = "R" + url_field;
			}
			else if (u.first == 6 && u.second == 6) {
				url_field = "S" + url_field;
			}
			else if (u.first == 6 && u.second == 7) {
				url_field = "T" + url_field;
			}
			else if (u.first == 7 && u.second == 0) {
				url_field = "U" + url_field;
			}
			else if (u.first == 7 && u.second == 1) {
				url_field = "V" + url_field;
			}
			else if (u.first == 7 && u.second == 2) {
				url_field = "W" + url_field;
			}
			else if (u.first == 7 && u.second == 3) {
				url_field = "X" + url_field;
			}
			else if (u.first == 7 && u.second == 4) {
				url_field = "Y" + url_field;
			}
			else if (u.first == 7 && u.second == 5) {
				url_field = "Z" + url_field;
			}
			else if (u.first == 7 && u.second == 6) {
				url_field = "." + url_field;
			}
			else if (u.first == 7 && u.second == 7) {
				url_field = "-" + url_field;
			}
		}
		while (url_field.substr(0, 3) == "000") {
			url_field = url_field.substr(3);
		}
	}
	else {
		for (int i = 1; i < 14; i++) {
			if (i != 1) {
				url_field = url_field + ".";
			}
			for (int j = 1; j < 7; j++) {
				if (array[i][j] == 0) {
					url_field = url_field + "0";
				}
				else if (array[i][j] == 1) {
					url_field = url_field + "1";
				}
				else if (array[i][j] == 2) {
					url_field = url_field + "2";
				}
				else if (array[i][j] == 3) {
					url_field = url_field + "3";
				}
				else if (array[i][j] == 4) {
					url_field = url_field + "4";
				}
				else if (array[i][j] == 5) {
					url_field = url_field + "5";
				}
				else if (array[i][j] == 6) {
					url_field = url_field + "6";
				}
				else if (array[i][j] == 7) {
					url_field = url_field + "7";
				}
				else if (array[i][j] == 8) {
					url_field = url_field + "8";
				}
				else if (array[i][j] == 9) {
					url_field = url_field + "9";
				}
			}
		}
		while (url_field.substr(0, 7) == "000000.") {
			url_field = url_field.substr(7);
		}
		string target = "0. ";     // 検索文字列
		string replacement = "."; // 置換文字列
		if (!target.empty()) {
			std::string::size_type pos = 0;
			while ((pos = url_field.find(target, pos)) != std::string::npos) {
				url_field.replace(pos, target.length(), replacement);
				pos += replacement.length();
			}
		}
		url_field = "~" + url_field;

	}

	string url_tumo;
	for (int i = 0; i < haipuyo.size(); i++) {
		if (haipuyo[i].first == 1 && haipuyo[i].second == 1) {
			url_tumo = url_tumo + "01";
		}
		if (haipuyo[i].first == 1 && haipuyo[i].second == 2) {
			url_tumo = url_tumo + "c1";
		}
		if (haipuyo[i].first == 1 && haipuyo[i].second == 3) {
			url_tumo = url_tumo + "o1";
		}
		if (haipuyo[i].first == 1 && haipuyo[i].second == 4) {
			url_tumo = url_tumo + "A1";
		}
		if (haipuyo[i].first == 1 && haipuyo[i].second == 5) {
			url_tumo = url_tumo + "M1";
		}
		if (haipuyo[i].first == 2 && haipuyo[i].second == 1) {
			url_tumo = url_tumo + "21";
		}
		if (haipuyo[i].first == 2 && haipuyo[i].second == 2) {
			url_tumo = url_tumo + "e1";
		}
		if (haipuyo[i].first == 2 && haipuyo[i].second == 3) {
			url_tumo = url_tumo + "q1";
		}
		if (haipuyo[i].first == 2 && haipuyo[i].second == 4) {
			url_tumo = url_tumo + "C1";
		}
		if (haipuyo[i].first == 2 && haipuyo[i].second == 5) {
			url_tumo = url_tumo + "O1";
		}
		if (haipuyo[i].first == 3 && haipuyo[i].second == 1) {
			url_tumo = url_tumo + "41";
		}
		if (haipuyo[i].first == 3 && haipuyo[i].second == 2) {
			url_tumo = url_tumo + "g1";
		}
		if (haipuyo[i].first == 3 && haipuyo[i].second == 3) {
			url_tumo = url_tumo + "s1";
		}
		if (haipuyo[i].first == 3 && haipuyo[i].second == 4) {
			url_tumo = url_tumo + "E1";
		}
		if (haipuyo[i].first == 3 && haipuyo[i].second == 5) {
			url_tumo = url_tumo + "Q1";
		}
		if (haipuyo[i].first == 4 && haipuyo[i].second == 1) {
			url_tumo = url_tumo + "61";
		}
		if (haipuyo[i].first == 4 && haipuyo[i].second == 2) {
			url_tumo = url_tumo + "i1";
		}
		if (haipuyo[i].first == 4 && haipuyo[i].second == 3) {
			url_tumo = url_tumo + "u1";
		}
		if (haipuyo[i].first == 4 && haipuyo[i].second == 4) {
			url_tumo = url_tumo + "G1";
		}
		if (haipuyo[i].first == 4 && haipuyo[i].second == 5) {
			url_tumo = url_tumo + "S1";
		}
		if (haipuyo[i].first == 5 && haipuyo[i].second == 1) {
			url_tumo = url_tumo + "81";
		}
		if (haipuyo[i].first == 5 && haipuyo[i].second == 2) {
			url_tumo = url_tumo + "k1";
		}
		if (haipuyo[i].first == 5 && haipuyo[i].second == 3) {
			url_tumo = url_tumo + "w1";
		}
		if (haipuyo[i].first == 5 && haipuyo[i].second == 4) {
			url_tumo = url_tumo + "L1";
		}
		if (haipuyo[i].first == 5 && haipuyo[i].second == 5) {
			url_tumo = url_tumo + "U1";
		}
	}

	string url_jyouken;
	string url_color;
	switch (jyouken) {
	case 1:
		url_jyouken = "2";
		break;
	case 2:
		url_jyouken = "a";
		break;
	case 3:
		url_jyouken = "b";
		break;
	case 4:
		url_jyouken = "c";
		break;
	case 5:
		url_jyouken = "d";
		break;
	case 6:
		url_jyouken = "u";
		url_color = "0";
		break;
	case 7:
		url_jyouken = "v";
		break;
	case 8:
		url_jyouken = "w";
		break;
	case 9:
		url_jyouken = "x";
		break;
	case 10:
		url_jyouken = "E";
		break;
	case 11:
		url_jyouken = "F";
		break;
	case 12:
		url_jyouken = "G";
		break;
	case 13:
		url_jyouken = "H";
		break;
	case 14:
		url_jyouken = "I";
		break;
	case 15:
		url_jyouken = "J";
		break;
	case 16:
		url_jyouken = "Q";
		break;
	case 17:
		url_jyouken = "R";
		break;
	}

	if (color == "　") {
		url_color = "0";
	}
	else if(color =="赤"){
		url_color = "1";
	}
	else if (color == "緑") {
		url_color = "2";
	}
	else if (color == "青") {
		url_color = "3";
	}
	else if (color == "黄") {
		url_color = "4";
	}
	else if (color == "紫") {
		url_color = "5";
	}
	else if (color == "おじゃま") {
		url_color = "6";
	}
	else if (color == "色") {
		url_color = "7";
	}

	string url_number;
	switch (n_jyouken) {
	case 0:
		url_number = "0";
		break;
	case 1:
		url_number = "1";
		break;
	case 2:
		url_number = "2";
		break;
	case 3:
		url_number = "3";
		break;
	case 4:
		url_number = "4";
		break;
	case 5:
		url_number = "5";
		break;
	case 6:
		url_number = "6";
		break;
	case 7:
		url_number = "7";
		break;
	case 8:
		url_number = "8";
		break;
	case 9:
		url_number = "9";
		break;
	case 10:
		url_number = "a";
		break;
	case 11:
		url_number = "b";
		break;
	case 12:
		url_number = "c";
		break;
	case 13:
		url_number = "d";
		break;
	case 14:
		url_number = "e";
		break;
	case 15:
		url_number = "f";
		break;
	case 16:
		url_number = "g";
		break;
	case 17:
		url_number = "h";
		break;
	case 18:
		url_number = "i";
		break;
	case 19:
		url_number = "j";
		break;
	case 20:
		url_number = "k";
		break;
	case 21:
		url_number = "l";
		break;
	case 22:
		url_number = "m";
		break;
	case 23:
		url_number = "n";
		break;
	case 24:
		url_number = "o";
		break;
	case 25:
		url_number = "p";
		break;
	case 26:
		url_number = "q";
		break;
	case 27:
		url_number = "r";
		break;
	case 28:
		url_number = "s";
		break;
	case 29:
		url_number = "t";
		break;
	case 30:
		url_number = "u";
		break;
	case 31:
		url_number = "v";
		break;
	case 32:
		url_number = "w";
		break;
	case 33:
		url_number = "x";
		break;
	case 34:
		url_number = "y";
		break;
	case 35:
		url_number = "z";
		break;
	case 36:
		url_number = "A";
		break;
	case 37:
		url_number = "B";
		break;
	case 38:
		url_number = "C";
		break;
	case 39:
		url_number = "D";
		break;
	case 40:
		url_number = "E";
		break;
	case 41:
		url_number = "F";
		break;
	case 42:
		url_number = "G";
		break;
	case 43:
		url_number = "H";
		break;
	case 44:
		url_number = "I";
		break;
	case 45:
		url_number = "J";
		break;
	case 46:
		url_number = "K";
		break;
	case 47:
		url_number = "L";
		break;
	case 48:
		url_number = "M";
		break;
	case 49:
		url_number = "N";
		break;
	case 50:
		url_number = "O";
		break;
	case 51:
		url_number = "P";
		break;
	case 52:
		url_number = "Q";
		break;
	case 53:
		url_number = "R";
		break;
	case 54:
		url_number = "S";
		break;
	case 55:
		url_number = "T";
		break;
	case 56:
		url_number = "U";
		break;
	case 57:
		url_number = "V";
		break;
	case 58:
		url_number = "W";
		break;
	case 59:
		url_number = "X";
		break;
	case 60:
		url_number = "Y";
		break;
	case 61:
		url_number = "Z";
		break;
	case 62:
		url_number = ".";
		break;
	case 63:
		url_number = "-";
		break;
	}

	url_jyouken = url_jyouken + url_color + url_number;

	url = url + url_field + "_" + url_tumo + "__" + url_jyouken;

	LPWSTR wurl = stringtowidechar(url);
	ShellExecute(NULL, L"open", wurl, NULL, NULL, SW_SHOWNORMAL);

	return url;

}

//画像認識で配ぷよ認識
void haipuyo_generation(Mat img) {
	for (int i = 0;; i++) {
		cv::Rect roi(213, 151 + 23 * i, 16, 16);
		cv::Rect roi2(229, 151 + 23 * i, 16, 16);
		Mat jikupuyo = img(roi);
		Mat kopuyo = img(roi2);
		vector<double> jiku = {};
		vector<double> ko = {};
		jiku.push_back(compare(jikupuyo, n));
		jiku.push_back(compare(jikupuyo, re));
		jiku.push_back(compare(jikupuyo, g));
		jiku.push_back(compare(jikupuyo, b));
		jiku.push_back(compare(jikupuyo, y));
		jiku.push_back(compare(jikupuyo, p));
		jiku.push_back(compare(jikupuyo, o));
		ko.push_back(compare(kopuyo, n));
		ko.push_back(compare(kopuyo, re));
		ko.push_back(compare(kopuyo, g));
		ko.push_back(compare(kopuyo, b));
		ko.push_back(compare(kopuyo, y));
		ko.push_back(compare(kopuyo, p));
		ko.push_back(compare(kopuyo, o));

		vector<double>::iterator iter_j = max_element(jiku.begin(), jiku.end());
		int index_j = distance(jiku.begin(), iter_j);
		vector<double>::iterator iter_k = max_element(ko.begin(), ko.end());
		int index_k = distance(ko.begin(), iter_k);

		if (index_j == 0) { break; }

		pair<int, int> p(index_j, index_k);
		haipuyo.push_back(p);
	}
}

//ぷよの連結数管理
void renketu(int array1[][8], int array2[][8], int array3[][8], int array4[][8], int array5[][8]) {
	for (int i = 2; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (i == 2) {//12段目の場合、13段目は連結にカウント対象外
				if (array1[i][j] == 1) {
					renketu1[i][j] = array1[i + 1][j] + array1[i][j - 1] + array1[i][j + 1];
				}
				else if (array2[i][j] == 1) {
					renketu2[i][j] = array2[i + 1][j] + array2[i][j - 1] + array2[i][j + 1];
				}
				else if (array3[i][j] == 1) {
					renketu3[i][j] = array3[i + 1][j] + array3[i][j - 1] + array3[i][j + 1];
				}
				else if (array4[i][j] == 1) {
					renketu4[i][j] = array4[i + 1][j] + array4[i][j - 1] + array4[i][j + 1];
				}
				else if (array5[i][j] == 1) {
					renketu5[i][j] = array5[i + 1][j] + array5[i][j - 1] + array5[i][j + 1];
				}
			}
			else {
				if (array1[i][j] == 1) {
					renketu1[i][j] = array1[i - 1][j] + array1[i + 1][j] + array1[i][j - 1] + array1[i][j + 1];
				}
				else if (array2[i][j] == 1) {
					renketu2[i][j] = array2[i - 1][j] + array2[i + 1][j] + array2[i][j - 1] + array2[i][j + 1];
				}
				else if (array3[i][j] == 1) {
					renketu3[i][j] = array3[i - 1][j] + array3[i + 1][j] + array3[i][j - 1] + array3[i][j + 1];
				}
				else if (array4[i][j] == 1) {
					renketu4[i][j] = array4[i - 1][j] + array4[i + 1][j] + array4[i][j - 1] + array4[i][j + 1];
				}
				else if (array5[i][j] == 1) {
					renketu5[i][j] = array5[i - 1][j] + array5[i + 1][j] + array5[i][j - 1] + array5[i][j + 1];
				}
			}
		}
	}
}

//消えるぷよ管理
void seed_generation(int array1[][8], int array2[][8], int array3[][8], int array4[][8], int array5[][8]) {
	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (array1[i][j] > 2) {
				seed1[i][j] = 1;
			}
			else if (array1[i][j] == 2) {
				if (array1[i - 1][j] >1) {//上
					seed1[i][j] = 1;
					seed1[i - 1][j] = 1;
				}
				if (array1[i + 1][j] >1) {//下
					seed1[i][j] = 1;
					seed1[i + 1][j] = 1;
				}
				if (array1[i][j - 1] >1) {//左
					seed1[i][j] = 1;
					seed1[i][j - 1] = 1;
				}
				if (array1[i][j + 1] >1) {//右
					seed1[i][j] = 1;
					seed1[i][j + 1] = 1;
				}
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (seed1[i][j] == 1) {
				if (array1[i - 1][j] == 1)
					seed1[i - 1][j] = 1;
				if (array1[i + 1][j] == 1)
					seed1[i + 1][j] = 1;
				if (array1[i][j - 1] == 1)
					seed1[i][j - 1] = 1;
				if (array1[i][j + 1] == 1)
					seed1[i][j + 1] = 1;
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (array2[i][j] > 2) {
				seed2[i][j] = 1;
			}
			else if (array2[i][j] == 2) {
				if (array2[i - 1][j] >1) {//上
					seed2[i][j] = 1;
					seed2[i - 1][j] = 1;
				}
				if (array2[i + 1][j] > 1) {//下
					seed2[i][j] = 1;
					seed2[i + 1][j] = 1;
				}
				if (array2[i][j - 1] > 1) {//左
					seed2[i][j] = 1;
					seed2[i][j - 1] = 1;
				}
				if (array2[i][j + 1] > 1) {//右
					seed2[i][j] = 1;
					seed2[i][j + 1] = 1;
				}
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (seed2[i][j] == 1) {
				if (array2[i - 1][j] == 1)
					seed2[i - 1][j] = 1;
				if (array2[i + 1][j] == 1)
					seed2[i + 1][j] = 1;
				if (array2[i][j - 1] == 1)
					seed2[i][j - 1] = 1;
				if (array2[i][j + 1] == 1)
					seed2[i][j + 1] = 1;
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (array3[i][j] > 2) {
				seed3[i][j] = 1;
			}
			else if (array3[i][j] == 2) {
				if (array3[i - 1][j] > 1) {//上
					seed3[i][j] = 1;
					seed3[i - 1][j] = 1;
				}
				if (array3[i + 1][j] > 1) {//下
					seed3[i][j] = 1;
					seed3[i + 1][j] = 1;
				}
				if (array3[i][j - 1] > 1) {//左
					seed3[i][j] = 1;
					seed3[i][j - 1] = 1;
				}
				if (array3[i][j + 1] > 1) {//右
					seed3[i][j] = 1;
					seed3[i][j + 1] = 1;
				}
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (seed3[i][j] == 1) {
				if (array3[i - 1][j] == 1)
					seed3[i - 1][j] = 1;
				if (array3[i + 1][j] == 1)
					seed3[i + 1][j] = 1;
				if (array3[i][j - 1] == 1)
					seed3[i][j - 1] = 1;
				if (array3[i][j + 1] == 1)
					seed3[i][j + 1] = 1;
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (array4[i][j] > 2) {
				seed4[i][j] = 1;
			}
			else if (array4[i][j] == 2) {
				if (array4[i - 1][j] > 1) {//上
					seed4[i][j] = 1;
					seed4[i - 1][j] = 1;
				}
				if (array4[i + 1][j] > 1) {//下
					seed4[i][j] = 1;
					seed4[i + 1][j] = 1;
				}
				if (array4[i][j - 1] > 1) {//左
					seed4[i][j] = 1;
					seed4[i][j - 1] = 1;
				}
				if (array4[i][j + 1] > 1) {//右
					seed4[i][j] = 1;
					seed4[i][j + 1] = 1;
				}
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (seed4[i][j] == 1) {
				if (array4[i - 1][j] == 1)
					seed4[i - 1][j] = 1;
				if (array4[i + 1][j] == 1)
					seed4[i + 1][j] = 1;
				if (array4[i][j - 1] == 1)
					seed4[i][j - 1] = 1;
				if (array4[i][j + 1] == 1)
					seed4[i][j + 1] = 1;
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (array5[i][j] > 2) {
				seed5[i][j] = 1;
			}
			else if (array5[i][j] == 2) {
				if (array5[i - 1][j] > 1) {//上
					seed5[i][j] = 1;
					seed5[i - 1][j] = 1;
				}
				if (array5[i + 1][j] > 1) {//下
					seed5[i][j] = 1;
					seed5[i + 1][j] = 1;
				}
				if (array5[i][j - 1] > 1) {//左
					seed5[i][j] = 1;
					seed5[i][j - 1] = 1;
				}
				if (array5[i][j + 1] > 1) {//右
					seed5[i][j] = 1;
					seed5[i][j + 1] = 1;
				}
			}
		}
	}

	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (seed5[i][j] == 1) {
				if (array5[i - 1][j] == 1)
					seed5[i - 1][j] = 1;
				if (array5[i + 1][j] == 1)
					seed5[i + 1][j] = 1;
				if (array5[i][j - 1] == 1)
					seed5[i][j - 1] = 1;
				if (array5[i][j + 1] == 1)
					seed5[i][j + 1] = 1;
			}
		}
	}
}

//落下処理、処理変えてみたらむしろ遅くなった
void fall2(int array[][8]) {
	for (int i = 1; i < 7; i++) {
		string row;
		for (int j = 13; j > 0; j--) {
			row += to_string(array[j][i]);
		}
		for (size_t c = row.find_first_of("0"); c != string::npos; c = c = row.find_first_of("0")) {
			row.erase(c, 1);
		}
		for (int j = 0; j < 13; j++) {
			if (j < row.size()) {
				array[13 - j][i] = row[j] - '0';
			}
			else {
				array[13 - j][i] = 0;
			}
		}
	}
}

//落下処理
void fall(int array[][8]) {
	for (int i = 1; i < 7; i++) {
		while (1) {
			int j = 13;
			while (array[j][i] != 0 && j > 0) {//下から見て空マスを探す
				j = j - 1;
			}
			if (j == 0) {//13段全部埋まっているとき
				break;
			}
			else {
				j = j - 1;
				while (array[j][i] == 0 && j > 0) {//空マスより上にあるぷよを探す
					j = j - 1;
				}
				if (j == 0) { break; }
				else {
					for (j = j; j > 0; j--) {//落下
						array[j + 1][i] = field[j][i];
					}
					array[1][i] = 0;
				}
			}
		}
	}
}

void first_field(int field[][8]) {
	//各色の配置
	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (field[i][j] == 1)
				field1[i][j] = 1;
			else if (field[i][j] == 2)
				field2[i][j] = 1;
			else if (field[i][j] == 3)
				field3[i][j] = 1;
			else if (field[i][j] == 4)
				field4[i][j] = 1;
			else if (field[i][j] == 5)
				field5[i][j] = 1;
			else if (field[i][j] == 6)
				field6[i][j] = 1;
			else if (field[i][j] == 9)
				field6[i][j] = 2;
		}
	}

	//連結数判定
	renketu(field1, field2, field3, field4, field5);

	//seed生成1
	seed_generation(renketu1, renketu2, renketu3, renketu4, renketu5);

	//消えるマス（おじゃま含まず）
	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			de_col[i][j] = seed1[i][j] + seed2[i][j] + seed3[i][j] + seed4[i][j] + seed5[i][j];
		}
	}

	//おじゃま消去判定
	for (int i = 2; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (de_col[i][j] == 1) {
				if (field[i + 1][j] == 6) { de_ojama[i + 1][j] = 1; }//下
				if (field[i - 1][j] == 6) {//上
					if (i != 2) {//13段目は除外
						de_ojama[i - 1][j] = 1;
					}
				}
				if (field[i][j + 1] == 6) { de_ojama[i][j + 1] = 1; }//右
				if (field[i][j - 1] == 6) { de_ojama[i][j - 1] = 1; }//左
			}
		}
	}

	//固ぷよ
	for (int i = 2; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (field[i][j] == 9) {
				if (de_col[i - 1][j] + de_col[i + 1][j] + de_col[i][j - 1] + de_col[i][j + 1] > 1) {
					de_ojama[i][j] = 1;
				}
				else if (de_col[i - 1][j] + de_col[i + 1][j] + de_col[i][j - 1] + de_col[i][j + 1] == 1) {
					field[i][j] = 6;
				}
			}
		}
	}

	//消えるマス（おじゃま含む）
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			de[c][r] = de_col[c][r] + de_ojama[c][r];
		}
	}
}

//各列の最上段ぷよ
void top_generation(int array[][8]) {
	for (int i = 1; i < 7; i++) {
		int j = 1;
		while (array[j][i] == 0) {
			j++;
		}
		top[i - 1] = j - 1;
	}
}

//ぷよ消すだけ
void puyo_delete(int de[][8], int field[][8]) {
	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (de[i][j] == 1) {
				field[i][j] = 0;
			}
		}
	}
}

//初期化
void initialization() {
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			//de[c][r] = 0;
			de_col[c][r] = 0;
			de_ojama[c][r] = 0;
			field1[c][r] = 0;
			field2[c][r] = 0;
			field3[c][r] = 0;
			field4[c][r] = 0;
			field5[c][r] = 0;
			field6[c][r] = 0;
			renketu1[c][r] = 0;
			renketu2[c][r] = 0;
			renketu3[c][r] = 0;
			renketu4[c][r] = 0;
			renketu5[c][r] = 0;
			seed1[c][r] = 0;
			seed2[c][r] = 0;
			seed3[c][r] = 0;
			seed4[c][r] = 0;
			seed5[c][r] = 0;
		}
	}
}

int hakkanum;
int clear_count_z;
vector<vector<pair<int, int>>> answerlist;
vector<int> hakkanumlist;
//答え表示
void dispans(vector<pair<int, int>> &answer) {
	answerlist.push_back(answer);
	hakkanumlist.push_back(hakkanum);

	/*cout << "clear!!" << endl;
	for (int l = 0; l < answer.size(); l++) {
		cout << answer[l].first;
		if (answer[l].second == 0) {
			cout << "↑" << endl;
		}
		else if (answer[l].second == 1) {
			cout << "↓" << endl;
		}
		else if (answer[l].second == 2) {
			cout << "←" << endl;
		}
		else if (answer[l].second == 3) {
			cout << "→" << endl;
		}
	}*/
	clear_count += 1;
	if (hakkanum == 0) {
		clear_count_z += 1;
	}
}

//c色同時消しクリアチェック
int multicolor;
void multi_clear(int seed1[][8], int seed2[][8], int seed3[][8], int seed4[][8], int seed5[][8], vector<pair<int, int>> &answer) {
	if (9 < jyouken && jyouken < 12) {
		multicolor = abs(memcmp(seed1, zero, sizeof(zero))) + abs(memcmp(seed2, zero, sizeof(zero))) + abs(memcmp(seed3, zero, sizeof(zero))) + abs(memcmp(seed4, zero, sizeof(zero))) + abs(memcmp(seed5, zero, sizeof(zero)));
		switch (jyouken) {
		case 10:
			if (multicolor == n_jyouken) {
				dispans(answer);
			}
			break;
		case 11:
			if (multicolor >= n_jyouken) {
				dispans(answer);
			}
			break;
		}
	}
}

//同時消しクリアチェック
int doujikeshi;
void doujikeshi_clear(int de[][8], int seed1[][8], int seed2[][8], int seed3[][8], int seed4[][8], int seed5[][8], vector<pair<int, int>> &answer) {
	if (11 < jyouken && jyouken < 14) {
		doujikeshi = 0;
		if (color == "　") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (de[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "赤") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (seed1[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "緑") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (seed2[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "青") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (seed3[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "黄") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (seed4[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "紫") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (seed5[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "おじゃま") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (de_ojama[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}
		else if (color == "色") {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (de_col[c][r] == 1) {
						doujikeshi += 1;
					}
				}
			}
		}

		switch (jyouken) {
		case 12:
			if (doujikeshi == n_jyouken) {
				dispans(answer);
			}
			break;
		case 13:
			if (doujikeshi >= n_jyouken) {
				dispans(answer);
			}
			break;
		}
	}
}

//全消しクリア条件
void allclear_check(vector<pair<int, int>> &answer) {
	if (color == "　") {//ぷよ全て消すべし　//壁,鉄ぷよ導入時は要改変
		if (tetsu_flag == 1) {
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					if (field[c][r] == 8) {
						tetsu_field[c][r] = 0;
					}
					else {
						tetsu_field[c][r] = field[r][c];
					}
				}
			}
			if (memcmp(tetsu_field, zero, sizeof(zero)) == 0) {
				clear_flag = 1;
			}
		}
		else {
			if (memcmp(field, zero_field, sizeof(zero_field)) == 0) {
				clear_flag = 1;
			}
		}
	}
	else if (color == "赤") {//赤ぷよ全て消すべし
		if (memcmp(field1, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "緑") {//緑ぷよ全て消すべし
		if (memcmp(field2, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "青") {//青ぷよ全て消すべし
		if (memcmp(field3, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "黄") {//黄ぷよ全て消すべし
		if (memcmp(field4, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "紫") {//紫ぷよ全て消すべし
		if (memcmp(field5, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "おじゃま") {//おじゃまぷよ全て消すべし
		if (memcmp(field6, zero, sizeof(zero)) == 0) {
			clear_flag = 1;
		}
	}
	else if (color == "色") {//色ぷよ全て消すべし
		if (abs(memcmp(field1, zero, sizeof(zero))) + abs(memcmp(field2, zero, sizeof(zero))) + abs(memcmp(field3, zero, sizeof(zero))) + abs(memcmp(field4, zero, sizeof(zero))) + abs(memcmp(field5, zero, sizeof(zero))) == 0) {
			clear_flag = 1;
		}
	}

	if (clear_flag == 1) {
		dispans(answer);
		clear_flag = 0;
	}
}

int check_flag = 0;
int pchain;
int num[5];
int t;

//枝刈り
void cut_check(int k, int tekazu) {
	check_flag = 0;
	for (t = 0; t < 5; t++) {
		num[t] = 0;
	}
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			num[0] += field1[c][r];
			num[1] += field2[c][r];
			num[2] += field3[c][r];
			num[3] += field4[c][r];
			num[4] += field5[c][r];
		}
	}

	for (t = k + 1; t < tekazu; t++) {
		num[haipuyo[t].first - 1] += 1;
		num[haipuyo[t].second - 1] += 1;
	}

	if (5 < jyouken && jyouken < 10) {//連鎖条件の枝刈り
		pchain = (num[0] / 4) + (num[1] / 4) + (num[2] / 4) + (num[3] / 4) + (num[4] / 4);
		if (pchain < n_jyouken) {
			check_flag = 1;
		}
	}
	if (jyouken == 1 || jyouken == 8 || jyouken == 9) {//全消し条件の枝刈り
		if (color == "　" || color == "色") {
			if ((num[0] > 0 && num[0] < 4) || (num[1] > 0 && num[1] < 4) || (num[2] > 0 && num[2] < 4) || (num[3] > 0 && num[3] < 4) || (num[4] > 0 && num[4] < 4)) {
				check_flag = 1;
			}
		}
		if (color == "赤") {
			if (num[0] > 0 && num[0] < 4) {
				check_flag = 1;
			}
		}
		if (color == "緑") {
			if (num[0] > 1 && num[1] < 4) {
				check_flag = 1;
			}
		}
		if (color == "青") {
			if (num[2] > 0 && num[2] < 4) {
				check_flag = 1;
			}
		}
		if (color == "緑") {
			if (num[3] > 0 && num[3] < 4) {
				check_flag = 1;
			}
		}
		if (color == "紫") {
			if (num[4] > 0 && num[4] < 4) {
				check_flag = 1;
			}
		}
	}
	if (jyouken == 10 || jyouken == 11) {//n色同時消し条件の枝刈り
		if (((num[0] > 3) + (num[1] > 3) + (num[2] > 3) + (num[3] > 3) + (num[4] > 3)) < n_jyouken) {
			check_flag = 1;
		}
	}
	if (jyouken == 12 || jyouken == 13) {//n個同時消し条件の枝刈り
		if (color == "　" || color == "色") {
			if ((num[0] + num[1] + num[2] + num[3] + num[4]) < n_jyouken) {
				check_flag = 1;
			}
		}
		if (color == "赤") {
			if (num[0] < n_jyouken) {
				check_flag = 1;
			}
		}
		if (color == "緑") {
			if (num[1] < n_jyouken) {
				check_flag = 1;
			}
		}
		if (color == "青") {
			if (num[2] < n_jyouken) {
				check_flag = 1;
			}
		}
		if (color == "緑") {
			if (num[3] < n_jyouken) {
				check_flag = 1;
			}
		}
		if (color == "紫") {
			if (num[4] < n_jyouken) {
				check_flag = 1;
			}
		}
	}
}

//1連鎖目になるぷよのうち、最初から盤面にあるぷよの数を出力
void hakkaposition(vector<int[15][8]> &tmpfield, int seed1[][8], int seed2[][8], int seed3[][8], int seed4[][8], int seed5[][8]) {
	int hakka[15][8] = {};
	hakkanum = 0;
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			if (seed1[c][r] == 1) {
				hakka[c][r] = 1;
			}
			else if (seed2[c][r] == 1) {
				hakka[c][r] = 2;
			}
			else if (seed3[c][r] == 1) {
				hakka[c][r] = 3;
			}
			else if (seed4[c][r] == 1) {
				hakka[c][r] = 4;
			}
			else if (seed5[c][r] == 1) {
				hakka[c][r] = 5;
			}
		}
	}
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			if (hakka[c][r] != 0) {
				if (hakka[c][r] == tmpfield[0][c][r]) {
					hakkanum += 1;
				}
			}
		}
	}
}

//実際にツモを置く処理
int r, c;
void haichi(int k, int tekazu, vector<int[15][8]> &tmpfield, vector<pair<int, int>> &answer) {
	if (k == tekazu) {
		//確認用
		a += 1;
		if (a % 10000 == 0) {
			//確認用
			//cout << a << endl;
		}
	}
	else {
		for (int i = 1; i < 7; i++) {
			top_generation(tmpfield[k]);
			//配置制限
			if (top[i - 1] == 0) { goto LABEL2; }//すでに13段目まで埋まっている場合は飛ばす
			else {
				switch (i) {
				case 1:
					if (top[1] == 0) {//2列目が13段目まで埋まっていれば1列目には置けない
						goto LABEL2;
					}
					if (top[1] == 1) {//2列目が12段目まで埋まっているとき
						if (top[3] > 1) {//クイックターン不可}
							if (top[2] > 2 && top[3] > 2 && top[4] < 2) {
								goto LABEL2;
							}
							else if (top[2] != 2 && top[3] != 2 && top[4] != 2 && top[5] != 2) {
								goto LABEL2;
							}
						}
					}
					else if (top[0] == 1) {//1列目が12段目まで埋まっている場合
						if (top[3] < 2) {
							if (top[1] != 2 && top[2] != 2) {
								goto LABEL2;
							}
						}
						else if (top[4] < 2) {
							if (top[1] != 2 && top[2] != 2 && top[3] != 2) {
								goto LABEL2;
							}
						}
						else {
							if (top[1] != 2 && top[2] != 2 && top[3] != 2 && top[4] != 2 && top[5] != 2) {
								goto LABEL2;
							}
						}
					}
					break;
				case 2:
					if (top[1] == 1) {//2列目12段まで埋まっているとき
						if (top[3] > 1) {//クイックターン不可
							if (top[2] > 2 && top[3] > 2 && top[4] < 2) {
								goto LABEL2;
							}
							else if (top[2] != 2 && top[3] != 2 && top[4] != 2 && top[5] != 2) {
								goto LABEL2;
							}
						}
					}
					break;
				case 4:
					if (top[3] == 1) {//4列目12段まで埋まっているとき
						if (top[1] > 1) {
							if (top[0] != 2 && top[1] != 2 && top[2] != 2) {
								goto LABEL2;
							}
						}
					}
					break;
				case 5:
					if (top[3] == 0) {//4列目が13段目まで埋まっていれば5列目には置けない
						goto LABEL2;
					}
					if (top[3] == 1) {//4列目が12段目まで埋まっている場合
						if (top[1] > 1) {//クイックターン不可}
							if (top[0] != 2 && top[1] != 2 && top[2] != 2) {
								goto LABEL2;
							}
						}
					}
					break;
				case 6:
					if (top[3] == 0 || top[4] == 0) {//4or5列目が13段目まで埋まっていれば1列目には置けない
						goto LABEL2;
					}
					if (top[3] == 1) {//4列目が12段目まで埋まっている場合
						if (top[1] > 1) {//クイックターン不可}
							if (top[0] != 2 && top[1] != 2 && top[2] != 2) {
								goto LABEL2;
							}
						}
					}
					else if (top[4] == 1) {//5列目が12段目まで埋まっている場合
						if (top[2] > 2 && top[3] > 2 && top[1] < 2) {
							goto LABEL2;
						}
						else if (top[0] != 2 && top[1] != 2 && top[2] != 2 && top[3] != 2) {
							goto LABEL2;
						}
					}
					break;
				}

				for (int j = 0; j < 4; j++) {
					initialization();

					pair<int, int> place(i, j);
					answer[k] = place;

					//フィールド記憶
					for (c = 1; c < 14; c++) {
						for (r = 1; r < 7; r++) {
							field[c][r] = tmpfield[k][c][r];
						}
					}
					top_generation(field);

					//配置
					switch (j) {
					case 0:
						field[top[i - 1]][i] = haipuyo[k].first;
						if (top[i - 1] != 1) {//14段目処理
							field[top[i - 1] - 1][i] = haipuyo[k].second;//ここ要回し条件
						}
						break;
					case 1:
						if (haipuyo[k].first == haipuyo[k].second) {//ゾロの場合不要
							goto LABEL;
						}
						if (top[i - 1] == 1) {//14段目に子ぷよは置けない
							goto LABEL;
						}
						else {
							field[top[i - 1]][i] = haipuyo[k].second;
							field[top[i - 1] - 1][i] = haipuyo[k].first;
						}
						break;
					case 2:
						if (haipuyo[k].first == haipuyo[k].second) {//ゾロの場合不要
							goto LABEL;
						}
						if (i == 1) {
							goto LABEL;
						}
						if (top[i - 2] == 0) {//左列が13段目まで埋まっている場合
							goto LABEL;
						}
						else if (top[i - 2] == 1) {//左列が12段目まで埋まっている場合
							switch (i) {
							case 2:
								if (top[3] < 2) {
									if (top[1] != 2 && top[2] != 2) {
										goto LABEL;
									}
								}
								else if (top[4] < 2) {
									if (top[1] != 2 && top[2] != 2 && top[3] != 2) {
										goto LABEL;
									}
								}
								else {
									if (top[1] != 2 && top[2] != 2 && top[3] != 2 && top[4] != 2 && top[5] != 2) {
										goto LABEL;
									}
								}
								break;
							case 3:
								if (top[3] > 1) {//クイックターン不可
									if (top[2] > 2 && top[3] > 2 && top[4] < 2) {
										goto LABEL;
									}
									else if (top[2] != 2 && top[3] != 2 && top[4] != 2 && top[5] != 2) {
										goto LABEL;
									}
								}
								break;
							case 5:
								break;
							case 6:
								break;
							}
						}
						else {
							field[top[i - 1]][i] = haipuyo[k].first;
							field[top[i - 2]][i - 1] = haipuyo[k].second;
						}
						break;
					case 3:
						if (i == 6) {
							goto LABEL;
						}
						if (top[i] == 0) {//右列が13段目まで埋まっている場合
							goto LABEL;
						}
						//else if (top[i] == 1) {//右列が12段目まで埋まっている場合
						//}
						else {
							field[top[i - 1]][i] = haipuyo[k].first;
							field[top[i]][i + 1] = haipuyo[k].second;
						}
						break;
					}

					//フィールド配置、連結数判定、消えるマス判定
					first_field(field);
					//マルチ条件のクリアチェック
					multi_clear(seed1, seed2, seed3, seed4, seed5, answer);
					//同時消し条件のクリアチェック
					doujikeshi_clear(de, seed1, seed2, seed3, seed4, seed5, answer);

					//配置後のフィールド記憶
					if (k < tekazu - 1) {
						for (c = 1; c < 14; c++) {
							for (r = 1; r < 7; r++) {
								tmpfield[k + 1][c][r] = field[c][r];
							}
						}
					}

					//連鎖処理
					chain = 0;
					while (memcmp(de, zero, sizeof(zero))) {
						if (chain == 0 && k == tekazu - 1) {
							hakkaposition(tmpfield,seed1,seed2,seed3,seed4,seed5);
						}
						chain += 1;
						//ぷよ消去
						puyo_delete(de, field);
						//落下判定
						fall(field);
						//フィールド記憶
						if (k < tekazu - 1) {
							for (c = 1; c < 14; c++) {
								for (r = 1; r < 7; r++) {
									tmpfield[k + 1][c][r] = field[c][r];
								}
							}
						}
						//初期化
						initialization();
						first_field(field);
						//マルチ条件のクリアチェック
						multi_clear(seed1, seed2, seed3, seed4, seed5, answer);
						//同時消し条件のクリアチェック
						doujikeshi_clear(de, seed1, seed2, seed3, seed4, seed5, answer);
					}

					//クリア条件
					switch (jyouken) {
					case 1://cぷよ全て消すべし
						allclear_check(answer);
						break;
					case 6://n連鎖するべし
						if (chain == n_jyouken) {
							dispans(answer);
						}
						break;
					case 7://n連鎖以上するべし
						if (chain >= n_jyouken) {
							dispans(answer);
						}
						break;
					case 8://n連鎖&cぷよ全て消すべし
						if (chain == n_jyouken) {
							allclear_check(answer);
						}
						break;
					case 9://n連鎖以上&cぷよ全て消すべし
						if (chain >= n_jyouken) {
							allclear_check(answer);
						}
						break;
					}
					if (field[2][3] != 0) {//窒息処理
						goto LABEL;
					}
					cut_check(k, tekazu);
					if (check_flag == 1) {
						goto LABEL;
					}

					//再帰
					haichi(k + 1, tekazu, tmpfield, answer);
				LABEL:
					;
				}
			}
		LABEL2:
			;
		}
	}
}

int d = 0;
vector<int> clearcount_check;
vector<vector<vector<pair<int, int>>>> answerlist2;
vector<vector<pair<int, int>>> haipuyolist;
vector<vector<int>> hakkanumlist2;

void tumopattern(int k, int tekazu, vector<int> tumocolor,vector<int> tumopuyonum, vector<vector<int>> tmptumopuyonum, vector<int[15][8]> &tmpfield, vector<pair<int, int>> &answer) {
	if (k == tekazu) {
		d += 1;
		if (d % 10000 == 0) {
			//cout << d << endl;
		}
		/*vector<int> puyonum_check(5, 0);
		for (int i = 0; i < tekazu; i++) {
			puyonum_check[haipuyo[i].first-1] += 1;
			puyonum_check[haipuyo[i].second-1] += 1;
		}*/
		//if (tmptumopuyonum[0] == puyonum_check) {
			//cout << "aaa" << endl;
			//解答チェック
			for (int c = 1; c < 14; c++) {
				for (int r = 1; r < 7; r++) {
					field[c][r] = tmpfield[0][c][r];
				}
			}
			initialization();
			first_field(field);
			haichi(0, tekazu, tmpfield, answer);
			if (clear_count > 0){
				haipuyolist.push_back(haipuyo);
				clearcount_check.push_back(clear_count);
				answerlist2.push_back(answerlist);
				hakkanumlist2.push_back(hakkanumlist);
			}
		//}
	}
	else {
		for (int i = 0; i < tumocolor.size(); i++) {
			for (int j = i; j < tumocolor.size(); j++) {
				tumopuyonum = tmptumopuyonum[k];
				if (tumopuyonum[tumocolor[i] - 1] > 0 && tumopuyonum[tumocolor[j] - 1] > 0) {
					pair<int, int> p(tumocolor[i], tumocolor[j]);
					haipuyo[k] = p;
					tumopuyonum[tumocolor[i] - 1] -= 1;
					tumopuyonum[tumocolor[j] - 1] -= 1;
					if (k < tekazu - 1) {
						tmptumopuyonum[k + 1] = tumopuyonum;
					}
					tumopattern(k + 1, tekazu, tumocolor, tumopuyonum, tmptumopuyonum, tmpfield, answer);
				}
			}
		}
	}
}

int renketu_3;
int renketu_2;
void renketusu(int renketu1[][8], int renketu2[][8], int renketu3[][8], int renketu4[][8], int renketu5[][8]) {
	renketu_3 = 0;
	renketu_2 = 0;
	for (int i = 1; i < 14; i++) {
		for (int j = 1; j < 7; j++) {
			if (renketu1[i][j] == 2 || renketu2[i][j] == 2 || renketu3[i][j] == 2 || renketu4[i][j] == 2 || renketu5[i][j] == 2) {
				renketu_3 += 1;
			}
			else if (renketu1[i][j] == 1 || renketu2[i][j] == 1 || renketu3[i][j] == 1 || renketu4[i][j] == 1 || renketu5[i][j] == 1) {
				renketu_2 += 1;
			}
		}
	}
	renketu_2 = (renketu_2 - renketu_3 * 2) / 2;
}

//ジェネレータくん
void generate() {
	cout << "条件を選択してください(番号で入力)" << endl;
	cout << "1:cぷよすべて消すべし" << endl;
	cout << "2:n色消すべし(未実装)" << endl;
	cout << "3:n色以上消すべし(未実装)" << endl;
	cout << "4:cぷよn個消すべし" << endl;
	cout << "5:cぷよn個以上消すべし" << endl;
	cout << "6:n連鎖するべし" << endl;
	cout << "7:n連鎖以上するべし" << endl;
	cout << "8:n連鎖＆cぷよ全て消すべし" << endl;
	cout << "9:n連鎖以上＆cぷよ全て消すべし" << endl;
	cout << "10:n色同時に消すべし" << endl;
	cout << "11:n色以上同時に消すべし" << endl;
	cout << "12:cぷよn個同時に消すべし" << endl;
	cout << "13:cぷよn個以上同時に消すべし" << endl;
	cout << "14:cぷよn箇所同時に消すべし(未実装)" << endl;
	cout << "15:cぷよn箇所以上同時に消すべし(未実装)" << endl;
	cout << "16:cぷよn連結で消すべし(未実装)" << endl;
	cout << "17:cぷよn連結以上で消すべし(未実装)" << endl;

	cin >> jyouken;
	if (jyouken == 0) {
		cout << "条件入力が不正です" << endl;
		return;
	}

	switch (jyouken) {
	case 1:
		cout << "cを入力してください(空白は全角スペース)" << endl;
		cin >> color;
		break;
	case 2:
	case 3:
	case 6:
	case 7:
	case 10:
	case 11:
		cout << "nを入力してください" << endl;
		cin >> n_jyouken;
		if (n_jyouken == 0) {
			cout << "入力が不正です" << endl;
			return;
		}
		break;
	case 8:
	case 9:
		cout << "nを入力してください" << endl;
		cin >> n_jyouken;
		if (n_jyouken == 0) {
			cout << "入力が不正です" << endl;
			return;
		}
		cout << "cを入力してください(空白は全角スペース)" << endl;
		cin >> color;
		break;
	case 12:
	case 13:
		cout << "cを入力してください(空白は全角スペース)" << endl;
		cin >> color;
		cout << "nを入力してください" << endl;
		cin >> n_jyouken;
		if (n_jyouken == 0) {
			cout << "入力が不正です" << endl;
			return;
		}
		break;
	}

	int mondaisu = 0;
	int count;
	cout << "赤ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(1);
	}
	cout << "緑ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(2);
	}
	cout << "青ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(3);
	}
	cout << "黄ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(4);
	}
	cout << "紫ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(5);
	}
	cout << "おじゃまぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(6);
	}
	cout << "壁の個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(7);
	}
	cout << "鉄ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(8);
		tetsu_flag = 1;
	}
	cout << "固ぷよの個数" << endl;
	cin >> count;
	if (count != 0) {
		puyonum.push_back(count);
		colorlist.push_back(9);
		kata_flag = 1;
	}
	cout << "手数" << endl;
	cin >> tekazu;

	vector<int> puyonum_clone = puyonum;
	//壁生成
	for (int i = 0; i < 15; i++) {
		field[i][0] = 7;
		field[i][7] = 7;
		zero_field[i][0] = 7;
		zero_field[i][7] = 7;
	}
	for (int j = 0; j < 8; j++) {
		field[0][j] = 7;
		field[14][j] = 7;
		zero_field[0][j] = 7;
		zero_field[14][j] = 7;
	}

	vector<vector<int>> tmptumopuyonum(tekazu);

	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
	std::uniform_int_distribution<> randpuyo(0, puyonum.size() - 1);

	while (1) {
		haipuyo.clear();

		for (int c = 1; c < 14; c++) {
			for (int r = 1; r < 7; r++) {
				field[c][r] = 0;
			}
		}
		//初期化
		puyonum = puyonum_clone;
		initialization();
		clear_count = 0;
		clear_count_z = 0;
		int cnt = 0;
		int jp;
		int kp;
		//ツモ生成
		for (int i = 0;; i++) {
			if (cnt == tekazu * 2) {
				break;
			}
			int rndp = randpuyo(mt);
			if (colorlist[rndp] < 6) {
				if (puyonum[rndp] > 0) {
					cnt = cnt + 1;
					if (cnt % 2 == 0) {
						jp = colorlist[rndp];
						puyonum[rndp] -= 1;
					}
					else {
						kp = colorlist[rndp];
						pair<int, int> tumo(jp, kp);
						puyonum[rndp] -= 1;
						haipuyo.push_back(tumo);
					}
				}
			}
		}

		//ツモに配分されたぷよ数
		//int tumopuyonum[5] = {};
		vector<int> tumopuyonum(5, 0);

		if (haipuyo[0].first < 0) {
			continue;
		}
		for (int i = 0; i < tekazu; i++) {
			tumopuyonum[haipuyo[i].first- 1] += 1;
			tumopuyonum[haipuyo[i].second -1] += 1;
		}

		tmptumopuyonum[0]=tumopuyonum;

		//ツモに配分されたぷよの色の種類
		vector<int> tumocolor;
		for (int i = 0; i < 5; i++) {
			if (tumopuyonum[i] > 0) {
				tumocolor.push_back(i+1);
			}
		}

		////最後の手がゾロを除外
		//if (haipuyo[tekazu - 1].first == haipuyo[tekazu - 1].second) {
		//	continue;
		//}

		std::uniform_int_distribution<> randrow(1, 6);
		int rown;
		vector<int> zerovec(puyonum.size(), 0);
		
		//フィールド生成
		while (puyonum != zerovec) {
			top_generation(field);
			rown = randrow(mt);
			int rndp = randpuyo(mt);
			if (puyonum[rndp] > 0) {
				field[top[rown - 1]][rown] = colorlist[rndp];
				puyonum[rndp] -= 1;
			}
		}

		first_field(field);
		if (memcmp(de, zero, sizeof(zero))) {
			continue;
		}

		renketusu(renketu1, renketu2, renketu3, renketu4, renketu5);

		//初期フィールド記憶
		vector<int[15][8]> tmpfield(tekazu);
		for (int c = 1; c < 14; c++) {
			for (int r = 1; r < 7; r++) {
				tmpfield[0][c][r] = field[c][r];
			}
		}

		vector<pair<int, int>> answer(tekazu);

		clearcount_check.clear();
		haipuyolist.clear();
		answerlist.clear();
		answerlist2.clear();
		hakkanumlist.clear();
		hakkanumlist2.clear();

		tumopattern(0, tekazu, tumocolor, tumopuyonum, tmptumopuyonum, tmpfield, answer);
		if (clearcount_check.size() > 0) {
			if (renketu_3 + renketu_2 > 3) {
				vector<int>::iterator iter_c = min_element(clearcount_check.begin(), clearcount_check.end());
				int index = distance(clearcount_check.begin(), iter_c);
				if (haipuyolist[index][tekazu - 1].first != haipuyolist[index][tekazu - 1].second) {
					mondaisu += 1;
					cout << "連結数" << endl;
					cout << "3連結＝" << renketu_3 << ", 2連結＝" << renketu_2 << endl;
					cout << "回答数＝" << *iter_c << "/" << accumulate(clearcount_check.begin(), clearcount_check.end(), 0) << endl;
					/*cout << "発火色の元々ある数" << endl;
					for (int i = 0; i < hakkanumlist2.size(); i++) {
						cout << hakkanumlist2[i][0] << "," ;
					}
					cout << "" << endl;*/
					cout << "ぷよ図" << endl;
					for (int c = 1; c < 14; c++) {
						for (int r = 1; r < 7; r++) {
							cout << tmpfield[0][c][r];
						}
						cout << "" << endl;
					}
					cout << "ぷよ譜" << endl;
					for (int i = 0; i < haipuyo.size(); i++) {
						cout << haipuyolist[index][i].first << haipuyolist[index][i].second << endl;
					}

					url_export(tmpfield[0], haipuyolist[index]);
				}
			}
		}

		if (mondaisu == 5) {
			break;
		}

		
		//haichi(0, tekazu, tmpfield, answer);
		////if (clear_count_z > 0) {
		//if(clear_count > 0){
		//	cout << "回答数＝" << clear_count << endl;
		//	cout << "回答数2＝" << clear_count_z << endl;
		//	/*for (int i = 0; i < answerlist.size(); i++) {
		//		cout << "clear!" << endl;
		//		for (int l = 0; l < answerlist[i].size(); l++) {
		//			cout << answerlist[i][l].first;
		//			if (answerlist[i][l].second == 0) {
		//				cout << "↑" << endl;
		//			}
		//			else if (answerlist[i][l].second == 1) {
		//				cout << "↓" << endl;
		//			}
		//			else if (answerlist[i][l].second == 2) {
		//				cout << "←" << endl;
		//			}
		//			else if (answerlist[i][l].second == 3) {
		//				cout << "→" << endl;
		//			}
		//		}
		//	
		//	}*/
		//	cout << "ぷよ図" << endl;
		//	for (int c = 1; c < 14; c++) {
		//		for (int r = 1; r < 7; r++) {
		//			cout << tmpfield[0][c][r];
		//		}
		//		cout << "" << endl;
		//	}
		//	cout << "ぷよ譜" << endl;
		//	for (int i = 0; i < haipuyo.size(); i++) {
		//		cout << haipuyo[i].first << haipuyo[i].second << endl;
		//	}

		//	cout << "連結数" << endl;
		//	cout << "3連結＝" << renketu_3 << ", 2連結＝" << renketu_2 << endl;

		//	//cout << "発火色のうち初期盤面にある数＝" << hakkanum << endl;

		//	cout << url_export(tmpfield[0],haipuyo) << endl;

		//	mondaisu += 1;
		//	if (mondaisu == 5) {
		//		break;
		//	}
			
		//}
	}
}

void solver() {
	//url入力
	url_generation();

	//壁生成
	for (int i = 0; i < 15; i++) {
		field[i][0] = 7;
		field[i][7] = 7;
		zero_field[i][0] = 7;
		zero_field[i][7] = 7;
	}
	for (int j = 0; j < 8; j++) {
		field[0][j] = 7;
		field[14][j] = 7;
		zero_field[0][j] = 7;
		zero_field[14][j] = 7;
	}

	int tekazu = haipuyo.size();

	//初期フィールド記憶
	vector<int[15][8]> tmpfield(tekazu);
	for (int c = 1; c < 14; c++) {
		for (int r = 1; r < 7; r++) {
			tmpfield[0][c][r] = field[c][r];
		}
	}

	vector<pair<int, int>> answer(tekazu);

	haichi(0,tekazu,tmpfield,answer);
	cout << "回答数＝" << clear_count << endl;
	for (int i = 0; i < answerlist.size(); i++) {
		cout << "clear!" << endl;
		for (int l = 0; l < answerlist[i].size(); l++) {
			cout << answerlist[i][l].first;
			if (answerlist[i][l].second == 0) {
				cout << "↑" << endl;
			}
			else if (answerlist[i][l].second == 1) {
				cout << "↓" << endl;
			}
			else if (answerlist[i][l].second == 2) {
				cout << "←" << endl;
			}
			else if (answerlist[i][l].second == 3) {
				cout << "→" << endl;
			}
		}
	}
}

int main() {

	//なぞぷよジェネレータくん
	//generate();

	//なぞぷよソルバくん
	solver();

	//なぞぷよスクショくん
	//screenshot();
}