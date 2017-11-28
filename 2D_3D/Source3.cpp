#include <iostream>
#include<sstream>
#include<fstream>
#include <time.h>
#include <omp.h>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
fstream fout("out.xyz", ios::out);
void get3D(int ccc, Mat F, Mat L_P, Mat R_P)
{
	//**************************************************read pair image
	char str[15], strL[20] = "L/", strR[20] = "R/";
	if (ccc > 99)
	{
		sprintf_s(str, "0%d.jpg", ccc);
	}
	else if (ccc > 9)
	{
		sprintf_s(str, "00%d.jpg", ccc);
	}
	else { sprintf_s(str, "000%d.jpg", ccc); }
	strcat_s(strL, str);
	strcat_s(strR, str);
	Mat L = imread(strL);//讀圖片
	Mat R = imread(strR);//
	vector<Mat> Lpt;//左邊取點
	vector<Mat> Rpt;//右邊取點
	//****************************sample points
	for (int i = 0; i < L.rows; ++i)//取點
	{
		uchar tempmaxL = 0, tempmaxR = 0;
		Point maxL(0, 0), maxR(0, 0);
		for (int j = 0; j < L.cols; ++j)
		{
			if (L.ptr<uchar>(i, j)[2] > tempmaxL)
			{
				tempmaxL = L.ptr<uchar>(i, j)[2]; maxL.x = i; maxL.y = j;
			}
			if (R.ptr<uchar>(i, j)[2] > tempmaxR)
			{
				tempmaxR = R.ptr<uchar>(i, j)[2]; maxR.x = i; maxR.y = j;
			}
		}
		if (tempmaxL > 200){ Lpt.push_back((Mat_<double>(3, 1) << maxL.y, maxL.x, 1)); }
		if (tempmaxR > 200){ Rpt.push_back((Mat_<double>(3, 1) << maxR.y, maxR.x, 1)); }
	}//calculate I= F X
	vector<Mat> III;
	for (int i = 0; i < Lpt.size(); ++i)
	{
		III.push_back(F*Lpt[i]);
	}//projection error
	vector<int> refR;//記錄L第I個對應到R的哪個
	for (int i = 0; i < Lpt.size(); ++i)
	{
		int mindistance = 0;
		double value = DBL_MAX;
		double tmpp;
		for (int j = 0; j < Rpt.size(); ++j)
		{
			tmpp = abs(III[i].dot(Rpt[j]));
			if (tmpp < value)
			{
				mindistance = j;
				value = tmpp;
			}
		}
		refR.push_back((value < 0.01 ? mindistance : -1));
	}//解SVD 3D點輸出 solve svd  
	vector<Point3d> D3point;
	Mat svdsvd(4, 4, CV_64FC1, Scalar(0));
	Mat ans;
	for (int i = 0; i < Lpt.size(); ++i)
	{
		if (refR[i] != -1)
		{
			svdsvd.row(0) = (L_P.row(2)*(Lpt[i].ptr<double>(0)[0]) - L_P.row(0));
			svdsvd.row(1) = (L_P.row(2)*(Lpt[i].ptr<double>(1)[0]) - L_P.row(1));
			svdsvd.row(2) = (R_P.row(2)*(Rpt[refR[i]].ptr<double>(0)[0]) - R_P.row(0));
			svdsvd.row(3) = (R_P.row(2)*(Rpt[refR[i]].ptr<double>(1)[0]) - R_P.row(1));
			SVD::solveZ(svdsvd, ans);
			ans /= ans.ptr<double>(3)[0];
			D3point.push_back(Point3d(ans.ptr<double>(0)[0], ans.ptr<double>(1)[0], ans.ptr<double>(2)[0]));
		}
	}
	char tmppp[100];
	for (int z = 0; z < D3point.size(); ++z)//output  3d point data 
	{
		sprintf_s(tmppp, "%f\t%f\t%f\n", D3point[z].x, D3point[z].y, D3point[z].z);
		fout << tmppp;
	}
	/////*************************************************************
}
int main()
{
	//參數設定 Left/Right camera 
	Mat L_K = (Mat_<double>(3, 3) << 1031.107181256652, 0.000000000000, 300.315754576448, 0.000000000000, 1031.922726531495, 596.128631557107, 0.000000000000, 0.000000000000, 1.000000000000);//intrinsic parameter 
	Mat L_RT = (Mat_<double>(3, 4) << 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000);//extrinsic parameter 
	Mat R_K = (Mat_<double>(3, 3) << 1034.184348727144, 0.000000000000, 417.455360200033, 0.000000000000, 1034.889311554467, 615.052068517334, 0.000000000000, 0.000000000000, 1.000000000000);//intrinsic parameter 
	Mat R_RT = (Mat_<double>(3, 4) << 0.960420625577, 0.010140226529, 0.278369175326, -70.168656978049, -0.009291786823, 0.999947294142, -0.004367108410, -0.002608446641, -0.278398787108, 0.001607713956, 0.960464226607, 13.498989529018);//extrinsic parameter 
	Mat F = (Mat_<double>(3, 3) << -0.000000050362, 0.000002369273, -0.001301863689, 0.000000811157, -0.000000004467, -0.011984436200, -0.000689813339, 0.010646819054, 1.000000000000);//Fundmental matrix 
	Mat L_P = L_K*L_RT;
	Mat R_P = R_K*R_RT;
	int count = 364;//總圖片數
	clock_t c1 = clock(), c2;//計時
	cout << "程式執行中" << endl;
#pragma omp parallel for //parallel caculation ,openMP boost
	for (int ccc = 0; ccc < count; ++ccc)//iteration over every picture
	{	
		get3D(ccc, F, L_P, R_P);//input(iteration,Fundmental matrix,Left parameter,Right parameter)
	}
	fout.close();
	c2 = clock();
	printf("執行時間 %lf 秒", (c2 - c1) / (double)(CLOCKS_PER_SEC));
	system("pause");
}