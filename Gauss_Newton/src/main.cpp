#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(){
	double ar=1.0, br=2.0, cr=1.0;//真实参数值
	double ae=2.0, be=-1.0,ce=7.0;//估计参数值
	int N=100;
	double w_sigma=1.0;			  //噪声干扰值
	double inv_sigma=1.0/w_sigma;
	cv::RNG rng;
	
	//生成数据点，用两个向量来存储
	vector<double> x_data, y_data;
	for(int i=0;i<N;i++){
		double x=i/100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar*x*x+br*x+cr)+rng.gaussian(w_sigma*w_sigma));	
	}
	
	//Gauss_Newton迭代
	int iterations = 100;//最大迭代次数
	double cost = 0, lastcost = 0;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	for(int i;i<iterations;i++){
		//初始化海森矩阵
		Matrix3d H = Matrix3d::Zero();
		Vector3d b = Vector3d::Zero();
		cost = 0;
		//每迭代一次要计算100个点
		for(int j=0;j<N;j++){
			double xi = x_data[j],yi = y_data[j];
			double error = yi - exp(ae*xi*xi+be*xi+ce);
			Vector3d J;
			//初始化雅可比矩阵
			J[0] = -xi*xi*exp(ae*xi*xi+be*xi+ce);
			J[1] = -xi*exp(ae*xi*xi+be*xi+ce);
			J[2] = -exp(ae*xi*xi+be*xi+ce);
		H += J*J.transpose();//3*3
		b += -error*J;		//代码定义的J为列向量，公式中定义为行向量
		cost += error*error;//目标函数值
		}
		
		//求步长，通过解线形方程Hx=b
		Vector3d dx = H.ldlt().solve(b);//LDLT分解方法
		if (isnan(dx[0])){
			cout << "result is nan!" << endl;
			break;
		}
		
		//如果目标函数不再减小就停止
		if(i>0 && cost >=lastcost)
		{
			cout << "cost: " << cost << ">=last cost: " << lastcost << ", break." << endl;
			break;
		}
		//更新变量，上一时刻的值+变量
		ae += dx[0];
		be += dx[1];
		ce += dx[2];
		lastcost = cost;
		cout << "total cost: "<<cost<<", \t\t update: "<<dx.transpose()<<
		"\t\t estimated params: "<< ae<<","<<be<<","<<ce<<endl;
	}
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
	cout<<"solve time cost = "<< time_used.count() << " seconds. "<<endl;
	cout<<"estimated abc = "<< ae << ", " <<be << ", " << ce <<endl;
	return 0;
}


