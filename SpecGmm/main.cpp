//
//  main.cpp
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/4.
//  Copyright (c) 2014 Huang, Tse-Han. All rights reserved.
//

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/KroneckerProduct>

#ifndef __SpecGmm__D3Matrix__
#define __SpecGmm__D3Matrix__
#include "D3Matrix.h"
#endif /* defined(__SpecGmm__D3Matrix__) */

#ifndef __SpecGmm__Test__
#include "test.h"
#endif /* defined(__SpecGmm__Test__) */

#ifndef __SpecGmm__TesnsorPower__
#include "TensorPower.h"
#endif /* defined(__SpecGmm__TesnsorPower__) */

#ifndef __SpecGmm__SpecGmm__
#include "SpecGmm.h"
#endif /* defined(__SpecGmm__SpecGmm__) */

#ifndef SpecGmm_DataGenerator_h
#include "DataGenerator.h"
#endif

using namespace std;
using namespace Eigen;

template <typename Derived>
void test2(const MatrixBase<Derived> &input, MatrixBase<Derived> &output);

void testTFuction();
void testTensorPower();
void testSpecGmm();
void testRandom();

int main(int argc, const char * argv[]) {

    //TensorPower::testTFuction();
    //testTensorPower();
    //testOuter();
    testSpecGmm();
    //testRandom();
    
    return 0;
}

void testRandom() {
    // mean std
    
    /*
    normal_distribution<double> normal(3.0, 4.0);
    default_random_engine generator;
    generator.seed(1000);
    cout << normal(generator) << endl;
     cout << normal(generator) << endl;
     cout << normal(generator) << endl;
     */
    //DataGenerator test(5,3,10,0.1,10);
    //cout << test.X();
}


// SVD Example
template <typename Derived>
void test2(const MatrixBase<Derived> &input, MatrixBase<Derived> &output) {
    
    MatrixXf m = MatrixXf::Random(3,2);
    cout << "Here is the matrix m:" << endl << m << endl;
    const JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
    cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
    cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
    Vector3f rhs(1, 0, 0);
    cout << "Now consider this rhs vector:" << endl << rhs << endl;
    cout << "A least-squares solution of m*x = rhs is:" << endl << svd.solve(rhs) << endl;

    output=svd.matrixU();

}

void testTensorPower() {
    
    MatrixXd t1(3,3);
    MatrixXd t2(3,3);
    MatrixXd t3(3,3);
    t1 << 1,2,3,4,5,6,7,8,9;
    t2 << 10,11,12,13,14,15,16,17,18;
    t3 << 19,20,21,22,23,24,25,26,27;
    
    D3Matrix<MatrixXd> T(3,3,3);
    T.setLayer(0, t1);
    T.setLayer(1, t2);
    T.setLayer(2, t3);
    
    TensorPower<MatrixXd> power(T, 10, 10);
    //power.compute(T, 10, 10);
    //cout<< "ans" << power.theta() << endl;
    
    MatrixXd solution(3,1);
    solution << 0.458361, 0.570097, 0.681832;
    
    cout << "===== TEST CASE RESULT : TensorPower ===== " << endl;
    cout << "Test error should be nearly zero" << endl;
    cout << "test error = " << endl << power.theta()-solution << endl ;
    
}

void testSpecGmm() {
    // Center =
    //5.5599    7.0493
    //7.4210    2.9892
    //0.7244    4.7072
    //3.6730    4.3834
    MatrixXd X(4,20);
    X << 5.7896,5.6359,5.5301,5.5245,5.0695,5.8445,5.2919,5.6945,5.9686,5.2731
        ,6.8700,6.7659,7.1758,6.3803,7.0528,7.3412,7.0229,7.4555,6.9957,7.3049,
        7.1263,7.9713,7.1027,6.5886,7.5687,7.3943,7.7713,7.2862,7.5625,7.1713
        ,2.7254,2.9215,2.5417,2.9199,2.7602,3.2831,2.6862,2.9441,2.7830,2.9657,
        0.5780,0.8247,0.6257,1.1336,0.6533,0.6168,1.1998,0.4550,0.5137,0.3889
        ,4.9361,4.3415,5.3407,4.4333,5.0126,4.9296,4.7340,4.4581,4.5895,5.0449,
        3.3472,3.7278,4.0639,3.7934,3.4680,3.3921,3.7881,4.1433,3.3441,3.9458
        ,4.4003,4.0163,4.4981,3.9663,4.5073,3.7697,3.9186,4.2988,3.9143,4.7658;
    

    /**
    Center =
    6.5445    3.1597    1.8806
    4.4010    6.9023    3.2828
    4.3265    0.2085    4.7405
    4.0927    5.8968    5.3529
    1.5270    2.7491    5.8788
    */
    MatrixXd Y(5,30);
    Y << 6.2927,6.5660,6.2210,6.4903,6.6331,6.5048,6.4256,6.2634,6.5338,6.8035,
    3.4894,3.3732,3.1784,3.2992,2.9455,3.4549,2.5877,3.2452,3.3003,2.9435,2.5123,
    1.6103,1.7337,1.7756,1.3176,2.2257,1.4583,1.9476,1.5875,1.7245,
    4.9342,4.0187,4.1915,4.5295,4.3297,4.2107,4.7183,4.7614,4.4910,4.3328,7.1378,
    7.4547,6.8908,6.6993,7.2232,7.5199,6.6356,6.6785,6.9969,6.8513,3.6953,3.1425,
    3.7438,3.4817,3.3844,3.6345,3.6526,3.5146,3.7795,2.9550,
    5.0358,4.0134,4.5307,4.0849,4.1098,4.3846,4.2170,4.5567,4.1470,4.2717,0.4247,
    0.7256,0.0022,0.6104,0.9517,0.3202,0.3222,0.0822,0.0413,0.5754,5.1657,5.2033,
    4.4487,4.8632,4.2583,3.8923,4.7593,4.9443,4.6721,4.1768,
    4.2165,4.1169,4.4406,3.7166,4.2819,3.6317,4.1789,4.3707,4.4039,3.9090,5.8345,
    5.8985,5.5333,6.1905,5.8443,6.2548,5.9095,6.1363,5.6464,6.1243,5.5234,5.5585,
    5.5840,5.2372,4.8715,5.2481,5.6063,4.8724,5.5533,5.1258,
    1.2900,1.1129,1.3845,1.5063,1.0579,1.4099,1.5899,1.8939,1.5669,1.5464,2.8398,
    3.0675,2.6226,2.5526,3.0347,3.1389,3.0805,2.6282,2.6482,2.8601,5.6453,5.6529,
    6.0796,5.6192,6.0094,6.5443,6.1125,6.1944,6.0349,5.7927;
    
    
    int nDimension = 50;
    int nGaussian =10;
    int nDataPerGaussian =1000;
    double noise =10; //variance
    double unitRadius =10;
    
    int64 stamp1 = GetTimeMs64();
    DataGenerator data(nDimension, nGaussian, nDataPerGaussian, pow(noise,0.5), unitRadius);
    
    int64 stamp2 = GetTimeMs64();
    
    SpecGmm test(data.X(), nGaussian);
    int64 stamp3 = GetTimeMs64();
    
    data.evaluate(test.centers());
    int64 stamp4 = GetTimeMs64();
    
    cout << endl;
    cout << "time1=" << (stamp2-stamp1)<< endl;
    cout << "time2=" << (stamp3-stamp2)<< endl;
    cout << "time3=" << (stamp4-stamp3)<< endl;
//    cout << "time4" << (stamp2-stamp1)<< endl;
}

