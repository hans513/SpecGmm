//
//  main.cpp
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/4.
//  Copyright (c) 2014 Huang, Tse-Han. All rights reserved.
//

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/KroneckerProduct>
#include <Eigen/Core>

#ifndef __SpecGmm__D3Matrix__
#define __SpecGmm__D3Matrix__
#include "D3Matrix.h"
#endif /* defined(__SpecGmm__D3Matrix__) */

#ifndef __SpecGmm__Test__
#define __SpecGmm__Test__
#include "test.h"
#endif /* defined(__SpecGmm__Test__) */

#ifndef __SpecGmm__TesnsorPower__
#define __SpecGmm__TesnsorPower__
#include "TensorPower.h"
#endif /* defined(__SpecGmm__TesnsorPower__) */

#ifndef __SpecGmm__SpecGmm__
#define __SpecGmm__SpecGmm__
#include "SpecGmm.h"
#endif /* defined(__SpecGmm__SpecGmm__) */

using namespace std;
using namespace Eigen;

MatrixXd test(MatrixXd a, MatrixXd b);

template <typename Derived>
void test2(const MatrixBase<Derived> &input, MatrixBase<Derived> &output);

template <typename Derived>
void outer(const MatrixBase<Derived> &a, MatrixBase<Derived> &b);

template <typename Derived>
void tFunction(const MatrixBase<Derived> &T, MatrixBase<Derived> &u, MatrixBase<Derived> &v, MatrixBase<Derived> &w);

void testTFuction();
void testTensorPower();
void testOuter();
void testSpecGmm();

//const bool DBG = 0;

int main(int argc, const char * argv[]) {

    // insert code here...
    //MatrixXf m(2,2);
    //m(0,0) = 3;
    //m(1,0) = 2.5;
    //m(0,1) = -1;
    //m(1,1) = m(1,0) + m(0,1);
    
    Eigen::MatrixXf a(3,1);
    a << 1, 2, 3;
    
    Eigen::MatrixXf b(3,1);
    b<< 4, 5, 6;
  
    //Eigen::Tensor<double, 4> t(10, 10, 10, 10);
    //t(0, 1, 2, 3) = 42.0;
    
    //test2(a, b);
    //TensorPower::testTFuction();
    //testTensorPower();
    //testOuter();
    testSpecGmm();
    //Test t;
    
    
    return 0;
}

MatrixXd test(MatrixXd a, MatrixXd b) {
    return a+b;
}

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
    MatrixXd X(4,20);
    // Center =
    //5.5599    7.0493
    //7.4210    2.9892
    //0.7244    4.7072
    //3.6730    4.3834
    X << 5.7896,5.6359,5.5301,5.5245,5.0695,5.8445,5.2919,5.6945,5.9686,5.2731
        ,6.8700,6.7659,7.1758,6.3803,7.0528,7.3412,7.0229,7.4555,6.9957,7.3049,
        7.1263,7.9713,7.1027,6.5886,7.5687,7.3943,7.7713,7.2862,7.5625,7.1713
        ,2.7254,2.9215,2.5417,2.9199,2.7602,3.2831,2.6862,2.9441,2.7830,2.9657,
        0.5780,0.8247,0.6257,1.1336,0.6533,0.6168,1.1998,0.4550,0.5137,0.3889
        ,4.9361,4.3415,5.3407,4.4333,5.0126,4.9296,4.7340,4.4581,4.5895,5.0449,
        3.3472,3.7278,4.0639,3.7934,3.4680,3.3921,3.7881,4.1433,3.3441,3.9458
        ,4.4003,4.0163,4.4981,3.9663,4.5073,3.7697,3.9186,4.2988,3.9143,4.7658;
    

    SpecGmm test(X, 2);
    
}

/*
 function [theta, lambda, deflate]=tensorPower(T,L,N)

*/

