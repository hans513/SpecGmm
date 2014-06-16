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
//#include <Eigen/KroneckerProduct>
//#include <spiral_wht.h>

#ifndef __SpecGmm__D3Matrix__
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

#ifndef __SpecGmm__Fastfood__
#include "Fastfood.h"
#endif /* defined(__SpecGmm__Fastfood__) */

#ifndef SpecGmm_SpecGmmRandomize_h
#include "SpecGmmRandomize.h"
#endif

#ifndef SpecGmm_GaussianData_h
#include "GaussianData.h"
#endif

using namespace std;
using namespace Eigen;

template <typename Derived>
void svdExample(const MatrixBase<Derived> &input, MatrixBase<Derived> &output);

void testTFuction();
void testTensorPower();
void testSpecGmm();
void specGmmExp();

void test() {
    
}


int main(int argc, const char * argv[]) {

    //TensorPower::testTFuction();
    //testTensorPower();
    //testOuter();

    //test();
    //Fastfood::test();
    
    testSpecGmm();
    //specGmmExp();
    //testHadamardGen();            // in Fastfood.h
    //test_FastfoodRangeFinder();   // in Fastfood.h
    
    return 0;
}

// SVD Example
template <typename Derived>
void svdExample(const MatrixBase<Derived> &input, MatrixBase<Derived> &output) {
    
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

double evaluate(MatrixXd centers, MatrixXd estimate) {
    
    if (estimate.rows()!=centers.rows()) {
        cout << endl <<"ERROR:DataGenerator evaluate"<< endl;
        cout << "resultEvaluation dimension mismatch "<< endl;
        return 0;
    }
    
    unsigned long dimension = centers.rows();
    unsigned long gaussian = centers.cols();
    unsigned long nEstimate = estimate.cols();
    
    //cout << "finalEstimate row:"<< finalEstimate.rows() << endl;
    //cout << "estimate row:"<< estimate.rows() << endl;
    //cout << "nEstimate:"<< nEstimate <<"    mGaussian:"<<mGaussian << endl;
    //MatrixXd finalEstimate(mDimension, mGaussian);
    //finalEstimate.leftCols(nEstimate) = estimate;
    
    MatrixXd finalEstimate = estimate;
    
    // Assign all the real center to the nearest estimater center
    
    MatrixXd bestMatch(dimension, gaussian);
    
    double error = 0;
    
    for (unsigned long i=0; i<gaussian; i++) {
        MatrixXd currentRep = centers.col(i).replicate(1,nEstimate);
        MatrixXd diff = finalEstimate - currentRep;
        diff = diff.array().pow(2);
        VectorXd dist = diff.colwise().sum();
        
        MatrixXf::Index minRow, minCol;
        error += pow(dist.minCoeff(&minRow, &minCol),0.5);
        bestMatch.col(i) = finalEstimate.col(minRow);
    }
    
    cout << endl <<" ===== Best match ===== " << endl;
    cout << bestMatch << endl;
    cout << " ===== Original ===== " << endl;
    cout << centers << endl;
    cout << endl << "avg RMSE=" << error/gaussian << endl;
    
    return (error/gaussian);
}



void testSpecGmm() {

    const bool RANDOM = true;
    
    int nDimension = 50;
    int nGaussian = 10;
    int nDataPerGaussian = 1000;
    double noise = 1; //variance
    double unitRadius =10;
    
    int64 stamp1 = GetTimeMs64();
    DataGenerator data(nDimension, nGaussian, nDataPerGaussian, pow(noise,0.5), unitRadius);
        
    if(RANDOM) {
        //SpecGmm test(data.X(), nGaussian);
        SpecGmmRandomize test(data.X(), nGaussian);
        data.evaluate(test.centers());
    }
    /*
    else {
        GaussianData data;
        SpecGmm test(data.Data5_90, nGaussian);
        evaluate(data.Data5_90_center, test.centers());
    }
     */

    int64 stamp4 = GetTimeMs64();
    cout << endl;
    cout << "Total time=" << (stamp4-stamp1)<< endl;
    
}

void specGmmExp() {
    
    int nDimension = 50;
    int nGaussian = 10;
    int nDataPerGaussian = 1000;

    double unitRadius =10;
    
    int nRepeat = 30;
    double rmse = 0;
    int64 time = 0;
    
    //int nNoise = 6;
    int noiseList[] = {0,3,6,9,12,15,18,21,24,27,30,-1};
    
    int nNoise = 0;
    while (noiseList[nNoise] != -1) nNoise++;
    
    double* resultArray = new double[nNoise];
    int64* timeArray = new int64[nNoise];
    
    for (int expIndex=0; expIndex<nNoise; expIndex++) {
        
        double noise = noiseList[expIndex];
    
        for (int repIndex=0; repIndex<nRepeat; repIndex++) {
            
            cout << "expIndex:" << expIndex <<" repIndex:" << repIndex << endl;
        
            int64 stamp1 = GetTimeMs64();
            DataGenerator data(nDimension, nGaussian, nDataPerGaussian, pow(noise,0.5), unitRadius);
        
            SpecGmm test(data.X(), nGaussian);
            rmse += data.evaluate(test.centers());
        
            int64 stamp4 = GetTimeMs64();
            cout << endl;
            cout << "Total time=" << (stamp4-stamp1)<< endl;
            time += (stamp4-stamp1);
        }
        
        resultArray[expIndex] = rmse/nRepeat;
        timeArray[expIndex] = time/nRepeat;
        
        rmse=0;
        time=0;
    }
    
    cout << endl<<"AVG RMSE"<<endl;
    for (int i=0; i<nNoise; i++) {
        cout << resultArray[i] << "\t";
    }
    
    cout << endl<< endl <<"AVG TIME"<<endl;
    
    for (int i=0; i<nNoise; i++) {
        cout << timeArray[i] << "\t";
    }
}
