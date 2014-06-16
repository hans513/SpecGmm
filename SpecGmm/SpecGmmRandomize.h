//
//  SpecGmmRandomize.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/25.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//



#ifndef SpecGmm_SpecGmmRandomize_h
#define SpecGmm_SpecGmmRandomize_h

#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

#endif

#ifndef __SpecGmm__D3Matrix__
#include "D3Matrix.h"
#endif /* defined(__SpecGmm__D3Matrix__) */

#ifndef __SpecGmm__Test__
#include "test.h"
#endif /* defined(__SpecGmm__Test__) */

#ifndef __SpecGmm__TesnsorPower__
#include "TensorPower.h"
#endif /* defined(__SpecGmm__TesnsorPower__) */

#ifndef __SpecGmm__Fastfood__
#include "Fastfood.h"
#endif /* defined(__SpecGmm__Fastfood__) */

using namespace Eigen;


class SpecGmmRandomize {
    
public:
    
    static const bool DBG = true;
    static const bool TIME_MEASURE = true;
    
    void compute(MatrixXd X, unsigned long K){
        mK = K;
        
        int64 t0 = GetTimeMs64();
        
        unsigned long nData = X.cols();
        unsigned long nDimension = X.rows();

        unsigned long p = K + floor(nDimension/10)<100? floor(nDimension/10): 100;
        if (p<nDimension) p = nDimension;
        
        
        if (DBG) cout << "=== SpecGmmRandomize ===" << endl;
        if (DBG) cout << "nData=" << nData << endl;
        if (DBG) cout << "nDimension=" << nDimension << endl;
        if (DBG) cout << "p=" << p << endl;
        
        FastfoodRangeFinder ffFinder(X, p);
        
        // For preventing from storing huge matrix, we calculate svd(B*B')
        MatrixXd B = ffFinder.Q().transpose()* X;
        
        
        //[Ub,D,Vb] = svd(B*B');
        //eigenvalues = diag(D);
        
        //MatrixXd C = X*X.transpose()/nData;
             
        int64 t1 = GetTimeMs64();
        if (TIME_MEASURE) cout << "Time: Matrix Multiplication=" << (t1-t0)<< endl;
        
        const JacobiSVD<MatrixXd> svd(B*B.transpose()/nData, ComputeThinU | ComputeThinV);
        MatrixXd Ub = svd.matrixU();
        MatrixXd U = ffFinder.Q() * Ub;

        VectorXd S = svd.singularValues();
        unsigned long nNoise = S.rows() - K;
        double sigma = S.tail(nNoise).sum()/nNoise;
        VectorXd D = S.array() - sigma;
        // When matrix size change, we need another variable to store the result
        VectorXd validD = D.head(K).array().pow(-0.5);
        MatrixXd W = U.leftCols(K) * validD.asDiagonal();
        
        if (DBG) cout << "Estimate sigma:" << sigma << endl;
        if (DBG) cout << "U:" << endl << U << endl;
        if (DBG) cout << "D:" << endl << validD << endl;
        if (DBG) cout << "S:" << endl << S << endl;
        if (DBG) cout << endl << endl;
        if (DBG) cout << "W" << endl << W;
        
        int64 t2 = GetTimeMs64();
        if (TIME_MEASURE) cout << "Time: SVD decompostion=" << (t2-t1)<< endl;
        
        
        D3Matrix<MatrixXd> EWtX3(K,K,K);
        for (int i=0; i<nData; i++) {
            MatrixXd temp = W.transpose() * X.col(i);
            MatrixXd temp2 = outer(temp,temp)->getLayer(0);
            EWtX3 += *outer(temp2, temp);
        }
        
        int64 t3 = GetTimeMs64();
        if (TIME_MEASURE) cout << "Time: Tensor forming=" << (t3-t2)<< endl;
        
        // EWtX3 = EWtX3/nData;
        for (int i=0; i<EWtX3.layers(); i++) {
            MatrixXd temp = EWtX3.getLayer(i);
            temp = temp.array() / nData;
            EWtX3.setLayer(i, temp);
        }
        
        if (DBG) cout<< endl << endl;
        if (DBG) cout<< "EWtX3" << endl;
        if (DBG) EWtX3.print();
        
        MatrixXd EWtX = (W.transpose()*X).rowwise().sum().array() / nData;
        
        if (DBG) cout<< endl << endl;
        if (DBG) cout<< "EWtX" << endl << EWtX;
        
        D3Matrix<MatrixXd> sigTensor(K,K,K);
        
        for (int i=0; i<nDimension; i++){
            MatrixXd ei = MatrixXd::Zero(nDimension,1);
            ei(i,0) = 1;
            
            MatrixXd WtEi = W.transpose()*ei;
            MatrixXd temp1 = outer(EWtX,  WtEi)->getLayer(0);
            MatrixXd temp2 = outer(WtEi, EWtX)->getLayer(0);
            MatrixXd temp3 = outer(WtEi, WtEi)->getLayer(0);
            sigTensor +=   *outer(temp1, WtEi);
            sigTensor +=   *outer(temp2, WtEi);
            sigTensor +=   *outer(temp3, EWtX);
        }
        
        sigTensor = sigTensor*sigma;
        
        if (DBG) cout<< endl << endl;
        if (DBG) cout<< "sigTensor" << endl;
        if (DBG) sigTensor.print();
        
        D3Matrix<MatrixXd> T = EWtX3 - sigTensor;
        
        if (DBG) cout<< endl << endl;
        if (DBG) T.print();
        
        int64 t4 = GetTimeMs64();
        if (TIME_MEASURE) cout << "Time: Tensor calculation=" << (t4-t3)<< endl;
        
        tensorDecompose(T, W);
        
        int64 t5 = GetTimeMs64();
        if (TIME_MEASURE) cout << "Time: Tensor decompostion=" << (t5-t4)<< endl;
        
    }
    
    SpecGmmRandomize(MatrixXd X, unsigned long K) {
        compute(X, K);
    }
    
    template <typename Derived>
    void tensorDecompose(D3Matrix<Derived> T, MatrixXd W) {
        
        vector<MatrixXd> theta;
        vector<double> lambda;
        vector<MatrixXd> centers;
        
        while (lambda.size()==0  || abs(lambda.back())>0.01 ) {
            TensorPower<MatrixXd> current(T, 10, 10);
            T = *current.deflate();
            theta.push_back(current.theta());
            lambda.push_back(current.lambda());
            //cout << endl <<"theta="<<endl<<current.theta();
            //cout << endl <<"lambda="<<endl<<current.lambda();
            
            // Recover the center
            const JacobiSVD<MatrixXd> pinvSvd(W.transpose(), ComputeThinU | ComputeThinV);
            MatrixXd invTemp;
            pinvSvd.pinv(invTemp);
            MatrixXd result = current.lambda()*invTemp*current.theta();
            centers.push_back(result);
        }
        
        // Print out the result
        MatrixXd centerMatrix(W.rows(), lambda.size());
        cout << endl <<"TensorDecompose result =" << endl;
        cout << "lambda =" << endl << "\t";
        mWeight.clear();
        for (int i=0; i<centers.size(); i++) {
            centerMatrix.col(i) = centers.at(i).col(0);
            cout << pow(lambda.at(i),-2) <<"\t";
            mWeight.push_back(pow(lambda.at(i),-2));
        }
        cout << endl <<"centers ="<<endl<< centerMatrix << endl;
        mCenters = centerMatrix;
    }
    
    MatrixXd centers() {
        
        if (mK >= mCenters.cols()) return mCenters;
        return  mCenters.leftCols(mK);
        
        //cout << endl <<"mK =" << mK << endl;
        //cout << endl <<"mCenters col =" << mCenters.cols() << endl;
        //return mCenters.leftCols(mK);
    }
    
private:
    MatrixXd bestTheta;
    double bestLambda;
    MatrixXd deflate;
    MatrixXd mCenters;
    unsigned long mK;
    vector<double> mWeight;
};