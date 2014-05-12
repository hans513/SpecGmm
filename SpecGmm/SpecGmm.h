//
//  SpecGmm.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/11.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//

#ifndef SpecGmm_SpecGmm_h
#define SpecGmm_SpecGmm_h

#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>


#endif


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

using namespace Eigen;


class SpecGmm {
    
public:
    
    void testOuter() {
        
        MatrixXd t1(3,3);
        MatrixXd t2(1,3);
        
        t1 << 1,2,3,4,5,6,7,8,9;
        t2 << 1,2,3;
        
        outer(t1, t2);
        
        MatrixXd sol0(3,3);
        MatrixXd sol1(3,3);
        MatrixXd sol2(3,3);
        sol0 << 1,4,7,2,8,14,3,12,21;
        sol1 << 2,5,8,4,10,16,6,15,24;
        sol2 << 3,6,9,6,12,18,9,18,27;
        
    }
    
    void compute(MatrixXd X, unsigned long K) {
        
        unsigned long nData = X.cols();
        unsigned long nDimension = X.rows();
        
        cout << "nData=" << nData << endl;
        cout << "nDimension=" << nDimension << endl;
        
        MatrixXd C = X*X.transpose()/nData;
        
        const JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
        MatrixXd U = svd.matrixU().leftCols(K);
        VectorXd S = svd.singularValues();
        
        unsigned long nNoise = S.rows() - K;
        
        //MatrixXd sigmaD = D.tail(nNoise);
        double sigma = S.tail(nNoise).sum()/nNoise;
        VectorXd D = S.array() - sigma;
        D = D.head(K).array().pow(-0.5);
        MatrixXd W = U * D.asDiagonal();
        
        
        cout<< endl << endl;
        cout<< "W" << endl << W;

      
        D3Matrix<MatrixXd> EWtX3(K,K,K);
        for (int i=0; i<nData; i++) {
            
            VectorXd curX = X.col(i);
            MatrixXd temp = W.transpose()*curX;
            MatrixXd temp2 = outer(temp,temp)->getLayer(0);
            //D3Matrix<MatrixXd> temp3 = outer(temp2, temp);
            EWtX3 = EWtX3 + *outer(temp2, temp);
            
            //cout << "temp=" << temp << endl <<" temp2="<< temp2 << endl;
        }
        
        // EWtX3 = EWtX3/nData;
        for (int i=0; i<EWtX3.layers(); i++) {
            MatrixXd temp = EWtX3.getLayer(i);
            temp = temp.array() / nData;
            EWtX3.setLayer(i, temp);
        }
        
        cout<< endl << endl;
        cout<< "EWtX3" << endl;
        EWtX3.print();
        
        // EWtX = sum(W'*X,2)/nData;
        MatrixXd EWtX = (W.transpose()*X).rowwise().sum().array() / nData;
      
        cout<< endl << endl;
        cout<< "EWtX" << endl << EWtX;

        
        D3Matrix<MatrixXd> sigTensor(K,K,K);

        for (int i=0; i<nDimension; i++){
            MatrixXd ei = MatrixXd::Zero(nDimension,1);
            ei(i,0) = 1;
            
            cout<< "ei=" << ei << endl;
            
            MatrixXd WtEi = W.transpose()*ei;
            MatrixXd temp1 = outer(EWtX,  WtEi)->getLayer(0);
            MatrixXd temp2 = outer(WtEi, EWtX)->getLayer(0);
            MatrixXd temp3 = outer(WtEi, WtEi)->getLayer(0);
            sigTensor = sigTensor + *outer(temp1, WtEi);
            sigTensor = sigTensor + *outer(temp2, WtEi);
            sigTensor = sigTensor + *outer(temp3, EWtX);
        }
        
        
        // sigTensor = ESTIMATE_SIGMA*sigTensor;
        /*
        for (int i=0; i<sigTensor.layers(); i++) {
            MatrixXd temp = sigTensor.getLayer(i);
            temp = temp.array() * sigma;
            sigTensor.setLayer(i, temp);
        }*/
        sigTensor = sigTensor*sigma;

        cout<< endl << endl;
        cout<< "sigTensor" << endl;
        sigTensor.print();
        
        D3Matrix<MatrixXd> T = EWtX3 - sigTensor;
        
        cout<< endl << endl;
        
        T.print();
        
        tensorDecompose(T, W);
        
        //MatrixXd
        
        //cout << "U=" << U << endl <<" D="<< D <<"  sigma="<< sigma << endl;
        //cout << "W=" << W;
        
        
    }
    
    SpecGmm(MatrixXd X, unsigned long K) {
        compute(X, K);
    }
    
    template <typename Derived>
    void tensorDecompose(D3Matrix<Derived> T, MatrixXd W) {
        
        cout << endl << endl << endl;
        
        vector<MatrixXd> theta;
        vector<double> lambda;
        
        while (lambda.size()==0  || abs(lambda.back())>0.01 ) {
            TensorPower<MatrixXd> current(T, 10, 10);
            T = *current.deflate();
            theta.push_back(current.theta());
            lambda.push_back(current.lambda());
            
            cout << endl <<"theta="<<endl<<current.theta();
            cout << endl <<"lambda="<<endl<<current.lambda();
            
            //mu(:,i) = lambda(i)*pinv(W')*theta(:,i);
            const JacobiSVD<MatrixXd> pinvSvd(W.transpose(), ComputeThinU | ComputeThinV);
            MatrixXd invTemp;
            pinvSvd.pinv(invTemp);
            MatrixXd result = current.lambda()*invTemp*current.theta();
            cout << endl <<"result="<<endl<<result;
            cout << endl <<"check="<<W.transpose()*invTemp<<result;
            cout << endl <<"lambda back="<<lambda.back();
        }
        
    }


private:
    MatrixXd bestTheta;
    double bestLambda;
    MatrixXd deflate;
};