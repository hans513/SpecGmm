//
//  Fastfood.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/18.
//  Copyright (c) 2014 Huang, Tse-Han. All rights reserved.
//

#ifndef __SpecGmm__Fastfood__
#define __SpecGmm__Fastfood__

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>
#include <Eigen/Dense>

#endif /* defined(__SpecGmm__Fastfood__) */

#ifndef __SpecGmm__Test__
#include "Test.h"
#endif /* defined(__SpecGmm__Test__) */

void testHadamardGen();
void test_FastfoodRangeFinder();

class Fastfood {
public:
    
    // Usually m >> n
    Fastfood(unsigned long m, unsigned long n): mTargetRows(m), mTargetCols(n) {
        assert (m>n);
        mRealCols = pow2roundup(n);
        mRealRows = mRealCols > m? mRealCols: m;
        mBlkLength = mRealCols;
        
        mNumBlk = mRealRows / mRealCols;
        
        mResidualRows = (unsigned)(mRealRows % mRealCols);
        mResidualCols = (unsigned)mBlkLength;
        
        initialize();
    }
    
    Fastfood() {
        
    }
    
    void initialize();
    MatrixXd multiply(MatrixXd input);
    MatrixXd multiply2(MatrixXd input);
    
private:
    // The target dimension
    unsigned long mTargetRows;
    unsigned long mTargetCols;

    // The real dimension after making the length to be the power of 2
    unsigned long mRealRows;
    unsigned long mRealCols;
    
    // The residual matrix dimension
    unsigned mResidualRows;
    unsigned mResidualCols;

    // The length of each fastfood block block
    unsigned long mBlkLength;
    unsigned long mNumBlk;
    
    vector<VectorXi> mB;
    vector<VectorXi> mPI;
    vector<VectorXd> mG;
    MatrixXd mResidual;
};


class FastfoodRangeFinder {
    
public:
    
    FastfoodRangeFinder(MatrixXd X, long target) {

        mFf =  Fastfood(X.cols(), target);

        MatrixXd Y = mFf.multiply(X);
        //MatrixXd Z = mFf.multiply2(X);
        //MatrixXd diff = Y-Z;
        //cout << endl << "multiply dif" << endl<<diff.sum()<<endl;
        
        //cout << endl << "X after random ff" << endl<<Y<<endl;
        //MatrixXd Y;
        //mFf.multiply(X, Y);
        
        HouseholderQR<MatrixXd> qr(Y);
        mQ = MatrixXd::Identity(X.rows(),target);
        mQ = qr.householderQ() * mQ;
        
        //cout << endl << "Q:" << endl<< mQ <<endl;
    }

    MatrixXd Q() {
        return mQ;
    }
    
private:
    Fastfood mFf;
    MatrixXd mQ;
    //MatrixXd mR;
};



