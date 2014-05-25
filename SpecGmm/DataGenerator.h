//
//  DataGenerator.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/17.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//

#ifndef SpecGmm_DataGenerator_h
#define SpecGmm_DataGenerator_h

#include <random>
#include <vector>
#include <time.h>
#include <Eigen/Dense>

#endif

class DataGenerator {
    
public:
    
    DataGenerator(unsigned long nDimension, unsigned long nGaussian, unsigned long nDataPerGaussian, double noise, double unitRadius)
        : mDimension(nDimension),
            mGaussian(nGaussian),
            mDataPerGaussian(nDataPerGaussian),
            mNoise(noise),
            mUnitRadius(unitRadius){
                initialize();
            };

    void initialize() {

        mCenters = MatrixXd::Random(mDimension, mGaussian);

        for (int i=0; i<mGaussian; i++) {
            mCenters.col(i) = mCenters.col(i)/mCenters.col(i).norm();
            mCenters.col(i) *= mUnitRadius;
        }
        
        mX = MatrixXd::Zero(mDimension, mDataPerGaussian*mGaussian);
        
        random_device rd;
        default_random_engine generator(rd());

        for (unsigned long cluster=0; cluster<mGaussian; cluster++) {
            
            unsigned long margin = cluster * mDataPerGaussian;
            
            VectorXd currentCenter = mCenters.col(cluster);
            vector<normal_distribution<double>> normalVec;
            
            for (unsigned long dimension=0; dimension<mDimension; dimension ++) {
                normalVec.push_back(normal_distribution<double>(currentCenter(dimension), mNoise));
            }
            
            for (unsigned long index=0; index<mDataPerGaussian; index++) {
                for (unsigned long dim=0; dim<mDimension; dim++) {
                    mX(dim, margin+index) = normalVec.at(dim)(generator);
                }
            }
        }
    }
    
    MatrixXd X() {
        return mX;
    }
    
    MatrixXd center() {
        return mCenters;
    }
    
    double evaluate(MatrixXd estimate) {
        
        if (estimate.rows()!=mCenters.rows()) {
            cout << endl <<"ERROR:DataGenerator evaluate"<< endl;
            cout << "resultEvaluation dimension mismatch "<< endl;
            return 0;
        }
        
        //MatrixXd finalEstimate(mDimension, mGaussian);
        unsigned long nEstimate = estimate.cols();
        
        //cout << "finalEstimate row:"<< finalEstimate.rows() << endl;
        //cout << "estimate row:"<< estimate.rows() << endl;
        //cout << "nEstimate:"<< nEstimate <<"    mGaussian:"<<mGaussian << endl;

        
        //finalEstimate.leftCols(nEstimate) = estimate;
        
        MatrixXd finalEstimate = estimate;
        
        /*
        if (nEstimate != mGaussian) {
            unsigned long sizeDiff = mGaussian - nEstimate;
            
            if (sizeDiff>0) {
                finalEstimate.rightCols(sizeDiff) = MatrixXd::Zero(mDimension, sizeDiff);
                cout << "# of decomposed element is less than expected!! expect:" << mGaussian << "  only:" << nEstimate << endl;
                nEstimate = nEstimate + sizeDiff;
            } else {
                cout << endl << "ERROR:DataGenerator evaluate" << endl;
                cout <<"Decompsed center is more than expected. We already pick dominant centers!!" << endl;
                
            }
        }
        */
        
        // Assign all the real center to the nearest estimater center
         
        MatrixXd bestMatch(mDimension, mGaussian);
        
        double error = 0;
        
        for (unsigned long i=0; i<mGaussian; i++) {
            MatrixXd currentRep = mCenters.col(i).replicate(1,nEstimate);
            MatrixXd diff = finalEstimate - currentRep;
            diff = diff.array().pow(2);
            VectorXd dist = diff.colwise().sum();

            MatrixXf::Index minRow, minCol;
            error += pow(dist.minCoeff(&minRow, &minCol),0.5);
            bestMatch.col(i) = finalEstimate.col(minRow);
        }
        
        //cout << endl <<" ===== Best match ===== " << endl;
        //cout << bestMatch << endl;
        //cout << " ===== Original ===== " << endl;
        //cout << mCenters << endl;
        cout << endl << "avg RMSE=" << error/mGaussian << endl;
        
        return (error/mGaussian);
    }
    
    
private:
    unsigned long mDimension;
    unsigned long mGaussian;
    unsigned long mDataPerGaussian;
    double mNoise;
    double mUnitRadius;
    MatrixXd mCenters;
    MatrixXd mX;
    
};
