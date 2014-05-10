//
//  D3Matrix.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/9.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//

#ifndef __SpecGmm__D3Matrix__
#define __SpecGmm__D3Matrix__

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#endif /* defined(__SpecGmm__D3Matrix__) */

using namespace Eigen;
using namespace std;

template<typename Derived> class D3Matrix {
    
public:
    
    D3Matrix(long row, long col, long layer) {

        nRow = row;
        nCol = col;
        nLayer = layer;
        layerPtr = new MatrixXd[nLayer];
        
        for (long i=0; i<nLayer; i++) {
            layerPtr[i] =  MatrixXd::Zero(nRow,nCol);
        }
    }
    
    ~D3Matrix() {
        
    }
    
    MatrixXd getLayer(unsigned layer){
        if (layer>=nLayer) {
            cout << "D3<Matrix getLayer error  layer="<<layer<< "  max layer=" << nLayer << endl;
            MatrixXd zero = MatrixXd::Zero(nRow,nCol);
            return zero;
        }
        return layerPtr[layer];
    };
    
    
    bool setLayer(unsigned layer, MatrixBase<Derived> &matrix){
        if (layer>=nLayer) return false;
        layerPtr[layer] = matrix;
        return true;
    };
    
    long rows() {return nRow;}
    long cols() {return nCol;}
    long layers() {return nLayer;}
    
    void check(){
        cout<<"Good";
    };
    
    void print() {
        for (long i=0; i<nLayer; i++) cout << "D3Matrix layer"<< i << "=>\n"<< layerPtr[i] ;
        cout << endl;
    }

private:
    
    // Array of MatrixXd
    MatrixXd *layerPtr;
    long nRow;
    long nCol;
    long nLayer;
   
};