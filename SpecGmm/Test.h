//
//  Test.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/9.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//

#ifndef __SpecGmm__Test__
#define __SpecGmm__Test__

#include <iostream>
#include <Eigen/Dense>

#endif /* defined(__SpecGmm__Test__) */

#ifndef __SpecGmm__D3Matrix__
#define __SpecGmm__D3Matrix__
#include "D3Matrix.h"
#endif /* defined(__SpecGmm__D3Matrix__) */

//extern const bool DBG = false;

using namespace Eigen;


namespace Util
{

    
}


template <typename Derived>
D3Matrix<Derived>* outer(const MatrixBase<Derived> &A, MatrixBase<Derived> &B) {
    
    MatrixXd m = kroneckerProduct(A,B).eval();
    
    cout << "M row:" << m.rows() <<"  m.cols() " <<m.cols() << endl;
    
    //unsigned long s1 = A.rows(), s2 = A.cols(), s3 = B.rows(), s4 = B.cols();
    unsigned long size[3];
    unsigned long index = 0;
    
    if (A.rows()>1) size[index++] = A.rows();
    if (A.cols()>1) size[index++] = A.cols();
    if (B.rows()>1) size[index++] = B.rows();
    if (B.cols()>1) size[index++] = B.cols();
    
    if (index>3) cout << "outer: size is larger than 3!!" << endl;
    
    if (index<3) {
        cout << "outer: size is smaller than 3!!" << endl;
        size[2] = 1;
    }
    
    D3Matrix<MatrixXd> *ret = new D3Matrix<MatrixXd>(size[0], size[1], size[2]);
    
    long layerSize = size[0] * size[1];
    long colNeed = layerSize/m.rows();
    
    MatrixXd layer0 = m.leftCols(colNeed);
    
    //cout << "before"<<layer0;
    layer0.resize(size[0],size[1]);
    //cout << "after"<<layer0;
    // MatrixXd layer0 = m.block(0, ,layerSize-1);
    ret->setLayer(0, layer0);
    
    if (index == 3) {
        
        int i=1;
        for (unsigned long colInd=colNeed; colInd<m.cols(); colInd+=colNeed) {
            //MatrixXd temp = m.block(elementIndex, elementIndex+layerSize-1);
            
            MatrixXd temp  = m.middleCols(colInd, colNeed);
            temp.resize(size[0],size[1]);
            //cout << "i="<< i <<"  temp "<<temp;
            ret->setLayer(i++, temp);
            //MatrixXd aaa = ret->getLayer(i);
            //cout <<"check"<< aaa<< endl;
        }
    }
    
    
    return ret;
    
    //MatrixXd aaa = ret.getLayer(1);
    //cout <<"shit check"<< aaa<< endl;
    
    //cout <<"OuterProduct"<<endl;
    //ret.print();;
    //m.resize(3,3);
    
    //cout <<"OuterProduct" << m << endl;
    
    //MatrixXf aa;
    //test2(m, aa);
    //cout << "Cool" << aa << endl;
}



class Test {
    
public:
    
    Test() {
        check();
    }
    
    void check();
};

