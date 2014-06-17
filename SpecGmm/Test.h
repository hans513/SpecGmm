//
//  Test.h
//  SpecGmm
//
//  Created by Huang, Tse-Han on 2014/5/9.
//  Copyright (c) 2014年 Huang, Tse-Han. All rights reserved.
//



#ifndef __SpecGmm__Test__
#define __SpecGmm__Test__

#include <sys/time.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/KroneckerProduct>

//#include <ctime>
//#include <stdint.h>
#include <inttypes.h>
#endif /* defined(__SpecGmm__Test__) */

#ifndef __SpecGmm__D3Matrix__
#include "D3Matrix.h"
#endif /* defined(__SpecGmm__D3Matrix__) */


typedef int64_t      int64;
typedef uint64_t     uint64;

//extern const bool DBG = false;

using namespace Eigen;

template <typename Derived>
static D3Matrix<Derived> outer(const MatrixBase<Derived> &A, MatrixBase<Derived> &B) {
    
    MatrixXd m = kroneckerProduct(A,B).eval();
    
    //cout << "M row:" << m.rows() <<"  m.cols() " <<m.cols() << endl;
    
    unsigned long size[3];
    unsigned long index = 0;
    
    if (A.rows()>1) size[index++] = A.rows();
    if (A.cols()>1) size[index++] = A.cols();
    if (B.rows()>1) size[index++] = B.rows();
    if (B.cols()>1) size[index++] = B.cols();
    
    if (index>3) cout << "outer: size is larger than 3!!" << endl;
    else if (index<3) {
        //cout << "outer: size is smaller than 3!!" << endl;
        size[2] = 1;
    }
    
    D3Matrix<MatrixXd> ret(size[0], size[1], size[2]);
    
    long layerSize = size[0] * size[1];
    long colNeed = layerSize/m.rows();
    
    MatrixXd layer0 = m.leftCols(colNeed);
    layer0.resize(size[0],size[1]);
    ret.setLayer(0, layer0);
    
    if (index < 3)  return ret;
        
    int i=1;
    for (unsigned long colInd=colNeed; colInd<m.cols(); colInd+=colNeed) {
        MatrixXd temp  = m.middleCols(colInd, colNeed);
        temp.resize(size[0],size[1]);
        ret.setLayer(i++, temp);
    }

    return ret;
    
}

static void testOuter() {
    
    MatrixXd t1(3,3);
    MatrixXd t2(1,3);
    
    t1 << 1,2,3,4,5,6,7,8,9;
    t2 << 1,2,3;
    
    D3Matrix<MatrixXd> result = outer(t1, t2);
    
    MatrixXd sol0(3,3);
    MatrixXd sol1(3,3);
    MatrixXd sol2(3,3);
    sol0 << 1,4,7,2,8,14,3,12,21;
    sol1 << 2,5,8,4,10,16,6,15,24;
    sol2 << 3,6,9,6,12,18,9,18,27;
    D3Matrix<MatrixXd> solution(3,3,3);
    solution.setLayer(0, sol0);
    solution.setLayer(1, sol1);
    solution.setLayer(2, sol2);
    
    result = result - solution;
    result.print();
}


static int64 GetTimeMs64() {
#ifdef WIN32
    /* Windows */
    FILETIME ft;
    LARGE_INTEGER li;
    
    /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
     * to a LARGE_INTEGER structure. */
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    
    uint64 ret = li.QuadPart;
    ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
    ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */
    
    return ret;
#else
    /* Linux */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64 ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
    ret /= 1000;
    
    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000);
    
    return ret;
#endif
}

static inline long
pow2roundup (long x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

/*
class Test {
    
public:
    
    Test() {
        check();
    }
    
    void check();
};

*/