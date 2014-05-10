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

#include "D3Matrix.h"
#include "test.h"
//#include <Eigen/CXX11/Tensor>

using Eigen::MatrixXd;
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

static const bool DBG = false;

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
        testTFuction();
    
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


template <typename Derived>
void outer(const MatrixBase<Derived> &a, MatrixBase<Derived> &b) {
    
    MatrixXd m = kroneckerProduct(a,b).eval();
    
    cout <<"OuterProduct" << m << " size " << m.size() << endl;
    m.resize(3,3);
    
    cout <<"OuterProduct" << m << endl;
    
    MatrixXf aa;
    //test2(m, aa);
    //cout << "Cool" << aa << endl;
}



/*
 function [theta, lambda, deflate]=tensorPower(T,L,N)

*/



template <typename Derived>
MatrixXd tFunction(D3Matrix<Derived> &T, MatrixBase<Derived> &u, MatrixBase<Derived> &v, MatrixBase<Derived> &w) {
    
    // the length of the tensor
    const long dim = T.rows();
    
    if (u.rows()!= dim || v.rows()!=dim || w.rows()!=dim) {
        cout << "ERROR: tFunction: the dimension of input matrices are not correspondent to the Tensor" << endl;
        MatrixXd zero = MatrixXd::Zero(1,1);
        return zero;
    }
    
    const long s1 = u.cols(), s2 = v.cols(),  s3 = w.cols();
    
    D3Matrix<Derived> ret = D3Matrix<Derived>(s1, s2, s3);
    D3Matrix<Derived> temp = D3Matrix<Derived>(s1, s2, dim);
    
    /**
     % Treat the 3D tesor as lots of layer of 2D Matrix
     % In 2 dimension space => A(V1,V2) = V1'*A*V2
     % So it's just like deal with T(u,v) at layer j3
     */
    for (int j3=0; j3<dim; j3++) {

        MatrixXd a = u.transpose() * T.getLayer(j3) * v;
        temp.setLayer(j3,a);

        // For each column vectoer in w (usually there is only one cloumn)
        for (int wInd=0; wInd<s3; wInd++) {

            /**
             % A(V1,V2) at layer j3 * w(j3)
             % and the sum of every layer is the answer
             */
            MatrixXd b = ret.getLayer(wInd) + temp.getLayer(j3) * w(j3,wInd);
            ret.setLayer(wInd, b);
        }
    }

    // Debug
    if (DBG) for (int i=0; i<ret.layers(); i++) {
        cout << "RESULT layer "<< i <<" \n" <<ret.getLayer(i);
    }
    
    if (ret.layers() != 1) {
        cout << "tFunction Error!!  We can't handle returning 3D matrix yet" << endl;
        cout << "In this algorithm, we should return only 1 layer!!" << endl;
    }

    return ret.getLayer(0);

}

void testTFuction() {
    
    MatrixXd u = MatrixXd::Identity(3,3);
    MatrixXd v = MatrixXd::Identity(3,3);
    MatrixXd w(3,1);
    w << 1,1,1;
    
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

    MatrixXd result = tFunction(T,u,v,w);
    
    MatrixXd solution(3,3);
    solution << 30,33,36,39,42,45,48,51,54;
    
    cout << "===== TEST CASE RESULT : TFunction ===== " << endl;
    cout << "Test error should be a zero matrix" << endl;
    cout << "test error = " << endl << solution-result << endl ;
}


template <typename Derived> class TensorPower {
    
    public:
        TensorPower(MatrixBase<Derived> &T, int L, int N) {


        /*vvvvvvvv
         if size(T,1)~=size(T,2) || size(T,1)~=size(T,3)
         disp('ERROR')
         return;
         end
         */
            if (T.rows()!=T.cols() /* || T.row()!=  */) {
                return;
            }
            
           
         //k = size(T,1); vvvvvvvv
            const unsigned int k = T.row();
            
            
            
            /*
         theta = zeros(k, L, N+1);
         for tau=1:L
         
         theta_0 = randn(k,1);
         theta(:, tau, 1) = theta_0/norm(theta_0);
         */
            
            
            
            /*
         for t=2:N+1
         preTheta = theta(:, tau, t-1);
         tensorTheta = TFunction(T, eye(k), preTheta, preTheta);
         theta(:, tau, t) = tensorTheta / norm(tensorTheta);
         end
        
             
             
             
             end
         
         
         for tau=1:L
         
         cur = theta(:, tau, N+1);
         lambdaBuf(tau) = TFunction(T, cur, cur, cur);
         
         end
         
         [~, maxTau] = max(lambdaBuf);
         best = theta(:, maxTau, N+1);
         
         for i=2:N+1
         best(:,1) = TFunction(T, eye(k), best, best) / norm(TFunction(T, eye(k), best, best));
         end
         
         lambda = TFunction(T, best, best, best);
         
         deflate = T - lambda * outerProduct(outerProduct(best, best),best);
         theta = best;
             
        */
            
        }
    
    protected:
        MatrixBase<Derived> theta;
        float* lambda;
        MatrixBase<Derived> deflate;
};


