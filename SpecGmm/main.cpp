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
#include <Eigen/KroneckerProduct>
#include <spiral_wht.h>

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

using namespace std;
using namespace Eigen;

template <typename Derived>
void svdExample(const MatrixBase<Derived> &input, MatrixBase<Derived> &output);

void testTFuction();
void testTensorPower();
void testSpecGmm();
void testRandom();
void specGmmExp();

void test() {
    
}

//  Test case for FastfoodRangeFinder
void test_FastfoodRangeFinder() {
    unsigned ndimension = 18;
    unsigned nbasis = 6;
    unsigned ndataPoints = 40;

    // Random basis
    MatrixXd basis = MatrixXd::Random(ndimension, nbasis);

    // nDimension * nDataPoints (rank=nBasis)
    MatrixXd span = basis * MatrixXd::Random(nbasis, ndataPoints);
    
    FastfoodRangeFinder ffFinder(span, nbasis);
    
    unsigned rankDiff = 0;
    
    // Check if all the column vectors in the matrix q we got are linear dependent
    // with the Basis we have above.
    for (int i=0; i<nbasis; i++) {
        MatrixXd test(ndimension, nbasis+1);
        test << ffFinder.Q(), basis.col(i);
        
        FullPivLU<MatrixXd> lu(test);
        lu.setThreshold(pow(10,-10));
        rankDiff += (lu.rank()-nbasis);
            cout << endl << "threshold:" << lu.threshold() << endl;
    }
    
    cout << endl << "Rank diff:" << rankDiff << endl;
}

int main(int argc, const char * argv[]) {

    //TensorPower::testTFuction();
    //testTensorPower();
    //testOuter();
  
    //testRandom();
 
    //test();
   // Fastfood::test();
    
    
  //testSpecGmm();
    //specGmmExp();
    //testHadamardGen();
    test_FastfoodRangeFinder();
    
    return 0;
}

void testRandom() {
    // mean std
    
    /*
    normal_distribution<double> normal(3.0, 4.0);
    default_random_engine generator;
    generator.seed(1000);
    cout << normal(generator) << endl;
     cout << normal(generator) << endl;
     cout << normal(generator) << endl;
     */
    //DataGenerator test(5,3,10,0.1,10);
    //cout << test.X();
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

    /**
    Center =
    6.5445    3.1597    1.8806
    4.4010    6.9023    3.2828
    4.3265    0.2085    4.7405
    4.0927    5.8968    5.3529
    1.5270    2.7491    5.8788
    */
    MatrixXd Y(5,30);
    Y << 6.2927,6.5660,6.2210,6.4903,6.6331,6.5048,6.4256,6.2634,6.5338,6.8035,
    3.4894,3.3732,3.1784,3.2992,2.9455,3.4549,2.5877,3.2452,3.3003,2.9435,2.5123,
    1.6103,1.7337,1.7756,1.3176,2.2257,1.4583,1.9476,1.5875,1.7245,
    4.9342,4.0187,4.1915,4.5295,4.3297,4.2107,4.7183,4.7614,4.4910,4.3328,7.1378,
    7.4547,6.8908,6.6993,7.2232,7.5199,6.6356,6.6785,6.9969,6.8513,3.6953,3.1425,
    3.7438,3.4817,3.3844,3.6345,3.6526,3.5146,3.7795,2.9550,
    5.0358,4.0134,4.5307,4.0849,4.1098,4.3846,4.2170,4.5567,4.1470,4.2717,0.4247,
    0.7256,0.0022,0.6104,0.9517,0.3202,0.3222,0.0822,0.0413,0.5754,5.1657,5.2033,
    4.4487,4.8632,4.2583,3.8923,4.7593,4.9443,4.6721,4.1768,
    4.2165,4.1169,4.4406,3.7166,4.2819,3.6317,4.1789,4.3707,4.4039,3.9090,5.8345,
    5.8985,5.5333,6.1905,5.8443,6.2548,5.9095,6.1363,5.6464,6.1243,5.5234,5.5585,
    5.5840,5.2372,4.8715,5.2481,5.6063,4.8724,5.5533,5.1258,
    1.2900,1.1129,1.3845,1.5063,1.0579,1.4099,1.5899,1.8939,1.5669,1.5464,2.8398,
    3.0675,2.6226,2.5526,3.0347,3.1389,3.0805,2.6282,2.6482,2.8601,5.6453,5.6529,
    6.0796,5.6192,6.0094,6.5443,6.1125,6.1944,6.0349,5.7927;
    
    
    /**
     Center =
     2.4863    6.2180    2.1969
     1.7690    5.8282    5.6722
     4.9041    4.1286    6.0876
     5.4316    1.9568    4.2006
     6.0940    2.5484    2.8803
     */
    MatrixXd Z(5,90);
    Z << 0.8674,1.3265,2.7573,1.8804,0.1416,5.0654,-0.9336,1.9833,2.1388,3.5415,2.8834,1.4108, 2.4746,5.7950,2.6299,4.8442,2.7358,3.2088, 3.0487,1.6040,3.3663,1.6295,3.3148,5.8509, 2.3324,1.7696,4.3852,5.7039,4.4684,0.4518, 6.7290,6.3753,6.2034,6.2761,5.2560,8.4936, 8.7039,7.1524,3.6254,4.4255,5.8222,5.9668, 5.2811,5.6889,2.5032,7.2631,5.4315,7.2846, 6.9495,10.4626,8.1219,6.3865,7.5093,4.5076, 6.7120,5.1526,8.0921,4.8699,6.2209,6.9724, 2.6567,0.7782,4.2492,2.3070,2.5091,3.2530, 4.5174,5.0908,2.1737,2.2160,-0.9518,0.0011, 0.1886,2.1114,1.8652,2.7187,0.8387,1.6907, 1.7972,3.4776,1.0658,2.1931,1.9572,2.8176, 3.0017,-0.3386,5.8083,0.5801,1.5435,-0.3316
    ,3.7869,0.4281,0.0941,0.2596,5.1497,1.1787, 2.4617,2.2827,2.0119,2.7599,-1.1771,2.6021, 3.5661,1.1500,0.6658,-1.7396,4.9219,-2.7033, 4.3583,0.8210,0.0769,1.5264,2.0149,1.5557,-0.6497,2.1971,6.1008,2.0419,-0.6950,-1.0692, 5.5787,4.8727,2.2669,9.6592,5.2756,5.9687, 5.6432,3.7043,5.8412,7.8183,5.2391,9.2009, 9.1646,8.6267,7.7072,6.5966,4.2017,6.0265, 5.2819,7.0444,6.8566,4.0184,7.6117,7.0830, 7.8648,7.1234,3.1546,6.2112,6.6532,6.4569, 7.0530,5.4790,3.7434,4.5298,7.4962,7.1760, 4.7315,6.2212,8.0595,4.8932,5.2278,4.8477, 7.2287,4.4561,5.6136,5.5467,7.3468,4.9572, 2.4988,5.5357,5.5151,7.0388,6.2656,3.1066, 2.2896,7.4490,7.2798,5.1406,2.1287,5.0735
    ,5.9584,2.5061,5.0588,4.3784,4.3595,3.2595, 3.6853,5.7985,3.3650,6.2472,0.0134,4.8678, 4.1502,2.1160,5.6568,3.8050,3.8212,2.7667, 3.5174,4.2449,3.7845,2.2708,4.7782,5.6177, 5.1484,4.5096,1.3582,3.7697,5.4608,3.8032, 5.8742,6.6069,5.4648,3.1647,1.6561,2.0778, 2.8715,6.6692,0.9213,4.1949,1.8993,4.5592, 7.5176,3.6864,6.5500,7.2938,1.8686,4.5202, 2.3497,6.2354,7.9058,2.3785,4.2852,4.6505, 6.2873,1.4118,5.8467,1.7188,1.1743,2.8910, 5.8439,5.2356,9.4270,8.1141,5.9125,7.8128, 7.2812,6.6450,6.7061,6.4103,6.7668,7.6143, 2.0983,4.3899,1.8951,5.5926,4.0057,4.8710, 6.0944,7.2125,5.0500,3.2057,7.3222,7.5892, 3.9840,7.7902,7.9058,7.0068,6.1420,6.9266
    ,2.9627,5.5019,4.8678,7.5776,4.2191,4.1380, 3.1627,3.8012,6.3305,4.1032,4.6777,3.5000, 6.5682,6.8175,8.3662,6.7933,4.0573,5.1969, 5.6746,4.7763,5.3734,2.7173,7.6424,6.8135, 3.9888,5.0044,3.5623,7.7700,7.9516,2.2895, 2.2651,3.7750,4.4438,2.4285,3.1771,2.6735, 4.3682,1.6695,0.9419,0.7426,2.7966,1.8937, 2.8145,3.4756,2.8818,4.0158,0.4786,4.4282, 2.0236,3.7717,3.4440,1.2395,2.7010,0.9922, 2.3195,1.0685,-2.0611,2.8641,1.5591,-0.3799, 0.0099,4.6288,3.7777,3.8002,5.7395,2.3739, 5.7488,4.3108,2.9873,2.2532,0.8961,4.8054, 4.2944,4.7978,3.2394,3.0711,3.1788,5.2472, 2.8848,2.6734,2.5026,6.3850,4.6945,4.3869, 2.0943,1.3050,4.5281,1.6026,3.6803,7.1733
    ,6.6994,5.8203,10.8350,7.3898,4.4933,1.7429, 4.2019,3.6890,2.6874,7.4907,4.2090,7.9365, 5.0196,9.4705,7.5223,9.3288,7.1832,7.9104, 6.9136,3.4898,6.1867,2.8427,8.5079,10.1313, 7.1967,7.9476,6.2550,6.0038,4.5399,4.4775, 4.1293,3.4803,2.4686,2.7064,-0.5956,-0.0956, 3.1737,3.3036,3.5774,-2.0155,4.6489,2.4873, 2.8545,2.4728,4.2406,3.8845,0.8488,-0.9433, 2.0372,-0.9046,1.9371,2.0406,4.6889,0.6074, 3.8187,2.2461,2.3098,0.8640,0.8798,0.1105, 1.7116,3.3917,3.1669,2.2752,-0.2616,2.0805, 0.2595,0.7605,5.4995,-1.8328,2.1967,3.2718, 5.1594,2.1456,3.8647,2.5821,3.2940,-1.0121, 2.3603,1.6294,2.9978,2.9892,0.7495,1.5070, 2.9960,2.4337,3.8706,0.3832,2.4781,2.6003;
    
    // 5,3,30,10
    MatrixXd Zcenter(5,3);
    Zcenter << 2.4863,    6.2180,    2.1969,
    1.7690 , 5.8282 ,   5.6722,
    4.9041 , 4.1286 ,   6.0876,
    5.4316 , 1.9568 ,   4.2006,
    6.0940 , 2.5484 ,   2.8803;
    

    /**
     Center =
     6.3115    0.6983    3.8072    5.6404    4.8154
     1.9318    2.6395    4.9462    4.5021    4.7634
     2.7725    4.9458    4.2697    2.9812    0.5378
     2.9138    0.1446    4.8906    2.8154    4.1588
     0.1293    3.3597    1.0076    3.0995    3.1845
     1.2515    0.6013    2.1031    1.8841    4.5594
     6.2188    7.5109    3.6680    4.2362    2.3682
     */
    MatrixXd A(7,50);
    A << 6.7071,5.2372,6.9060,8.1990,5.7915,6.9530,6.0302,4.2524,6.6187,7.1249,0.1486,0.9322,0.1788,0.3763,0.7571,1.4757,-0.1896,0.1744,-1.1565,0.8835,2.7149,5.1368,4.5879,4.4529,5.1372,2.2161,4.3714,3.7959,4.3326,3.2298,4.4755,6.8827,4.1942,8.2695,3.8779,6.5378,4.3899,5.7586,6.0357,5.7004,5.8383,4.1918,6.0953,6.0157,3.0583,3.6718,5.7914,5.3302,3.4817,6.3130
    ,3.3010,1.9128,1.7471,3.5845,0.1955,2.7882,1.3874,1.0839,1.5471,2.1080,4.1701,1.8600,2.4635,2.1273,2.8388,3.0237,4.3446,1.0886,3.2099,1.8003,4.3530,6.2215,5.7800,3.4351,6.4975,6.5207,3.9999,3.9956,4.9655,4.5141,3.5142,4.5701,4.4332,4.3334,2.3045,4.7325,6.3346,3.7791,4.3227,6.0338,4.7648,3.6199,6.6328,6.3494,5.9924,5.4891,7.0429,5.0272,5.7299,4.3670
    ,1.8051,2.8143,4.9554,0.6784,1.8496,2.9558,2.6093,2.5431,4.2361,3.1755,4.0807,5.0188,3.7825,5.1335,5.0611,4.5051,4.9178,5.2883,4.2696,4.8737,3.6830,2.0320,5.7021,6.9013,4.0492,3.9130,4.9611,5.4291,4.7949,4.5437,3.4397,1.5918,4.4044,2.4176,2.7824,4.0732,1.8012,3.7591,4.1614,4.0272,0.3841,0.6333,1.3263,-0.1229,1.5404,0.9020,-0.7376,-0.5187,0.8550,1.7582
    ,3.7068,4.2939,1.3108,3.0097,2.0622,3.4991,3.0999,3.8082,3.0211,2.9303,1.2873,0.2165,-0.4467,-0.5784,0.3829,0.0867,0.1748,0.8062,2.1948,-0.0575,5.7562,4.8944,5.5363,5.3100,4.4746,4.4996,6.0127,4.6849,6.1644,4.2584,3.8710,2.1404,2.4433,1.4741,2.8915,2.7639,2.5861,2.9263,1.3454,1.5057,2.6968,3.8276,5.4419,5.2403,3.5775,3.9014,4.5192,1.7423,3.3543,3.1858
    ,-0.7954,1.4547,-0.2379,1.3764,1.1502,0.1637,0.5178,0.1947,0.8514,0.7759,4.3269,3.7841,4.6551,3.1052,3.8331,3.4521,4.2933,3.6173,3.5534,4.6249,0.1005,1.8826,-0.4608,0.7877,2.5402,-1.5410,0.9429,0.5832,1.1070,3.8195,4.1715,1.7025,3.8440,1.0325,3.0500,2.8488,3.6743,4.1301,3.2344,3.7584,2.6397,3.1658,3.3674,4.9120,1.7080,2.4960,3.7170,3.3527,2.3924,4.3498
    ,1.9427,-0.3112,0.9801,2.1882,1.5892,1.6170,-0.0737,2.5911,1.6090,2.2969,1.9269,1.0192,-0.2818,0.1532,0.3198,1.0703,0.5236,0.4124,0.0427,0.3898,0.7925,2.6756,3.2017,2.0602,2.7069,2.4122,0.6940,4.7353,0.5204,-0.5605,1.9150,3.2002,1.0639,3.2297,1.0207,2.4877,2.1964,1.0300,3.6266,1.0282,4.5030,4.8604,5.2330,4.9981,4.5336,3.3498,5.3542,4.3147,4.0125,5.0455
    ,5.0655,8.8436,4.8227,5.9840,5.6462,6.7236,6.2509,6.4925,6.1715,6.8198,8.5190,6.8388,6.8997,6.0751,9.3080,6.5794,6.5318,7.6229,7.4855,6.4045,4.9270,2.8963,4.4186,3.3534,5.1701,2.7866,2.8520,2.6501,4.3938,4.9116,4.7574,3.6277,4.9825,5.3040,5.1084,4.6044,3.6068,4.7471,4.3500,5.8861,3.3263,3.0579,2.3708,2.0565,2.5669,3.5118,0.2415,2.9416,2.0356,0.8878;
    
    // 7,5,10,1
    MatrixXd Acenter(7,5);
    Acenter <<
    6.3115,    0.6983,    3.8072,    5.6404, 4.8154,
    1.9318,    2.6395,    4.9462,    4.5021,    4.7634,
    2.7725 ,   4.9458 ,   4.2697 ,   2.9812 ,   0.5378,
    2.9138  ,  0.1446  ,  4.8906  ,  2.8154  ,  4.1588,
    0.1293   , 3.3597   , 1.0076   , 3.0995   , 3.1845,
    1.2515    ,0.6013    ,2.1031    ,1.8841    ,4.5594,
    6.2188    ,7.5109    ,3.6680    ,4.2362    ,2.3682;
    
    const bool RANDOM = true;
    
    int nDimension = 50;
    int nGaussian = 10;
    int nDataPerGaussian = 1000;
    double noise = 3; //variance
    double unitRadius =10;
    

        
    
    int64 stamp1 = GetTimeMs64();
    DataGenerator data(nDimension, nGaussian, nDataPerGaussian, pow(noise,0.5), unitRadius);
        
    if(RANDOM) {
        SpecGmm test(data.X(), nGaussian);
        data.evaluate(test.centers());
    }
    else {
        SpecGmm test(A, nGaussian);
        evaluate(Acenter, test.centers());
    }

    int64 stamp4 = GetTimeMs64();
    cout << endl;
    cout << "Total time=" << (stamp4-stamp1)<< endl;
    
}

void specGmmExp() {
    

    int nDimension = 50;
    int nGaussian = 10;
    int nDataPerGaussian = 1000;
   // double noise = 3; //variance
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
