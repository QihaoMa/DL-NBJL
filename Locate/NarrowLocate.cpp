#include "../pch.h"
#include "NarrowLocate.h"
#include "RANSAC.h"
#include "ZhangSuenThin.h"

/**
 * @brief 最小二乘拟合 拟合直线：ax+by+c=0
 * @param pts 样本点集
 * @param 系数tuple: lineFactor <a,b,c>
*/
void locate::NarrowLocate::LineFit(std::vector<cv::Point2f>& pts, std::tuple<float, float, float>& lineFactor)
{
    //ax+by+c=0
    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    get<0>(lineFactor) = line[1];
    get<1>(lineFactor) = -line[0];
    get<2>(lineFactor) = line[0] * line[3] - line[1] * line[2];
}
/**
 * @brief 求解交点坐标
 * @param w_line 
 * @param s_line 
 * @param intersect 
*/
void locate::NarrowLocate::SolvePoint(std::tuple<float, float, float>& w_line, std::tuple<float, float, float>& s_line, cv::Point2f& intersect) {
    Eigen::MatrixXd A = MatrixXd::Zero(2, 2);
    Eigen::MatrixXd b = MatrixXd::Zero(2, 1);
    A(0, 0) = get<0>(w_line);
    A(0, 1) = get<1>(w_line);
    A(1, 0) = get<0>(s_line);
    A(1, 1) = get<1>(s_line);
    b(0, 0) = get<2>(w_line) * (-1);
    b(1, 0) = get<2>(s_line) * (-1);
    Eigen::VectorXd a(2, 1);
    a = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    intersect.x = a(0, 0);
    intersect.y = a(1, 0);
}
/**
 * @brief 焊缝光条分离
 * @param src 
 * @param weld_dst 
 * @param stripe_dst 
*/
void locate::NarrowLocate::SplitSegment(cv::Mat& src, cv::Mat& weld_dst, cv::Mat& stripe_dst) {
    src.copyTo(weld_dst);
    src.copyTo(stripe_dst);
    for (int i = 0; i < src.rows; ++i)
    {
        uchar* pdata = src.ptr<uchar>(i);
        uchar* weld_pdata = weld_dst.ptr<uchar>(i);
        uchar* str_pdata = stripe_dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            uchar data = pdata[j];
            if (data == 127)
            {
                //焊缝区域
                str_pdata[j] = 0;
            }
            else if (data == 254)
            {
                //光条区域
                weld_pdata[j] = 0;
            }
        }
    }
}
/**
 * @brief 细定位
 * @param initialPoint 
 * @param refinedResult 
 * @param weldPointsList 
 * @param stripePointsList 
*/
void locate::NarrowLocate::RefinedPosition(std::vector<cv::Point2f>& weldPointsList, std::vector<cv::Point2f>& stripePointsList) {
    std::vector<cv::Point2f> nearWeldPoints, nearStridePoints;
    nearWeldPoints.clear();
    nearStridePoints.clear();
    for (auto& item : weldPointsList) {
        if (abs(item.x - m_InitialPoint.x) < m_WeldResampleRange && abs(item.y - m_InitialPoint.y) < m_WeldResampleRange) {
            nearWeldPoints.push_back(item);
        }
    }
    for (auto& item : stripePointsList) {
        if (abs(item.x - m_InitialPoint.x) < m_StripeResampleRange && abs(item.y - m_InitialPoint.y) < m_StripeResampleRange) {
            nearStridePoints.push_back(item);
        }
    }

    // 重采样
    sample::RANSAC wrr = sample::RANSAC(nearWeldPoints, m_RefinedInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> near_weldPoints;
    near_weldPoints.clear();
    near_weldPoints = wrr.GetInliers();

    sample::RANSAC srr = sample::RANSAC(nearStridePoints, m_RefinedInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> near_stripePoints;
    near_stripePoints.clear();
    near_stripePoints = srr.GetInliers();

    // 直线拟合
    std::tuple<float, float, float> weldLine, stripeLine;
    LineFit(near_weldPoints, weldLine);
    LineFit(near_stripePoints, stripeLine);

    // 求解交点-精确定位
    SolvePoint(weldLine, stripeLine, m_RefinedPoint);
}
/**
 * @brief 初始定位
 * @param segment 
 * @return 
*/
void locate::NarrowLocate::Locate() {
    // 剥离目标
    cv::Mat weld_roi, stripe_roi;
    SplitSegment(m_SegMat, weld_roi, stripe_roi);


    // 二值化
    cv::threshold(weld_roi, weld_roi, 0, 255, cv::THRESH_BINARY);
    cv::threshold(stripe_roi, stripe_roi, 0, 255, cv::THRESH_BINARY);


    // ZhangSuen细化
    thin::ZhangSuenThin zhangSuenThin_Weld = thin::ZhangSuenThin(weld_roi, 2);
    zhangSuenThin_Weld.ExtractCenters();
    std::vector<cv::Point2f> weldCenterPoints = zhangSuenThin_Weld.m_PointsList;

    thin::ZhangSuenThin zhangSuenThin_Stripe = thin::ZhangSuenThin(stripe_roi, 2);
    zhangSuenThin_Stripe.ExtractCenters();
    std::vector<cv::Point2f> stripeCenterPoints = zhangSuenThin_Stripe.m_PointsList;

    // RANSAC 去噪
    sample::RANSAC wr = sample::RANSAC(weldCenterPoints, m_InitialInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> weldPoints;
    weldPoints.clear();
    weldPoints = wr.GetInliers();
    sample::RANSAC sr = sample::RANSAC(stripeCenterPoints, m_InitialInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> stripePoints;
    stripePoints.clear();
    stripePoints = sr.GetInliers();

    // 最小二乘直线拟合
    std::tuple<float, float, float> weldLine, stripeLine;
    LineFit(weldPoints, weldLine);
    LineFit(stripePoints, stripeLine);


    // 求解交点-初次定位
    SolvePoint(weldLine, stripeLine, m_InitialPoint);

    // 精确定位
    RefinedPosition(weldPoints, stripePoints);


    // 卡尔曼滤波
    
}
/**
 * @brief 全参数构造
 * @param seg 分割图
 * @param weldResampleRange 焊缝条纹重采样范围--周围**像素
 * @param stripeResampleRange 激光条纹重采样范围--周围**像素
 * @param Iters RANSAC迭代次数
 * @param RansacSampleMinDis Ransac采样最小距离
 * @param InitialInlierThreshold 粗定位内点阈值
 * @param RefinedInlierThreshold 细定位内点阈值
*/
locate::NarrowLocate::NarrowLocate(cv::Mat seg, float weldResampleRange, float stripeResampleRange, int Iters,
    float RansacSampleMinDis, float InitialInlierThreshold, float RefinedInlierThreshold) {
    if (seg.empty()) {
        return;
    }
    m_SegMat = seg.clone();
    m_InitialPoint = cv::Point2f(NULL, NULL);
    m_RefinedPoint = cv::Point2f(NULL, NULL);
    m_WeldResampleRange = weldResampleRange;
    m_StripeResampleRange = stripeResampleRange;
    m_Iters = Iters;
    m_RansacSampleMinDis = RansacSampleMinDis;
    m_InitialInlierThreshold = InitialInlierThreshold;
    m_RefinedInlierThreshold = RefinedInlierThreshold;
}