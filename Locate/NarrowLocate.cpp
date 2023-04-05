#include "../pch.h"
#include "NarrowLocate.h"
#include "RANSAC.h"
#include "ZhangSuenThin.h"

/**
 * @brief ��С������� ���ֱ�ߣ�ax+by+c=0
 * @param pts �����㼯
 * @param ϵ��tuple: lineFactor <a,b,c>
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
 * @brief ��⽻������
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
 * @brief �����������
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
                //��������
                str_pdata[j] = 0;
            }
            else if (data == 254)
            {
                //��������
                weld_pdata[j] = 0;
            }
        }
    }
}
/**
 * @brief ϸ��λ
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

    // �ز���
    sample::RANSAC wrr = sample::RANSAC(nearWeldPoints, m_RefinedInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> near_weldPoints;
    near_weldPoints.clear();
    near_weldPoints = wrr.GetInliers();

    sample::RANSAC srr = sample::RANSAC(nearStridePoints, m_RefinedInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> near_stripePoints;
    near_stripePoints.clear();
    near_stripePoints = srr.GetInliers();

    // ֱ�����
    std::tuple<float, float, float> weldLine, stripeLine;
    LineFit(near_weldPoints, weldLine);
    LineFit(near_stripePoints, stripeLine);

    // ��⽻��-��ȷ��λ
    SolvePoint(weldLine, stripeLine, m_RefinedPoint);
}
/**
 * @brief ��ʼ��λ
 * @param segment 
 * @return 
*/
void locate::NarrowLocate::Locate() {
    // ����Ŀ��
    cv::Mat weld_roi, stripe_roi;
    SplitSegment(m_SegMat, weld_roi, stripe_roi);


    // ��ֵ��
    cv::threshold(weld_roi, weld_roi, 0, 255, cv::THRESH_BINARY);
    cv::threshold(stripe_roi, stripe_roi, 0, 255, cv::THRESH_BINARY);


    // ZhangSuenϸ��
    thin::ZhangSuenThin zhangSuenThin_Weld = thin::ZhangSuenThin(weld_roi, 2);
    zhangSuenThin_Weld.ExtractCenters();
    std::vector<cv::Point2f> weldCenterPoints = zhangSuenThin_Weld.m_PointsList;

    thin::ZhangSuenThin zhangSuenThin_Stripe = thin::ZhangSuenThin(stripe_roi, 2);
    zhangSuenThin_Stripe.ExtractCenters();
    std::vector<cv::Point2f> stripeCenterPoints = zhangSuenThin_Stripe.m_PointsList;

    // RANSAC ȥ��
    sample::RANSAC wr = sample::RANSAC(weldCenterPoints, m_InitialInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> weldPoints;
    weldPoints.clear();
    weldPoints = wr.GetInliers();
    sample::RANSAC sr = sample::RANSAC(stripeCenterPoints, m_InitialInlierThreshold, m_Iters, m_RansacSampleMinDis);
    std::vector<cv::Point2f> stripePoints;
    stripePoints.clear();
    stripePoints = sr.GetInliers();

    // ��С����ֱ�����
    std::tuple<float, float, float> weldLine, stripeLine;
    LineFit(weldPoints, weldLine);
    LineFit(stripePoints, stripeLine);


    // ��⽻��-���ζ�λ
    SolvePoint(weldLine, stripeLine, m_InitialPoint);

    // ��ȷ��λ
    RefinedPosition(weldPoints, stripePoints);


    // �������˲�
    
}
/**
 * @brief ȫ��������
 * @param seg �ָ�ͼ
 * @param weldResampleRange ���������ز�����Χ--��Χ**����
 * @param stripeResampleRange ���������ز�����Χ--��Χ**����
 * @param Iters RANSAC��������
 * @param RansacSampleMinDis Ransac������С����
 * @param InitialInlierThreshold �ֶ�λ�ڵ���ֵ
 * @param RefinedInlierThreshold ϸ��λ�ڵ���ֵ
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