#include "../pch.h"
#include "RANSAC.h"
/**
 * @brief 构造函数
 * @param pts 
 * @param inlierDisThreshold 内点阈值 粗/细定位不同
 * @param iters 迭代次数 默认：20
 * @param samplingMinDis 最小采样点间距 默认：8.0f
*/
sample::RANSAC::RANSAC(std::vector<cv::Point2f>& pts, float inlierDisThreshold, int iters, float samplingMinDis) {
    if (pts.size() == 0) {
        std::cerr << "Ransac list is empty！" << std::endl;
        return;
    }
    m_Points = pts;
    m_InlierDisThreshold = inlierDisThreshold;
    m_Iters = iters;
    m_SamplingMinDis = samplingMinDis;
}

std::vector<cv::Point2f> sample::RANSAC::GetInliers() {
    if (m_Points.size() <= 3 || m_InlierDisThreshold < 1e-6) {
        return m_Inliers;
    }

    std::vector<std::tuple<int, int>> sampled_indexes;
    std::vector<int> Is_Inlier(m_Points.size(), 0);
    std::vector<int> Is_Inlier_Temp(m_Points.size(), 0);

    int max_inlier_num = 0;
    int sample_count = 0;

    while (sample_count < m_Iters)
    {
        //3. 随机抽取两点(采样）
        tuple<int, int> p_tuple = this->Sampling(m_Points.size(), sampled_indexes);
        if (std::abs(m_Points[get<0>(p_tuple)].x - m_Points[get<1>(p_tuple)].x) < m_SamplingMinDis
            && std::abs(m_Points[get<0>(p_tuple)].y - m_Points[get<1>(p_tuple)].y) < m_SamplingMinDis)
        {
            continue;
        }
        else
        {
            sampled_indexes.push_back({ get<0>(p_tuple), get<1>(p_tuple) });
        }

        //4. 直线拟合
        tuple<double, double, double> line = this->LineFitting({ m_Points[get<0>(p_tuple)], m_Points[get<1>(p_tuple)] });



        //5. 基于拟合的直线区分内外点
        //内点数量
        int inlier_num = 0;

        for (int i = 0; i < m_Points.size(); ++i)
        {
            auto& p = m_Points[i];
            Is_Inlier_Temp[i] = 0;
            if (std::abs(get<0>(line) * p.x + get<1>(line) * p.y + get<2>(line)) < m_InlierDisThreshold)
            {
                Is_Inlier_Temp[i] = 1;
                inlier_num++;
                m_Inliers.push_back(m_Points[i]);
            }
        }

        if (inlier_num > max_inlier_num)
        {
            max_inlier_num = inlier_num;
            Is_Inlier = Is_Inlier_Temp;
        }
        //6. 更新迭代的最佳次数
        if (inlier_num != 0) {
            double epsilon = double(inlier_num) / (double)m_Points.size(); //内点比例
            double z = 0.99;                                                //所有样本中存在1个好样本的概率
            double n = 2.0;
            m_Iters = int(std::log(1.0 - z) / std::log(1.0 - std::pow(epsilon, n)));
        }
        sample_count++;
    }

    //7. 基于最优的结果所对应的内点做最终拟合

    m_Inliers.reserve(max_inlier_num);
    for (int i = 0; i < Is_Inlier.size(); ++i)
    {
        if (1 == Is_Inlier[i])
        {
            m_Inliers.push_back((cv::Point2f)m_Points[i]);
        }
    }
    return m_Inliers;
}


inline std::tuple<int, int> sample::RANSAC::Sampling(const int& index_size,
    const std::vector<std::tuple<int, int>>& sampled_indexes)
{
    assert(index_size > 2);

    while (true)
    {
        int index1 = rand() % index_size;
        int index2 = rand() % index_size;
        if (index1 == index2) { continue; }
        int min_index = std::min(index1, index2);
        int max_index = std::max(index1, index2);
        bool has_sampled = false;
        for (int k = 0; k < sampled_indexes.size(); ++k)
        {
            tuple<int, int> si = sampled_indexes[k];
            int i = get<0>(si);
            int j = get<1>(si);
            if (min_index == i && j == max_index)
            {
                has_sampled = true;
                break;
            }
        }
        if (has_sampled) { continue; }
        else { return { min_index, max_index }; }
    }
}

inline std::tuple<double, double, double> sample::RANSAC::LineFitting(const std::vector<cv::Point2f>& pts)
{
    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    double a = line[1];
    double b = -line[0];
    double c = line[0] * line[3] - line[1] * line[2];
    return { a, b, c };
}