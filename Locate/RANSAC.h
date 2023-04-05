#pragma once
/**
 * @brief 随机一致性采样
*/
namespace sample {
	class RANSAC
	{
	private:
		// 原始点集
		std::vector<cv::Point2f> m_Points;
		// 内点集
		std::vector<cv::Point2f> m_Inliers;
		// 内点距离阈值
		float m_InlierDisThreshold;
		// 迭代次数
		int m_Iters;
		// 采样点间的最小距离
		float m_SamplingMinDis;
	public:
		RANSAC(std::vector<cv::Point2f>& pts, float inlierDisThreshold, int iters, float samplingMinDis);
		std::vector<cv::Point2f> GetInliers();
	private:
		inline std::tuple<int, int> Sampling(const int& index_size,
			const std::vector<std::tuple<int, int>>& sampled_indexes);
		inline std::tuple<double, double, double> LineFitting(const std::vector<cv::Point2f>& pts);
	};
}