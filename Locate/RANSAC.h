#pragma once
/**
 * @brief ���һ���Բ���
*/
namespace sample {
	class RANSAC
	{
	private:
		// ԭʼ�㼯
		std::vector<cv::Point2f> m_Points;
		// �ڵ㼯
		std::vector<cv::Point2f> m_Inliers;
		// �ڵ������ֵ
		float m_InlierDisThreshold;
		// ��������
		int m_Iters;
		// ����������С����
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