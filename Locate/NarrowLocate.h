#pragma once
/**
 * @brief 定位算法 细化-采样-初始定位-精确定位
*/
namespace locate {
	class NarrowLocate
	{
	public:
		cv::Mat m_SegMat;			//分割Mat
		cv::Point2f m_InitialPoint;	//粗定位结果
		cv::Point2f m_RefinedPoint;	//细定位结果
		float m_WeldResampleRange;	
		float m_StripeResampleRange;
		float m_RansacSampleMinDis;
		float m_InitialInlierThreshold;
		float m_RefinedInlierThreshold;
		int m_Iters;
	protected:
		void LineFit(std::vector<cv::Point2f>& pts, std::tuple<float, float, float>& lineFactor);
		void SolvePoint(std::tuple<float, float, float>& w_line, std::tuple<float, float, float>& s_line,
			cv::Point2f& intersect);
		void SplitSegment(cv::Mat& src, cv::Mat& weld_dst, cv::Mat& stripe_dst);
		void RefinedPosition(std::vector<cv::Point2f>& weldPointsList, std::vector<cv::Point2f>& stripePointsList);
	public:
		void Locate();
		NarrowLocate(cv::Mat seg, float weldResampleRange, float stripeResampleRange, int Iters, float RansacSampleMinDis,
			float InitialInlierThreshold, float RefinedInlierThreshold);
		~NarrowLocate() {};
	};
}