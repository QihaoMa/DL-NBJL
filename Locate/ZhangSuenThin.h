#pragma once
/**
 * @brief ÌõÎÆÏ¸»¯
*/
namespace thin {
	class ZhangSuenThin
	{
	private:
		cv::Mat m_Image;
		cv::Mat m_ThinMat;
		int m_Height;
		int m_Width;
		int m_Step;
		int m_SampleStep;
	public:
		std::vector<cv::Point2f> m_PointsList;
	public:
		ZhangSuenThin(cv::Mat image, int sampleStep);
		void ExtractCenters();
		void saveThinMat(std::string fileName, cv::Scalar scala);
	protected:
		inline void GetCoordinate();
	};
}

