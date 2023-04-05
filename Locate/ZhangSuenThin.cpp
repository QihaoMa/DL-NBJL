#include "../pch.h"
#include "ZhangSuenThin.h"

thin::ZhangSuenThin::ZhangSuenThin(cv::Mat image, int sampleStep) {
	this->m_Image = image.clone();
	this->m_SampleStep = sampleStep;
	if (m_Image.empty()) {
		std::cerr << "Thin Image not be empty!" << endl;
		return;
	}
	if (m_Image.type() != CV_8UC1) {
		cv::threshold(m_Image, m_Image, 0, 255, cv::THRESH_BINARY);
	}
	m_Height = m_Image.rows;
	m_Width = m_Image.cols;
	m_Step = m_Image.step;
}

void thin::ZhangSuenThin::ExtractCenters() {
	m_ThinMat = m_Image.clone();
	bool ifEnd;
	int p1, p2, p3, p4, p5, p6, p7, p8;
	vector<uchar*> flag;
	uchar* img = m_ThinMat.data;

	while (true) {
		ifEnd = false;
		for (int i = 0; i < m_Height; ++i) {
			for (int j = 0; j < m_Width; ++j) {
				uchar* p = img + i * m_Step + j;
				if (*p == 0) {
					// background skip
					continue;
				}
				//�жϰ��������ص��ֵ(Ҫ���Ǳ߽�����),��Ϊǰ����(��ɫ255),��Ϊ1;��֮Ϊ0
				p1 = p[(i == 0) ? 0 : -m_Step] > 0 ? 1 : 0;
				p2 = p[(i == 0 || j == m_Width - 1) ? 0 : -m_Step + 1] > 0 ? 1 : 0;
				p3 = p[(j == m_Width - 1) ? 0 : 1] > 0 ? 1 : 0;
				p4 = p[(i == m_Height - 1 || j == m_Width - 1) ? 0 : m_Step + 1] > 0 ? 1 : 0;
				p5 = p[(i == m_Height - 1) ? 0 : m_Step] > 0 ? 1 : 0;
				p6 = p[(i == m_Height - 1 || j == 0) ? 0 : m_Step - 1] > 0 ? 1 : 0;
				p7 = p[(j == 0) ? 0 : -1] > 0 ? 1 : 0;
				p8 = p[(i == 0 || j == 0) ? 0 : -m_Step - 1] > 0 ? 1 : 0;
				if ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) >= 2 && (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) <= 6) //����1
				{
					//����2�ļ���
					int count = 0;
					if (p1 == 0 && p2 == 1) ++count;
					if (p2 == 0 && p3 == 1) ++count;
					if (p3 == 0 && p4 == 1) ++count;
					if (p4 == 0 && p5 == 1) ++count;
					if (p5 == 0 && p6 == 1) ++count;
					if (p6 == 0 && p7 == 1) ++count;
					if (p7 == 0 && p8 == 1) ++count;
					if (p8 == 0 && p1 == 1) ++count;
					if (count == 1 && p1 * p3 * p5 == 0 && p3 * p5 * p7 == 0) { //����2��3��4
						flag.push_back(p); //����ǰ������ӵ���ɾ��������
					}
				}
			}
		}
		//����ǵĵ�ɾ��    
		for (vector<uchar*>::iterator i = flag.begin(); i != flag.end(); ++i) {
			**i = 0;
			ifEnd = true;
		}
		flag.clear(); //��մ�ɾ������

		for (int i = 0; i < m_Height; ++i) {
			for (int j = 0; j < m_Width; ++j) {
				uchar* p = img + i * m_Step + j;
				if (*p == 0) continue;
				p1 = p[(i == 0) ? 0 : -m_Step] > 0 ? 1 : 0;
				p2 = p[(i == 0 || j == m_Width - 1) ? 0 : -m_Step + 1] > 0 ? 1 : 0;
				p3 = p[(j == m_Width - 1) ? 0 : 1] > 0 ? 1 : 0;
				p4 = p[(i == m_Height - 1 || j == m_Width - 1) ? 0 : m_Step + 1] > 0 ? 1 : 0;
				p5 = p[(i == m_Height - 1) ? 0 : m_Step] > 0 ? 1 : 0;
				p6 = p[(i == m_Height - 1 || j == 0) ? 0 : m_Step - 1] > 0 ? 1 : 0;
				p7 = p[(j == 0) ? 0 : -1] > 0 ? 1 : 0;
				p8 = p[(i == 0 || j == 0) ? 0 : -m_Step - 1] > 0 ? 1 : 0;
				if ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) >= 2 && (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) <= 6)
				{
					int count = 0;
					if (p1 == 0 && p2 == 1) ++count;
					if (p2 == 0 && p3 == 1) ++count;
					if (p3 == 0 && p4 == 1) ++count;
					if (p4 == 0 && p5 == 1) ++count;
					if (p5 == 0 && p6 == 1) ++count;
					if (p6 == 0 && p7 == 1) ++count;
					if (p7 == 0 && p8 == 1) ++count;
					if (p8 == 0 && p1 == 1) ++count;
					if (count == 1 && p1 * p3 * p7 == 0 && p1 * p5 * p7 == 0) {
						flag.push_back(p);
					}
				}
			}
		}
		//����ǵĵ�ɾ��    
		for (vector<uchar*>::iterator i = flag.begin(); i != flag.end(); ++i) {
			**i = 0;
			ifEnd = true;
		}
		flag.clear();
		if (!ifEnd) break; //��û�п���ɾ�������ص㣬���˳�ѭ��
	}
	// ��ȡ����
	this->GetCoordinate();
}

void thin::ZhangSuenThin::saveThinMat(std::string fileName, cv::Scalar scala) {
	Mat thinMat = Mat::zeros(Size(m_Width, m_Height), CV_8UC3);
	for (auto item : m_PointsList) {
		circle(thinMat, Point(item.x, item.y), 1, scala, -1);
	}
	cv::imwrite(fileName, thinMat);
}


inline void thin::ZhangSuenThin::GetCoordinate() {
	uchar* img = m_ThinMat.data;
	int counter = 0;
	for (int i = 0; i < m_Height; ++i) {
		for (int j = 0; j < m_Width; ++j) {
			uchar* p = img + i * m_Step + j;
			if (*p != 0) {
				counter += 1;
				if (counter % m_SampleStep == 0) {
					m_PointsList.push_back(cv::Point2f(j, i));
				}
			}
		}
	}
}