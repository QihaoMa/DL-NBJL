#include "../pch.h"
#include "KalmanFilter.h"

void filter::KalmanFilter::Init_Par(Eigen::VectorXd& x, Eigen::MatrixXd& P, Eigen::MatrixXd& R, Eigen::MatrixXd& Q,
    Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& H, Eigen::VectorXd& u)
{
    m_x = x;
    m_P = P;
    m_R = R;
    m_Q = Q;
    m_A = A;
    m_B = B;
    m_H = H;
    m_u = u;
}

void filter::KalmanFilter::Predict_State()
{
    Eigen::VectorXd tmp_state = m_A * m_x + m_B * m_u;
    m_x = tmp_state;
}

void filter::KalmanFilter::Predict_Cov()
{
    Eigen::MatrixXd tmp_cov = m_A * m_P * m_A.transpose() + m_Q;
    m_P = tmp_cov;
}

Eigen::VectorXd filter::KalmanFilter::Mea_Resd(Eigen::VectorXd& z)
{
    m_z = z;
    Eigen::VectorXd tmp_res = m_z - m_H * m_x;
    return tmp_res;
}

Eigen::MatrixXd filter::KalmanFilter::Cal_Gain()
{
    Eigen::MatrixXd tmp_gain = m_P * m_H.transpose() * (m_H * m_P * m_H.transpose() + m_R).inverse();
    return tmp_gain;
}

void filter::KalmanFilter::Update_State()
{
    Eigen::MatrixXd kal_gain = Cal_Gain();
    Eigen::VectorXd mea_res = Mea_Resd(m_z);
    m_x = m_x + kal_gain * mea_res;
}

void filter::KalmanFilter::Update_Cov()
{
    Eigen::MatrixXd kal_gain = Cal_Gain();
    Eigen::MatrixXd tmp_mat = kal_gain * m_H;
    m_P = (m_iden_mat - tmp_mat) * m_P;
}
