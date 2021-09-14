/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */

#include <pcl/common/transforms.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "glog/logging.h"

#include "lidar_localization/models/registration/icp_svd_registration.hpp"

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(
    const YAML::Node& node
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    // parse params:
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPSVDRegistration::ICPSVDRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPSVDRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    // set params:
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    euc_fitness_eps_ = euc_fitness_eps;
    max_iter_ = max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;
    input_target_kdtree_->setInputCloud(input_target_);

    return true;
}
// 重载父类的虚函数，使用面向对象的三大特性之一的多态
bool ICPSVDRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);

    // init estimation:
    transformation_.setIdentity();
    
    //
    // TODO: first option -- implement all computing logic on your own
    //
    // do estimation:
    int curr_iter = 0;
    while (curr_iter < max_iter_) {
        // TODO: apply current estimation:
        Eigen::Matrix4f temp_transformation_ = Eigen::Matrix4f::Identity();
        CloudData::CLOUD_PTR current_input_soucre(new CloudData::CLOUD());
        pcl::transformPointCloud(*transformed_input_source, *current_input_soucre, transformation_);
        //如果使用pcl::transformPointCloud(*transformed_input_source, *transformed_input_source, transformation_);轨迹会产生抖动，不知道原因
        // TODO: get correspondence:
        std::vector<Eigen::Vector3f> xs;
        std::vector<Eigen::Vector3f> ys;
        size_t point_num = GetCorrespondence(current_input_soucre,xs,ys);
        // TODO: do not have enough correspondence -- break:
        if (point_num<10)
        break;
        // TODO: update current transform:
        GetTransform(xs,ys,temp_transformation_);
        // TODO: whether the transformation update is significant:
        if(IsSignificant(temp_transformation_,trans_eps_))
        // TODO: update transformation:
        transformation_ =temp_transformation_*transformation_;//
        ++curr_iter;
    }

    // set output:
    result_pose = transformation_ * predict_pose;//
    //归一化
    Eigen::Quaternionf q(result_pose.block<3,3>(0,0));
    q.normalize();
    result_pose.block<3,3>(0,0) = q.toRotationMatrix();
    
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);
    
    return true;
}
//公式中xs为target,ys为source
size_t ICPSVDRegistration::GetCorrespondence(
    const CloudData::CLOUD_PTR &input_source, 
    std::vector<Eigen::Vector3f> &xs,
    std::vector<Eigen::Vector3f> &ys
) {
    const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;

    size_t num_corr = 0;
    // TODO: set up point correspondence

    for (size_t i = 0; i < input_source->size(); i++)
    {
        std::vector<float> distance;
        std::vector<int> index;
        input_target_kdtree_->nearestKSearch(input_source->at(i),1,index,distance);//第一个参数为基准点，返回target的索引
        if(distance[0]<MAX_CORR_DIST_SQR)
        {
            ys.push_back(Eigen::Vector3f(input_source->at(i).x,input_source->at(i).y,input_source->at(i).z));
            xs.push_back(Eigen::Vector3f(input_target_->at(index[0]).x,input_target_->at(index[0]).y,input_target_->at(index[0]).z));
            num_corr++;
        }
    }
    //
    return num_corr;
}
//公式中xs为target,ys为source,所求R为s->t,source为当前帧，target为局部地图点云
void ICPSVDRegistration::GetTransform(
    const std::vector<Eigen::Vector3f> &xs,
    const std::vector<Eigen::Vector3f> &ys,
    Eigen::Matrix4f &transformation_
) {
    const size_t N = xs.size();

    // TODO: find centroids of mu_x and mu_y:

    Eigen::Vector3f mu_x = Eigen::Vector3f::Zero();
    Eigen::Vector3f mu_y = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> x_h ;
    std::vector<Eigen::Vector3f> y_h ;
   for (size_t i = 0; i < N; i++)
   {
       mu_x +=xs[i];
       mu_y +=ys[i];
   }
   mu_x = mu_x/N;
   mu_y = mu_y/N;
    // TODO: build H:
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < N; i++)
    {
        x_h.push_back(xs[i]-mu_x);
        y_h.push_back(ys[i]-mu_y);
    }
       for (size_t i = 0; i < N; i++)
    {
        H+=y_h[i]*x_h[i].transpose();//y->x
    }
    // TODO: solve R:
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R =V*(U.transpose());
    // TODO: solve t:
    Eigen::Vector3f t= mu_x - R*mu_y;
    // TODO: set output:
    //transformation_ <<R,t,0,0,0,1;
    transformation_.block<3,3>(0,0) = R;
    transformation_.block<3,1>(0,3) = t;
}

bool ICPSVDRegistration::IsSignificant(
    const Eigen::Matrix4f &transformation,
    const float trans_eps
) {
    // a. translation magnitude -- norm:
    float translation_magnitude = transformation.block<3, 1>(0, 3).norm();
    // b. rotation magnitude -- angle:
    float rotation_magnitude = fabs(
        acos(
            (transformation.block<3, 3>(0, 0).trace() - 1.0f) / 2.0f
        )
    );

    return (
        (translation_magnitude > trans_eps) || 
        (rotation_magnitude > trans_eps)
    );
}

} // namespace lidar_localization