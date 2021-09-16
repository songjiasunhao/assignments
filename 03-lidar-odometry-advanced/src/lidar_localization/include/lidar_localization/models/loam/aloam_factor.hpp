// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

//
// TODO: implement analytic Jacobians for LOAM residuals in this file
// 

#include <eigen3/Eigen/Dense>

//
// TODO: Sophus is ready to use if you have a good undestanding of Lie algebra.
// 
#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <math.h>

Eigen::Matrix<double,3,3> skew(Eigen::Matrix<double,3,1>& mat_in);

struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

class LidarEdgeJFactor : public ceres::SizedCostFunction<1,4,3>{
	public:
	 	LidarEdgeJFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,Eigen::Vector3d last_point_b_ ,double s_)
		 :curr_point(curr_point_),last_point_a(last_point_a_),last_point_b(last_point_b_),s(s_){}
		  
		  virtual ~LidarEdgeJFactor(){}
		  virtual bool Evaluate(double const *const* parameters, double *residuals, double **jacobians)const{
				Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
				Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1] );
				Eigen::Vector3d lp;
				lp = q_last_curr * curr_point + t_last_curr; 

				Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
				Eigen::Vector3d de = last_point_a - last_point_b;
				double de_norm = de.norm();
				residuals[0] = nu.norm()/de_norm;

				if (jacobians != NULL)
				{
					if (jacobians[0]!= NULL)
					{
						Eigen::Matrix3d skew_de = skew(de);
						Eigen::Vector3d rp = q_last_curr* curr_point;
						Eigen::Matrix3d skew_rp = skew(rp);

						Eigen::Map<Eigen::Matrix<double,1,4,Eigen::RowMajor>> J_so3(jacobians[0]);//RowMajor,表示按列排列。
					//在一个大循环中要不断读取Matrix中的一段连续数据，如果你每次都用block operation 去引用数据太累,于是就事先将这些数据构造成若干Map，那么以后循环中就直接操作Map就行了。
					J_so3.setZero();
					J_so3.block<1,3>(0,0) = nu.transpose()*skew_de*(-skew_rp)/(nu.norm()*de_norm);

					Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> J_t(jacobians[1]);
					J_t = nu.transpose()*skew_de/(nu.norm()*de_norm);
					}
					
				}
				return true;
		  }

		Eigen::Vector3d curr_point;
		Eigen::Vector3d last_point_a;
		Eigen::Vector3d last_point_b;
		double s;
 };

class LidarPlaneJFactor : public ceres::SizedCostFunction<1,4,3>{
			public:
			Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
			double s;

			LidarPlaneJFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
								Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_,double s_)
			: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_), last_point_m(last_point_m_),s(s_) {}

		virtual ~LidarPlaneJFactor(){}
		  virtual bool Evaluate(double const *const* parameters, double *residuals, double **jacobians)const{
			
			
			Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
			Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1] );
			Eigen::Vector3d lp;

			lp = q_last_curr * curr_point + t_last_curr; //new point
			Eigen::Vector3d de = (last_point_l-last_point_j).cross(last_point_m-last_point_j);
			double nu = (lp-last_point_j).dot(de);
			double phi  = nu/(de.norm());
			residuals[0] = std::fabs(phi);//

			 if(jacobians != NULL)
			{
				if(jacobians[0] != NULL)
				{
					if (residuals[0]!=0)
					{
						phi = phi/residuals[0];//直接得出正负X/|X|
					}
					
					Eigen::Vector3d rp = q_last_curr*curr_point;
					Eigen::Matrix3d skew_rp = skew(rp);
					
					Eigen::Map<Eigen::Matrix<double,1,4,Eigen::RowMajor>> J_so3(jacobians[0]);
					J_so3.setZero();
					J_so3.block<1,3>(0,0) =phi*(de/de.norm()).transpose()*(-skew_rp);

					Eigen::Map<Eigen::Matrix<double,1,3,Eigen::RowMajor>> J_t(jacobians[1]);
					J_t = phi*(de/de.norm()).transpose();
				}
			}
			return true;
		  }

};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};

class PoseSO3Para:public ceres::LocalParameterization{
	public:

		PoseSO3Para(){}
		virtual ~PoseSO3Para(){}

		virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const{
		
		Eigen::Map<const Eigen::Quaterniond> quater(x);
		
		Eigen::Quaterniond delta_q;
		//getTransformFromSo3(Eigen::Map<const Eigen::Matrix<double,3,1>>(delta), delta_q);//将矩阵转化为四元数，还未写
		Eigen::Map<const Eigen::Vector3d> delta_so3(delta);
		delta_q = Sophus::SO3d::exp(delta_so3).unit_quaternion();

		Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);

		quater_plus = delta_q*quater;

		return true;

		}
		virtual bool ComputeJacobian(const double *x, double* jacobian) const
		{
			Eigen::Map<Eigen::Matrix<double, 4,3,Eigen::RowMajor>> j(jacobian);
			(j.topRows(3)).setIdentity();//按行分块，前3行
			(j.bottomRows(1)).setZero();

			return true;
		}
		virtual int GlobalSize() const {return 4;}
		virtual int LocalSize() const {return 3;}
};

