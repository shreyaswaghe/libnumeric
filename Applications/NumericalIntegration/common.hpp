#pragma once

#include "Libraries/Vector/vector.hpp"

namespace Quadrature {

template <typename T>
struct OneDimIntegrand {
	virtual ~OneDimIntegrand() = 0;
	virtual T eval(T x) const = 0;
};

template <typename T>
struct TwoDimIntegrand {
	virtual ~TwoDimIntegrand() = 0;
	virtual T eval(T x, T y) const = 0;
};

template <typename T, uint8_t outDim>
struct OneDimToVecIntegrand {
	virtual ~OneDimToVecIntegrand() = 0;
	virtual LinAlgebra::Vector<T, outDim> eval(T x) const = 0;
};

template <typename T, uint8_t inDim, uint8_t outDim>
struct VecToVecIntegrand {
	virtual ~VecToVecIntegrand() = 0;
	virtual LinAlgebra::Vector<T, outDim> eval(
		LinAlgebra::Vector<T, inDim> x) const = 0;
};

template <typename T>
OneDimIntegrand<T>::~OneDimIntegrand(){};

using RtoRIntegrand = OneDimIntegrand<real>;
using FtoFIntegrand = OneDimIntegrand<single>;

using RToVec2Integrand = OneDimToVecIntegrand<real, 2>;
using FToFVec2Integrand = OneDimToVecIntegrand<single, 2>;

using Vec2ToVec2Integrand = VecToVecIntegrand<real, 2, 2>;
using FVec2ToFVec2Integrand = VecToVecIntegrand<real, 2, 2>;

};	// namespace Quadrature
