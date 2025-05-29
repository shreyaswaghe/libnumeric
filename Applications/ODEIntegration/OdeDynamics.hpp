#pragma once

#include "Libraries/Matrix/matrix.hpp"
#include "Libraries/Vector/vector.hpp"

namespace ODEIntegration {

// Abstract class defining minimum interface for ODE Integration
template <typename T, ulong stateSize = 0>
struct ODEDynamicsVector {
	virtual ~ODEDynamicsVector() = 0;

	virtual void PreIntegration(LinAlgebra::Vector<T, stateSize>& x,
								real t) = 0;

	virtual void PostIntegration(LinAlgebra::Vector<T, stateSize>& x,
								 real t) = 0;

	virtual void Gradient(const LinAlgebra::Vector<T, stateSize>& x,
						  LinAlgebra::Vector<T, stateSize>& gradout,
						  real t) = 0;

	virtual T stateNorm(const LinAlgebra::Vector<T, stateSize>& x) = 0;
};

template <typename T, ulong stateRows = 0, ulong stateCols = 0>
struct ODEDynamicsMatrix {
	virtual ~ODEDynamicsMatrix() = 0;

	virtual void PreIntegration(LinAlgebra::Matrix<T, stateRows, stateCols>& x,
								real t) = 0;

	virtual void PostIntegration(LinAlgebra::Matrix<T, stateRows, stateCols>& x,
								 real t) = 0;

	virtual void Gradient(const LinAlgebra::Matrix<T, stateRows, stateCols>& x,
						  LinAlgebra::Matrix<T, stateRows, stateCols>& gradout,
						  real t) = 0;

	virtual T stateNorm(
		const LinAlgebra::Matrix<T, stateRows, stateCols>& x) = 0;
};

template <typename T, ulong stateSize>
ODEDynamicsVector<T, stateSize>::~ODEDynamicsVector(){};

template <typename T, ulong stateRows, ulong stateCols>
ODEDynamicsMatrix<T, stateRows, stateCols>::~ODEDynamicsMatrix(){};

};	// namespace ODEIntegration
