#pragma once

#include <vector>

#include "Libraries/Matrix/matrix.hpp"
#include "Libraries/Vector/vector.hpp"
#include "OdeDynamics.hpp"

namespace ODEIntegration {

namespace Internal {

struct DormandPrinceTableau {
	static constexpr real b[7] = {	//
		35.0 / 384.0,				//
		0.0,						//
		500.0 / 1113.0,				//
		125.0 / 192.0,				//
		-2187.0 / 6784.0,			//
		11.0 / 84.0,				//
		0};
	static constexpr real bstar[7] = {	//
		5179.0 / 57600.0,				//
		0.0,							//
		7571.0 / 16695.0,				//
		393.0 / 640.0,					//
		-92097.0 / 339200.0,			//
		187.0 / 2100.0,					//
		1.0 / 40.0};

	static constexpr real bdiff[7] = {
		b[0] - bstar[0], b[1] - bstar[1], b[2] - bstar[2], b[3] - bstar[3],
		b[4] - bstar[4], b[5] - bstar[5], b[6] - bstar[6]};

	static constexpr real c[7] = {0.0,		 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0,
								  8.0 / 9.0, 1.0,		1.0};

	static constexpr real a[7][6] = {
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
		{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
		{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
		 0.0, 0.0},
		{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
		 -5103.0 / 18656.0, 0.0},
		{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
		 11.0 / 84.0}};
};

struct CashKarpTableau {
	static constexpr real b[6] = {	//
		37.0 / 378.0,				//
		0.0,						//
		250.0 / 621.0,				//
		125.0 / 594.0,				//
		0.0,						//
		512.0 / 1771.0};
	static constexpr real bstar[6] = {	//
		2825.0 / 27648.0,				//
		0.0,							//
		18575.0 / 48384.0,				//
		13525.0 / 55296.0,				//
		277.0 / 14336.0,				//
		1.0 / 4.0};
	static constexpr real bdiff[6] = {b[0] - bstar[0], b[1] - bstar[1],
									  b[2] - bstar[2], b[3] - bstar[3],
									  b[4] - bstar[4], b[5] - bstar[5]};
	static constexpr real c[6] = {0.0,		 1.0 / 5.0, 3.0 / 10.0,
								  3.0 / 5.0, 1.0,		7.0 / 8.0};
	static constexpr real a[6][5] = {
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0},
		{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0},
		{3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0},
		{-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0},
		{1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0,
		 253.0 / 4096.0}};
};

};	// namespace Internal

template <ulong stateSize = 0, bool recordStats = false>
struct RungeKutta45Vector {
	ODEDynamicsVector<real, stateSize>& ode;
	using Tableau = Internal::CashKarpTableau;

	LinAlgebra::Vector<real, stateSize> k1;
	LinAlgebra::Vector<real, stateSize> k2;
	LinAlgebra::Vector<real, stateSize> k3;
	LinAlgebra::Vector<real, stateSize> k4;
	LinAlgebra::Vector<real, stateSize> k5;
	LinAlgebra::Vector<real, stateSize> k6;
	LinAlgebra::Vector<real, stateSize> kdiff;

	LinAlgebra::Vector<real, stateSize> l1;
	LinAlgebra::Vector<real, stateSize> l2;
	LinAlgebra::Vector<real, stateSize> l3;
	LinAlgebra::Vector<real, stateSize> l4;
	LinAlgebra::Vector<real, stateSize> l5;
	LinAlgebra::Vector<real, stateSize> l6;

	std::vector<uint8_t> numLoopEvals;
	std::vector<single> timeSteps;

	real h;
	const real hmin, hmax;
	const real atol, rtol;

	RungeKutta45Vector(ODEDynamicsVector<real, stateSize>& ode, real hinit = 1,
					   real hmin = 1e-6, real hmax = 20, real atol = 1e-6,
					   real rtol = 1e-6)
		: ode(ode), h(hinit), hmin(hmin), hmax(hmax), atol(atol), rtol(rtol) {}

	real step(LinAlgebra::Vector<real, stateSize>& state, const real t,
			  const real t_end);

	void operator()(const LinAlgebra::Vector<real, stateSize>& state,
					LinAlgebra::Vector<real, stateSize>& workState,
					const real t, const real t_end) {
		workState = state;
		real time = t;
		while (time < t_end) {
			ode.PreIntegration(workState, time);
			time += step(workState, time, t_end);
			ode.PostIntegration(workState, time);
		}
	}
};

template <ulong stateRows = 0, ulong stateCols = 0, bool recordStats = false>
struct RungeKutta45Matrix {
	ODEDynamicsMatrix<real, stateRows, stateCols>& ode;
	using Tableau = Internal::DormandPrinceTableau;

	LinAlgebra::Matrix<real, stateRows, stateCols> k1;
	LinAlgebra::Matrix<real, stateRows, stateCols> k2;
	LinAlgebra::Matrix<real, stateRows, stateCols> k3;
	LinAlgebra::Matrix<real, stateRows, stateCols> k4;
	LinAlgebra::Matrix<real, stateRows, stateCols> k5;
	LinAlgebra::Matrix<real, stateRows, stateCols> k6;
	LinAlgebra::Matrix<real, stateRows, stateCols> kdiff;

	LinAlgebra::Matrix<real, stateRows, stateCols> l1;
	LinAlgebra::Matrix<real, stateRows, stateCols> l2;
	LinAlgebra::Matrix<real, stateRows, stateCols> l3;
	LinAlgebra::Matrix<real, stateRows, stateCols> l4;
	LinAlgebra::Matrix<real, stateRows, stateCols> l5;
	LinAlgebra::Matrix<real, stateRows, stateCols> l6;

	std::vector<uint8_t> numLoopEvals;
	std::vector<single> timeSteps;

	real h;
	const real hmin, hmax;
	const real atol, rtol;

	RungeKutta45Matrix(ODEDynamicsMatrix<real, stateRows, stateCols>& ode,
					   real hinit = 1, real hmin = 1e-6, real hmax = 20,
					   real atol = 1e-6, real rtol = 1e-6)
		: ode(ode), h(hinit), hmin(hmin), hmax(hmax), atol(atol), rtol(rtol) {}

	real step(LinAlgebra::Matrix<real, stateRows, stateCols>& state,
			  const real t, const real t_end);

	void operator()(const LinAlgebra::Matrix<real, stateRows, stateCols>& state,
					LinAlgebra::Matrix<real, stateRows, stateCols>& workState,
					const real t, const real t_end) {
		workState = state;
		real time = t;
		while (time < t_end) {
			ode.PreIntegration(workState, time);
			time += step(workState, time, t_end);
			ode.PostIntegration(workState, time);
		}
	}
};

// Definitions of step
template <ulong stateSize, bool recordStats>
real RungeKutta45Vector<stateSize, recordStats>::step(
	LinAlgebra::Vector<real, stateSize>& state, const real t,
	const real t_end) {
	using namespace LinAlgebra;

	// alloc only happens on first invocation
	k1.alloc(state.size());
	k2.alloc(state.size());
	k3.alloc(state.size());
	k4.alloc(state.size());
	k5.alloc(state.size());
	k6.alloc(state.size());
	kdiff.alloc(state.size());

	l1.alloc(state.size());
	l2.alloc(state.size());
	l3.alloc(state.size());
	l4.alloc(state.size());
	l5.alloc(state.size());
	l6.alloc(state.size());

	real stateNorm = ode.stateNorm(state);

	real h = this->h;
	real hprop = 0.0;

	const uint8_t maxIters = 48;
	uint8_t loopEvals = 0;
	for (; loopEvals < maxIters; loopEvals++) {
		real err_estimate = 0.0;

		// this isn't particularly necessary but good to start on a clean slate
		k1.setZero();
		k2.setZero();
		k3.setZero();
		k4.setZero();
		k5.setZero();
		k6.setZero();

		// copy state over
		l1 = state;
		l2 = state;
		l3 = state;
		l4 = state;
		l5 = state;
		l6 = state;

		ode.Gradient(l1, k1, t);
		k1 *= h;

		l2 += Tableau::a[1][0] * k1;
		ode.Gradient(l2, k2, t + Tableau::c[1]);
		k2 *= h;

		l3 += Tableau::a[2][0] * k1;
		l3 += Tableau::a[2][1] * k2;
		ode.Gradient(l3, k3, t + Tableau::c[2]);
		k3 *= h;

		l4 += Tableau::a[3][0] * k1;
		l4 += Tableau::a[3][1] * k2;
		l4 += Tableau::a[3][2] * k3;
		ode.Gradient(l4, k4, t + Tableau::c[3]);
		k4 *= h;

		l5 += Tableau::a[4][0] * k1;
		l5 += Tableau::a[4][1] * k2;
		l5 += Tableau::a[4][2] * k3;
		l5 += Tableau::a[4][3] * k4;
		ode.Gradient(l5, k5, t + Tableau::c[4]);
		k5 *= h;

		l6 += Tableau::a[5][0] * k1;
		l6 += Tableau::a[5][1] * k2;
		l6 += Tableau::a[5][2] * k3;
		l6 += Tableau::a[5][3] * k4;
		l6 += Tableau::a[5][4] * k5;
		ode.Gradient(l6, k6, t + Tableau::c[5]);
		k6 *= h;

		// the difference in updates will be computed to k6
		kdiff.setZero();
		kdiff += Tableau::bdiff[0] * k1;
		kdiff += Tableau::bdiff[1] * k2;
		kdiff += Tableau::bdiff[2] * k3;
		kdiff += Tableau::bdiff[3] * k4;
		kdiff += Tableau::bdiff[4] * k5;
		kdiff += Tableau::bdiff[5] * k6;

		err_estimate = ode.stateNorm(kdiff);
		err_estimate /= atol + rtol * stateNorm;

		// compute proposed update timestep
		hprop = 0.97 * h * std::pow(err_estimate, -0.20);
		h = (h >= 0.0 ? std::max(hprop, 0.05 * h) : std::min(hprop, 20 * h));
		h = std::max(h, hmin);

		if (err_estimate <= 1.0 || h == hmin) {
			break;
		}
	}

	// if the step suggested is out of bounds, clip it to bounds and make
	// another update step calculation
	if (h > hmax || (t + h) > t_end) {
		loopEvals++;
		h = std::min(hmax, t_end - t);

		ode.Gradient(l1, k1, t);
		k1 *= h;

		l2 += Tableau::a[1][0] * k1;
		ode.Gradient(l2, k2, t + Tableau::c[1]);
		k2 *= h;

		l3 += Tableau::a[2][0] * k1;
		l3 += Tableau::a[2][1] * k2;
		ode.Gradient(l3, k3, t + Tableau::c[2]);
		k3 *= h;

		l4 += Tableau::a[3][0] * k1;
		l4 += Tableau::a[3][1] * k2;
		l4 += Tableau::a[3][2] * k3;
		ode.Gradient(l4, k4, t + Tableau::c[3]);
		k4 *= h;

		l5 += Tableau::a[4][0] * k1;
		l5 += Tableau::a[4][1] * k2;
		l5 += Tableau::a[4][2] * k3;
		l5 += Tableau::a[4][3] * k4;
		ode.Gradient(l5, k5, t + Tableau::c[4]);
		k5 *= h;

		l6 += Tableau::a[5][0] * k1;
		l6 += Tableau::a[5][1] * k2;
		l6 += Tableau::a[5][2] * k3;
		l6 += Tableau::a[5][3] * k4;
		l6 += Tableau::a[5][4] * k5;
		ode.Gradient(l6, k6, t + Tableau::c[5]);
		k6 *= h;
	}

	// k1 is now the full update step
	k1 *= Tableau::b[0];
	k1 += Tableau::b[1] * k2;
	k1 += Tableau::b[2] * k3;
	k1 += Tableau::b[3] * k4;
	k1 += Tableau::b[4] * k5;
	k1 += Tableau::b[5] * k6;

	// record stats if requested
	if constexpr (recordStats) {
		numLoopEvals.push_back(loopEvals);
		timeSteps.push_back(h);
	}

	// store integrator state
	this->h = h;

	// update state variable
	state += k1;

	// return timestep
	return h;
}

template <ulong stateRows, ulong stateCols, bool recordStats>
real RungeKutta45Matrix<stateRows, stateCols, recordStats>::step(
	LinAlgebra::Matrix<real, stateRows, stateCols>& state, const real t,
	const real t_end) {
	using namespace LinAlgebra;

	// alloc only happens on first invocation
	k1.alloc(state.rows(), state.cols());
	k2.alloc(state.rows(), state.cols());
	k3.alloc(state.rows(), state.cols());
	k4.alloc(state.rows(), state.cols());
	k5.alloc(state.rows(), state.cols());
	k6.alloc(state.rows(), state.cols());
	kdiff.alloc(state.rows(), state.cols());

	l1.alloc(state.rows(), state.cols());
	l2.alloc(state.rows(), state.cols());
	l3.alloc(state.rows(), state.cols());
	l4.alloc(state.rows(), state.cols());
	l5.alloc(state.rows(), state.cols());
	l6.alloc(state.rows(), state.cols());

	// this isn't particularly necessary but good to start on a clean slate
	k1.setZero();
	k2.setZero();
	k3.setZero();
	k4.setZero();
	k5.setZero();
	k6.setZero();

	real stateNorm = ode.stateNorm(state);

	real h = this->h;
	real hprop = 0.0;

	const uint8_t maxIters = 24;
	uint8_t loopEvals = 0;
	for (; loopEvals < maxIters;) {
		loopEvals++;
		real err_estimate = 0.0;

		// this isn't particularly necessary but good to start on a clean slate
		k1.setZero();
		k2.setZero();
		k3.setZero();
		k4.setZero();
		k5.setZero();
		k6.setZero();

		// copy state over
		l1 = state;
		l2 = state;
		l3 = state;
		l4 = state;
		l5 = state;
		l6 = state;

		ode.Gradient(l1, k1, t);
		k1 *= h;

		l2 += Tableau::a[1][0] * k1;
		ode.Gradient(l2, k2, t + Tableau::c[1]);
		k2 *= h;

		l3 += Tableau::a[2][0] * k1;
		l3 += Tableau::a[2][1] * k2;
		ode.Gradient(l3, k3, t + Tableau::c[2]);
		k3 *= h;

		l4 += Tableau::a[3][0] * k1;
		l4 += Tableau::a[3][1] * k2;
		l4 += Tableau::a[3][2] * k3;
		ode.Gradient(l4, k4, t + Tableau::c[3]);
		k4 *= h;

		l5 += Tableau::a[4][0] * k1;
		l5 += Tableau::a[4][1] * k2;
		l5 += Tableau::a[4][2] * k3;
		l5 += Tableau::a[4][3] * k4;
		ode.Gradient(l5, k5, t + Tableau::c[4]);
		k5 *= h;

		l6 += Tableau::a[5][0] * k1;
		l6 += Tableau::a[5][1] * k2;
		l6 += Tableau::a[5][2] * k3;
		l6 += Tableau::a[5][3] * k4;
		l6 += Tableau::a[5][4] * k5;
		ode.Gradient(l6, k6, t + Tableau::c[5]);
		k6 *= h;

		kdiff.setZero();
		kdiff += Tableau::bdiff[0] * k1;
		kdiff += Tableau::bdiff[1] * k2;
		kdiff += Tableau::bdiff[2] * k3;
		kdiff += Tableau::bdiff[3] * k4;
		kdiff += Tableau::bdiff[4] * k5;
		kdiff += Tableau::bdiff[5] * k6;

		err_estimate = ode.stateNorm(kdiff);
		err_estimate /= atol + rtol * stateNorm;

		// compute proposed update timestep
		hprop = 0.97 * h * std::pow(err_estimate, -0.20);
		h = std::max(std::min(hprop, 10 * h), 0.1 * h);

		if (err_estimate <= 1.0) {
			break;
		}
	}
	h = std::max(h, hmin);

	// if the step suggested is out of bounds, clip it to bounds and make
	// another update step calculation
	if (h > hmax || (t + h) > t_end) {
		loopEvals++;
		h = std::min(hmax, t_end - t);

		ode.Gradient(l1, k1, t);
		k1 *= h;

		l2 += Tableau::a[1][0] * k1;
		ode.Gradient(l2, k2, t + Tableau::c[1]);
		k2 *= h;

		l3 += Tableau::a[2][0] * k1;
		l3 += Tableau::a[2][1] * k2;
		ode.Gradient(l3, k3, t + Tableau::c[2]);
		k3 *= h;

		l4 += Tableau::a[3][0] * k1;
		l4 += Tableau::a[3][1] * k2;
		l4 += Tableau::a[3][2] * k3;
		ode.Gradient(l4, k4, t + Tableau::c[3]);
		k4 *= h;

		l5 += Tableau::a[4][0] * k1;
		l5 += Tableau::a[4][1] * k2;
		l5 += Tableau::a[4][2] * k3;
		l5 += Tableau::a[4][3] * k4;
		ode.Gradient(l5, k5, t + Tableau::c[4]);
		k5 *= h;

		l6 += Tableau::a[5][0] * k1;
		l6 += Tableau::a[5][1] * k2;
		l6 += Tableau::a[5][2] * k3;
		l6 += Tableau::a[5][3] * k4;
		l6 += Tableau::a[5][4] * k5;
		ode.Gradient(l6, k6, t + Tableau::c[5]);
		k6 *= h;
	}

	// k1 is now the full update step
	k1 *= Tableau::b[0];
	k1 += Tableau::b[1] * k2;
	k1 += Tableau::b[2] * k3;
	k1 += Tableau::b[3] * k4;
	k1 += Tableau::b[4] * k5;
	k1 += Tableau::b[5] * k6;

	// record stats if requested
	if constexpr (recordStats) {
		numLoopEvals.push_back(loopEvals);
		timeSteps.push_back(h);
	}

	// store integrator state
	this->h = h;

	// update state variable
	state += k1;

	// return timestep
	return h;
}

};	// namespace ODEIntegration
