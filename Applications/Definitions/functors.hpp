#pragma once

#include "Libraries/Vector/vector.hpp"

template <typename T>
struct RtoR {
	virtual ~RtoR() {};
	virtual T eval(T x) const = 0;
};

template <typename T>
struct XYtoR {
	virtual ~XYtoR() {};
	virtual T eval(T x, T y) const = 0;
};

template <typename T>
struct Vec2toR {
	virtual ~Vec2toR() {};
	virtual T eval(const LinAlgebra::Vector<T, 2>& x) const = 0;
};

template <typename T>
struct Vec3toR {
	virtual ~Vec3toR() {};
	virtual T eval(const LinAlgebra::Vector<T, 3>& x) const = 0;
};
