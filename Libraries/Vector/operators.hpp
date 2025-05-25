#pragma once

#include "vectorBase.hpp"

namespace LinAlgebra {

#define _LINALG_VECVEC_SIZECHECK \
	if (a.size() != b.size()) throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFVEC_SIZECHECK \
	if (this->size() != a.size()) throw std::runtime_error(LINALGSIZEERROR);

// CONVENIENT EXPRESSION WRAPPERS

enum class OPType { Add, Sub, SubLeft, Mul, Div, DivLeft, Assign };

template <typename T, ulong sa, OPType _op>
struct SVOP {
	const Vector<T, sa>& vec;
	const T scalar;
	static constexpr OPType op = _op;
};

template <typename T, ulong sa>
auto operator+(const Vector<T, sa>& a, const T scalar) {
	return SVOP<T, sa, OPType::Add>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator+(const T scalar, const Vector<T, sa>& a) {
	return SVOP<T, sa, OPType::Add>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator-(const Vector<T, sa>& a, const T scalar) {
	return SVOP<T, sa, OPType::Sub>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator-(const T scalar, const Vector<T, sa>& a) {
	return SVOP<T, sa, OPType::SubLeft>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator*(const Vector<T, sa>& a, T scalar) {
	return SVOP<T, sa, OPType::Mul>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator*(const T scalar, const Vector<T, sa>& a) {
	return SVOP<T, sa, OPType::Mul>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator/(const Vector<T, sa>& a, const T scalar) {
	return SVOP<T, sa, OPType::Div>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa>
auto operator/(const T scalar, const Vector<T, sa>& a) {
	return SVOP<T, sa, OPType::DivLeft>{.vec = a, .scalar = scalar};
}

template <typename T, ulong sa, ulong sb, OPType _op>
struct VVOP {
	const Vector<T, sa>& lhs;
	const Vector<T, sb>& rhs;
	static constexpr OPType op = _op;
};

template <typename T, ulong sa, ulong sb>
auto operator+(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return VVOP<T, sa, sb, OPType::Add>{.lhs = a, .rhs = b};
}

template <typename T, ulong sa, ulong sb>
auto operator-(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return VVOP<T, sa, sb, OPType::Sub>{.lhs = a, .rhs = b};
}

template <typename T, ulong sa, ulong sb>
auto operator*(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return VVOP<T, sa, sb, OPType::Mul>{.lhs = a, .rhs = b};
}

template <typename T, ulong sa, ulong sb>
auto operator/(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return VVOP<T, sa, sb, OPType::Div>{.lhs = a, .rhs = b};
}

#undef _LINALG_VECVEC_SIZECHECK
#undef _LINALG_SELFVEC_SIZECHECK

}  // namespace LinAlgebra
