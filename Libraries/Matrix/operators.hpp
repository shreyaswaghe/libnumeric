#pragma once

#include "Libraries/Vector/operators.hpp"
#include "matrixBase.hpp"

namespace LinAlgebra {

#define _LINALG_MATMAT_SIZECHECK                      \
	if (a.rows() != b.rows() || a.cols() != b.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

// CONVENIENT EXPRESSION WRAPPERS

template <typename T, ulong ra, ulong ca, OPType _op>
struct SMOP {
	const Matrix<T, ra, ca>& mat;
	const T scalar;
	static constexpr OPType op = _op;
};

template <typename T, ulong ra, ulong ca>
auto operator+(const Matrix<T, ra, ca>& a, const T scalar) {
	return SMOP<T, ra, ca, OPType::Add>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator+(const T scalar, const Matrix<T, ra, ca>& a) {
	return SMOP<T, ra, ca, OPType::Add>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator-(const Matrix<T, ra, ca>& a, const T scalar) {
	return SMOP<T, ra, ca, OPType::Sub>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator-(const T scalar, const Matrix<T, ra, ca>& a) {
	return SMOP<T, ra, ca, OPType::SubLeft>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator*(const Matrix<T, ra, ca>& a, T scalar) {
	return SMOP<T, ra, ca, OPType::Mul>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator*(const T scalar, const Matrix<T, ra, ca>& a) {
	return SMOP<T, ra, ca, OPType::Mul>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator/(const Matrix<T, ra, ca>& a, const T scalar) {
	return SMOP<T, ra, ca, OPType::Div>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca>
auto operator/(const T scalar, const Matrix<T, ra, ca>& a) {
	return SMOP<T, ra, ca, OPType::DivLeft>{.mat = a, .scalar = scalar};
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
struct MMOP {
	const Matrix<T, ra, ca>& lhs;
	const Matrix<T, rb, cb>& rhs;
	static constexpr OPType op = _op;
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator+(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return MMOP<T, ra, ca, rb, cb, OPType::Add>{.lhs = a, .rhs = b};
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator-(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return MMOP<T, ra, ca, rb, cb, OPType::Sub>{.lhs = a, .rhs = b};
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator*(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return MMOP<T, ra, ca, rb, cb, OPType::Mul>{.lhs = a, .rhs = b};
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator/(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	return MMOP<T, ra, ca, rb, cb, OPType::Div>{.lhs = a, .rhs = b};
}

#undef _LINALG_MATMAT_SIZECHECK

}  // namespace LinAlgebra
