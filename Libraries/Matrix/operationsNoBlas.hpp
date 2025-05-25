#pragma once

#include "matrixBase.hpp"
#include "vectorBase.hpp"

namespace LinAlgebra {

#define _LINALG_MATMAT_SIZECHECK                      \
	if (a.rows() != b.rows() || a.cols() != b.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFMAT_SIZECHECK                               \
	if (this->rows() != ma.rows() || this->cols() != ma.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_MATVEC_COMPATIBILITY_CHECK \
	if (A.cols() != v.size()) throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_MATMAT_COMPATIBILITY_CHECK \
	if (A.cols() != B.rows()) throw std::runtime_error(LINALGSIZEERROR);

// Addition

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator+(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	constexpr ulong rc = (ra == rb) ? ra : 0;
	constexpr ulong cc = (ca == cb) ? ca : 0;
	Matrix<T, rc, cc> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] + b[i];
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator+(const Matrix<T, ra, ca>& a, T scalar) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] + scalar;
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator+(T scalar, const Matrix<T, ra, ca>& a) {
	return a + scalar;
}

// Subtraction

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator-(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	constexpr ulong rc = (ra == rb) ? ra : 0;
	constexpr ulong cc = (ca == cb) ? ca : 0;
	Matrix<T, rc, cc> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] - b[i];
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator-(const Matrix<T, ra, ca>& a, T scalar) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] - scalar;
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator-(T scalar, const Matrix<T, ra, ca>& a) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = scalar - a[i];
	return c;
}

// Multiplication

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator*(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	constexpr ulong rc = (ra == rb) ? ra : 0;
	constexpr ulong cc = (ca == cb) ? ca : 0;
	Matrix<T, rc, cc> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] * b[i];
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator*(const Matrix<T, ra, ca>& a, T scalar) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] * scalar;
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator*(T scalar, const Matrix<T, ra, ca>& a) {
	return a * scalar;
}

// Division

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
auto operator/(const Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	_LINALG_MATMAT_SIZECHECK;
	constexpr ulong rc = (ra == rb) ? ra : 0;
	constexpr ulong cc = (ca == cb) ? ca : 0;
	Matrix<T, rc, cc> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] / b[i];
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator/(const Matrix<T, ra, ca>& a, T scalar) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] / scalar;
	return c;
}

template <typename T, ulong ra, ulong ca>
auto operator/(T scalar, const Matrix<T, ra, ca>& a) {
	Matrix<T, ra, ca> c(a.rows(), a.cols());
	for (ulong i = 0; i < c.size(); i++) c[i] = scalar / a[i];
	return c;
}

// Unary + -

template <typename T, ulong ra, ulong ca>
auto operator+(const Matrix<T, ra, ca>& a) {
	return a;
}

template <typename T, ulong ra, ulong ca>
auto operator-(const Matrix<T, ra, ca>& a) {
	return -T(1.0) * a;
}

// Matvec

template <typename T, ulong __rows, ulong __cols, ulong sv, ulong su>
void matvecOp(const Matrix<T, __rows, __cols>& A, const Vector<T, sv>& v,
			  Vector<T, su>& out) {
	// matrix stored column-majorly
	_LINALG_MATVEC_COMPATIBILITY_CHECK;
	out.setZero();
	for (ulong iCol = 0; iCol < A.cols(); iCol++) {
		const T vj = v[iCol];
		const T* A_jcol = A(0, iCol);
#pragma clang loop vectorize(enable)
		for (ulong iRow = 0; iRow < A.rows(); iRow++) {
			out[iRow] += A_jcol[iRow] * vj;
		}
	}
}

template <typename T, ulong __rows, ulong __cols>
template <ulong sa>
Vector<T, __rows> Matrix<T, __rows, __cols>::matvec(
	const Vector<T, sa>& a) const {
	Vector<T, __rows> result(this->rows());
	matvecOp(*this, a, result);
	return result;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong sa, ulong sb>
void Matrix<T, __rows, __cols>::matvecNoAlloc(const Vector<T, sa>& a,
											  Vector<T, sb>& out) const {
	if (a.size() > out.size())
		std::runtime_error("Not enough space in out vector");
	matvecOp(*this, a, out);
}

// Matmat
template <typename T>
void matmat_atomic() {}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
Matrix<T, ra, cb> matmat(const Matrix<T, ra, ca>& A,
						 const Matrix<T, rb, cb>& B) {
	_LINALG_MATMAT_COMPATIBILITY_CHECK;
	Matrix<T, ra, cb> C(A.rows(), B.cols());
}

// Self-assignment operators
template <typename T, ulong __rows, ulong __cols>
template <ulong r, ulong c>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator+=(
	const Matrix<T, r, c>& ma) {
	_LINALG_SELFMAT_SIZECHECK;
	for (ulong i = 0; i < ma.size(); i++) _data->operator[](i) += ma[i];
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong r, ulong c>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator-=(
	const Matrix<T, r, c>& ma) {
	_LINALG_SELFMAT_SIZECHECK;
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < ma.size(); i++) _data->operator[](i) -= ma[i];
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong r, ulong c>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator*=(
	const Matrix<T, r, c>& ma) {
	_LINALG_SELFMAT_SIZECHECK;
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < ma.size(); i++) _data->operator[](i) *= ma[i];
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong r, ulong c>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator/=(
	const Matrix<T, r, c>& ma) {
	_LINALG_SELFMAT_SIZECHECK;
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < ma.size(); i++) _data->operator[](i) /= ma[i];
	return *this;
}

#undef _LINALG_MATMAT_SIZECHECK
#undef _LINALG_SELFMAT_SIZECHECK
#undef _LINALG_MATMAT_COMPATIBILITY_CHECK
#undef _LINALG_MATVEC_COMPATIBILITY_CHECK

}  // namespace LinAlgebra
