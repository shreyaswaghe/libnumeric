#pragma once

#include "cblas.h"
#include "vectorBase.hpp"

namespace LinAlgebra {

#define _LINALG_VECVEC_SIZECHECK \
	if (a.size() != b.size()) throw std::runtime_error(LINALGSIZEERROR);

// Addition

template <typename T, ulong sa, ulong sb>
auto operator+(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	constexpr ulong sc = (sa == sb) ? sa : 0;
	Vector<T, sc> c(a.size());
	cblas_dcopy(a.size(), a(), 1, c(), 1);
	cblas_daxpy(a.size(), 1.0, b(), 1, c(), 1);
	return c;
}

template <typename T, ulong sa>
auto operator+(const Vector<T, sa>& a, T scalar) {
	T b[] = {scalar};
	Vector<T, sa> c(a.size());
	cblas_dcopy(a.size(), a(), 1, c(), 1);
	cblas_daxpy(a.size(), 1.0, b, 0, c(), 1);
	return c;
}

template <typename T, ulong sa>
auto operator+(T scalar, const Vector<T, sa>& a) {
	return a + scalar;
}

// Subtraction

template <typename T, ulong sa, ulong sb>
auto operator-(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	constexpr ulong sc = (sa == sb) ? sa : 0;
	Vector<T, sc> c(a.size());
	cblas_dcopy(a.size(), a(), 1, c(), 1);
	cblas_daxpy(a.size(), -1.0, b(), 1, c(), 1);
	return c;
}

template <typename T, ulong sa>
auto operator-(const Vector<T, sa>& a, T scalar) {
	T b[] = {scalar};
	Vector<T, sa> c(a.size());
	cblas_dcopy(a.size(), a(), 1, c(), 1);
	cblas_daxpy(a.size(), -1.0, b, 0, c(), 1);
	return c;
}

template <typename T, ulong sa>
auto operator-(T scalar, const Vector<T, sa>& a) {
	T b[] = {scalar};
	Vector<T, sa> c(a.size());
	cblas_dcopy(a.size(), b, 0, c(), 1);
	cblas_daxpy(a.size(), 1.0, a, 1, c(), 1);
	return c;
}

// Multiplication

template <typename T, ulong sa, ulong sb>
auto operator*(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	constexpr ulong sc = (sa == sb) ? sa : 0;
	Vector<T, sc> c(a.size());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] * b[i];
	return c;
}

template <typename T, ulong sa>
auto operator*(const Vector<T, sa>& a, T scalar) {
	Vector<T, sa> c(a.size());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] * scalar;
	return c;
}

template <typename T, ulong sa>
auto operator*(T scalar, const Vector<T, sa>& a) {
	return a * scalar;
}

// Division

template <typename T, ulong sa, ulong sb>
auto operator/(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	constexpr ulong sc = (sa == sb) ? sa : 0;
	Vector<T, sc> c(a.size());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] / b[i];
	return c;
}

template <typename T, ulong sa>
auto operator/(const Vector<T, sa>& a, T scalar) {
	Vector<T, sa> c(a.size());
	for (ulong i = 0; i < c.size(); i++) c[i] = a[i] / scalar;
	return c;
}

template <typename T, ulong sa>
auto operator/(T scalar, const Vector<T, sa>& a) {
	Vector<T, sa> c(a.size());
	for (ulong i = 0; i < c.size(); i++) c[i] = scalar / a[i];
	return c;
}

// Unary + -

template <typename T, ulong sa>
auto operator+(const Vector<T, sa>& a) {
	return a;
}

template <typename T, ulong sa>
auto operator-(const Vector<T, sa>& a) {
	return -T(1.0) * a;
}

// Dot product

template <typename T, ulong sa, ulong sb>
T dot(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	return cblas_dsdot(a.size(), a(), 1, b(), 1);
}

// self-assignment operators

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator+=(const Vector<T, sa>& a) {
	if (this->size() != a.size()) {
		throw std::runtime_error(LINALGSIZEERROR);
	}
	cblas_daxpy(a.size(), 1.0, a(), 1, _data(), 1);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator-=(const Vector<T, sa>& a) {
	if (this->size() != a.size()) {
		throw std::runtime_error(LINALGSIZEERROR);
	}
	cblas_daxpy(a.size(), -1.0, a(), 1, _data(), 1);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator*=(const Vector<T, sa>& a) {
	if (this->size() != a.size()) {
		throw std::runtime_error(LINALGSIZEERROR);
	}
	for (ulong i = 0; i < this->size(); i++) _data->operator[](i) *= a[i];
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator/=(const Vector<T, sa>& a) {
	if (this->size() != a.size()) {
		throw std::runtime_error(LINALGSIZEERROR);
	}
	for (ulong i = 0; i < this->size(); i++) _data->operator[](i) /= a[i];
	return *this;
}

}  // namespace LinAlgebra
