#include <concepts>
#include <stdexcept>
#include <type_traits>

#include "common.hpp"
#include "vectorBase.hpp"

namespace LinAlgebra {

// typing concepts
template <typename T, typename Derived>
concept IsVectorExpr = std::derived_from<Derived, VectorExpression<T, Derived>>;

template <typename T, typename U>
concept IsScalar = std::is_convertible_v<T, U> && !IsVectorExpr<T, U>;

template <typename T, typename Lhs, typename Rhs>
concept IsBinaryVectorExpression = IsVectorExpr<T, Lhs> && IsVectorExpr<T, Rhs>;

template <typename T, typename Lhs, typename Rhs>
concept IsLooseBinaryVectorExpression =
	IsBinaryVectorExpression<T, Lhs, Rhs> || IsVectorExpr<T, Lhs> ||
	IsVectorExpr<T, Rhs>;

// operator wrappers

template <typename T, typename Lhs, typename Rhs>
struct VectorAdd : VectorExpression<T, VectorAdd<T, Lhs, Rhs>> {
	Lhs& lhs;
	Rhs& rhs;

	VectorAdd(const Lhs& lhs, const Rhs& rhs) : lhs(lhs), rhs(rhs) {
		if constexpr (IsBinaryVectorExpression<T, Lhs, Rhs>) {
			if (lhs.size() != rhs.size())
				throw std::runtime_error(LINALGSIZEERROR);
		}
	}

	inline const T operator[](ulong i) const {
		if constexpr (IsBinaryVectorExpression<T, Lhs, Rhs>) {
			return lhs[i] + rhs[i];
		} else if constexpr (IsScalar<T, Lhs> && IsVectorExpr<T, Rhs>) {
			return lhs + rhs[i];
		} else if constexpr (IsScalar<T, Rhs> && IsVectorExpr<T, Lhs>) {
			return lhs[i] + rhs;
		} else {
			return lhs + rhs;
		}
	}
	inline ulong size() {
		if constexpr (!IsScalar<T, Lhs>) {
			return lhs.size();
		} else if constexpr (!IsScalar<T, Rhs>) {
			return rhs.size();
		} else {
			return 0;
		}
	}

	inline static constexpr bool leaf() {
		if constexpr (IsBinaryVectorExpression<T, Lhs, Rhs>) {
			return false;
		} else {
			return true;
		}
	}

	template <ulong sa>
	inline auto evalTo(Vector<T, sa>& a);
};

template <typename T, typename Lhs, typename Rhs>
	requires IsLooseBinaryVectorExpression<T, Lhs, Rhs>
struct VectorSub : VectorExpression<T, VectorSub<T, Lhs, Rhs>> {
	const Lhs& lhs;
	const Rhs& rhs;

	VectorSub(const Lhs& lhs, const Rhs& rhs) : lhs(lhs), rhs(rhs) {}

	inline T operator[](ulong i) { return lhs[i] - rhs[i]; }
	inline const T operator[](ulong i) const { return lhs[i] - rhs[i]; }
	inline ulong size() { return lhs.size(); }

	template <ulong sa>
	inline auto evalTo(Vector<T, sa>& a);
};

template <typename T, typename Lhs, typename Rhs>
	requires IsLooseBinaryVectorExpression<T, Lhs, Rhs>
struct VectorMul : VectorExpression<T, VectorMul<T, Lhs, Rhs>> {
	const Lhs& lhs;
	const Rhs& rhs;

	VectorMul(const Lhs& lhs, const Rhs& rhs) : lhs(lhs), rhs(rhs) {}

	inline T operator[](ulong i) { return lhs[i] * rhs[i]; }
	inline const T operator[](ulong i) const { return lhs[i] * rhs[i]; }
	inline ulong size() { return lhs.size(); }

	template <ulong sa>
	inline auto evalTo(Vector<T, sa>& a);
};

template <typename T, typename Lhs, typename Rhs>
	requires IsLooseBinaryVectorExpression<T, Lhs, Rhs>
struct VectorDiv : VectorExpression<T, VectorDiv<T, Lhs, Rhs>> {
	const Lhs& lhs;
	const Rhs& rhs;

	VectorDiv(const Lhs& lhs, const Rhs& rhs) : lhs(lhs), rhs(rhs) {}

	inline T operator[](ulong i) { return lhs[i] * rhs[i]; }
	inline const T operator[](ulong i) const { return lhs[i] * rhs[i]; }
	inline ulong size() { return lhs.size(); }

	template <ulong sa>
	inline auto evalTo(Vector<T, sa>& a);
};

// function templates

// + Operators
template <typename T, typename Derived1>
auto operator+(const VectorExpression<T, Derived1>& a);

template <typename Lhs, typename Rhs>
auto operator+(const Lhs& a, const Rhs& b) {
	using T = decltype(a[0] + b[0]);
	return VectorAdd<T, Lhs, Rhs>(a, b);
}

// - Operators

template <typename T, typename Derived1>
auto operator-(const VectorExpression<T, Derived1>& a);

template <typename T, typename Lhs, typename Rhs>
auto operator-(const Lhs& a, const Rhs& b) {
	return VectorSub<T, Lhs, Rhs>(a, b);
}

// * Operator

template <typename T, typename Lhs, typename Rhs>
auto operator*(const Lhs& a, const Rhs& b) {
	return VectorMul<T, Lhs, Rhs>(a, b);
}

// / Operator

template <typename T, typename Lhs, typename Rhs>
auto operator/(const Lhs& a, const Rhs& b) {
	return VectorDiv<T, Lhs, Rhs>(a, b);
}

};	// namespace LinAlgebra
