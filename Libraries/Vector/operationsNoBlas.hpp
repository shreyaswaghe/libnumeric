#pragma once

#include "operators.hpp"
#include "vectorBase.hpp"

namespace LinAlgebra {

#define _LINALG_VECVEC_SIZECHECK \
	if (a.size() != b.size()) throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFVEC_SIZECHECK \
	if (this->size() != a.size()) throw std::runtime_error(LINALGSIZEERROR);

// NO-BLAS ATOMIC OPERATIONS
template <typename T, ulong sa, ulong sb>
void add(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] += b[i];
}

template <typename T, ulong sa>
void add(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] += b;
}

template <typename T, ulong sa, ulong sb>
void sub(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] -= b[i];
}

template <typename T, ulong sa>
void sub(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] -= b;
}

template <typename T, ulong sa, ulong sb>
void subLeft(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b[i] - a[i];
}

template <typename T, ulong sa>
void subLeft(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b - a[i];
}

template <typename T, ulong sa, ulong sb>
void mul(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] *= b[i];
}

template <typename T, ulong sa>
void mul(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] *= b;
}

template <typename T, ulong sa, ulong sb>
void div(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] /= b[i];
}

template <typename T, ulong sa>
void div(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] /= b;
}

template <typename T, ulong sa, ulong sb>
void divLeft(Vector<T, sa>& a, const Vector<T, sb>& b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b[i] / a[i];
}

template <typename T, ulong sa>
void divLeft(Vector<T, sa>& a, T b) {
#pragma clang loop vectorize(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b / a[i];
}

template <typename T, ulong sa, ulong sb>
void copyTo(Vector<T, sa>& a, const Vector<T, sb>& b) {
	std::memcpy(a(), b(), a.size() * sizeof(T));
}

template <typename T, ulong sa, OPType _op, OPType _destop>
struct SVOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		add(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		sub(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		sub(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		add(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.vec[i], dest[i]);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.vec[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], dest[i] * exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], exp.vec[i], -dest[i] * exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] = std::fma(dest[i], -exp.vec[i], dest[i] * exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.vec);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		div(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] + exp.scalar;
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.vec[i] - exp.scalar;
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.vec.size(); i++)
			dest[i] /= exp.scalar - exp.vec[i];
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		mul(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Add, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Add>& exp) {
		copyTo(dest, exp.vec);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Sub, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Sub>& exp) {
		copyTo(dest, exp.vec);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::SubLeft, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::SubLeft>& exp) {
		copyTo(dest, exp.vec);
		mul(dest, -1.0);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Mul, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Mul>& exp) {
		copyTo(dest, exp.vec);
		mul(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::Div, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::Div>& exp) {
		copyTo(dest, exp.vec);
		div(dest, exp.scalar);
	};
};

template <typename T, ulong sa>
struct SVOPImpl<T, sa, OPType::DivLeft, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const SVOP<T, sa, OPType::DivLeft>& exp) {
		dest.setOne();
		mul(dest, exp.scalar);
		div(dest, exp.vec);
	};
};

template <typename T, ulong sa, ulong sb, OPType _op, OPType _destop>
struct VVOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		add(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		add(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Add> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		sub(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		sub(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Sub> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Mul> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Div> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Add, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Add>& exp) {
		copyTo(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Sub, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Sub>& exp) {
		copyTo(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Mul, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Mul>& exp) {
		copyTo(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, ulong sa, ulong sb>
struct VVOPImpl<T, sa, sb, OPType::Div, OPType::Assign> {
	template <ulong __size>
	static void apply(Vector<T, __size>& dest,
					  const VVOP<T, sa, sb, OPType::Div>& exp) {
		copyTo(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

// Unary Operators
template <typename T, ulong sa>
auto operator+(const Vector<T, sa>& a) {
	return a;
}

template <typename T, ulong sa>
auto operator-(const Vector<T, sa>& a) {
	return -T(1.0) * a;
}

// Dot Product

template <typename T, ulong sa, ulong sb>
T dotOp(const Vector<T, sa>& a, const Vector<T, sb>& b) {
	_LINALG_VECVEC_SIZECHECK;
	T result = T(0.0);
	for (ulong i = 0; i < a.size(); i++) result = std::fma(a[i], b[i], result);
	return result;
}

template <typename T, ulong __size>
template <ulong sa>
T Vector<T, __size>::dot(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	return dotOp(*this, a);
}

// DISPATCH TO COMPUTATIONAL KERNELS ON ASSIGNMENT
// ADDITION
template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator+=(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	add(*this, a);
	return *this;
}

template <typename T, ulong __size>
Vector<T, __size>& Vector<T, __size>::operator+=(const T& a) {
	add(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator+=(const SVOP<T, sa, _op>& exp) {
	SVOPImpl<T, sa, _op, OPType::Add>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, ulong sb, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator+=(
	const VVOP<T, sa, sb, _op>& exp) {
	VVOPImpl<T, sa, sb, _op, OPType::Add>::apply(*this, exp);
	return *this;
}

// SUBTRACTION
template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator-=(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	sub(*this, a);
	return *this;
}

template <typename T, ulong __size>
Vector<T, __size>& Vector<T, __size>::operator-=(const T& a) {
	sub(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator-=(const SVOP<T, sa, _op>& exp) {
	SVOPImpl<T, sa, _op, OPType::Sub>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, ulong sb, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator-=(
	const VVOP<T, sa, sb, _op>& exp) {
	VVOPImpl<T, sa, sb, _op, OPType::Sub>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator*=(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	mul(*this, a);
	return *this;
}

template <typename T, ulong __size>
Vector<T, __size>& Vector<T, __size>::operator*=(const T& a) {
	mul(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator*=(const SVOP<T, sa, _op>& exp) {
	SVOPImpl<T, sa, _op, OPType::Mul>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, ulong sb, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator*=(
	const VVOP<T, sa, sb, _op>& exp) {
	VVOPImpl<T, sa, sb, _op, OPType::Mul>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator/=(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	div(*this, a);
	return *this;
}

template <typename T, ulong __size>
Vector<T, __size>& Vector<T, __size>::operator/=(const T& a) {
	div(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator/=(const SVOP<T, sa, _op>& exp) {
	SVOPImpl<T, sa, _op, OPType::Div>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, ulong sb, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator/=(
	const VVOP<T, sa, sb, _op>& exp) {
	VVOPImpl<T, sa, sb, _op, OPType::Div>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
Vector<T, __size>& Vector<T, __size>::operator=(const Vector<T, __size>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	copyTo(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa>
Vector<T, __size>& Vector<T, __size>::operator=(const Vector<T, sa>& a) {
	_LINALG_SELFVEC_SIZECHECK;
	copyTo(*this, a);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator=(const SVOP<T, sa, _op>& exp) {
	SVOPImpl<T, sa, _op, OPType::Assign>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __size>
template <ulong sa, ulong sb, OPType _op>
Vector<T, __size>& Vector<T, __size>::operator=(
	const VVOP<T, sa, sb, _op>& exp) {
	VVOPImpl<T, sa, sb, _op, OPType::Assign>::apply(*this, exp);
	return *this;
}

#undef _LINALG_VECVEC_SIZECHECK
#undef _LINALG_SELFVEC_SIZECHECK

}  // namespace LinAlgebra
