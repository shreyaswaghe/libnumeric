#pragma once

#include <stdexcept>

#include "matrixBase.hpp"
#include "operators.hpp"

namespace LinAlgebra {

#define _LINALG_MATMAT_SIZECHECK                      \
	if (a.rows() != b.rows() || a.cols() != b.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_SELFMAT_SIZECHECK                             \
	if (this->rows() != a.rows() || this->cols() != a.cols()) \
		throw std::runtime_error(LINALGSIZEERROR);

#define _LINALG_MATVEC_COMPATIBILITY_CHECK             \
	if (A.cols() != a.size() || out.size() < A.rows()) \
		throw std::runtime_error(LINALGSIZEERROR);

// NO-BLAS ATOMIC OPERATIONS
template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void add(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] += b[i];
}

template <typename T, ulong ra, ulong ca>
void add(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] += b;
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void sub(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] -= b[i];
}

template <typename T, ulong ra, ulong ca>
void sub(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] -= b;
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void subLeft(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b[i] - a[i];
}

template <typename T, ulong ra, ulong ca>
void subLeft(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b - a[i];
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void mul(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] *= b[i];
}

template <typename T, ulong ra, ulong ca>
void mul(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] *= b;
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void div(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] /= b[i];
}

template <typename T, ulong ra, ulong ca>
void div(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] /= b;
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void divLeft(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b[i] / a[i];
}

template <typename T, ulong ra, ulong ca>
void divLeft(Matrix<T, ra, ca>& a, T b) {
#pragma clang loop vectorize(enable) unroll(enable)
	for (ulong i = 0; i < a.size(); i++) a[i] = b / a[i];
}

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
void copyTo(Matrix<T, ra, ca>& a, const Matrix<T, rb, cb>& b) {
	std::memcpy(a(), b(), a.size() * sizeof(T));
}

template <typename T, ulong rA, ulong cA, ulong sa, ulong sb>
void matvecOp(const Matrix<T, rA, cA>& A, const Vector<T, sa>& a,
			  Vector<T, sb>& out) {
	_LINALG_MATVEC_COMPATIBILITY_CHECK;
	out.setZero();
	for (ulong iCol = 0; iCol < A.cols(); iCol++) {
		const T vj = a[iCol];
		const T* A_jcol = A(0, iCol);
		for (ulong iRow = 0; iRow < A.rows(); iRow++) {
			out[iRow] += A_jcol[iRow] * vj;
		}
	}
}

template <typename T, ulong ra, ulong ca, OPType _op, OPType _destop>
struct SMOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		add(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		add(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		sub(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		sub(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		sub(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		add(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-1.0 / exp.scalar, exp.mat[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(-exp.scalar, 1.0 / exp.mat[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], dest[i] * exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], exp.mat[i], -dest[i] * exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] = std::fma(dest[i], -exp.mat[i], dest[i] * exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		mul(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		div(dest, exp.scalar);
		mul(dest, exp.mat);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		div(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] + exp.scalar;
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.mat[i] - exp.scalar;
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.mat.size(); i++)
			dest[i] /= exp.scalar - exp.mat[i];
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		div(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		mul(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

// WHERE RESULT IS SET TO DEST
template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Add, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Add>& exp) {
		copyTo(dest, exp.mat);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Sub, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Sub>& exp) {
		copyTo(dest, exp.mat);
		sub(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::SubLeft, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::SubLeft>& exp) {
		copyTo(dest, exp.mat);
		mul(dest, -1.0);
		add(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Mul, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Mul>& exp) {
		copyTo(dest, exp.mat);
		mul(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::Div, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::Div>& exp) {
		copyTo(dest, exp.mat);
		div(dest, exp.scalar);
	};
};

template <typename T, ulong ra, ulong ca>
struct SMOPImpl<T, ra, ca, OPType::DivLeft, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const SMOP<T, ra, ca, OPType::DivLeft>& exp) {
		dest.setOne();
		mul(dest, exp.scalar);
		div(dest, exp.mat);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb, OPType _op,
		  OPType _destop>
struct MMOPImpl;

// WHERE RESULT IS ADDED TO DEST

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		add(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		add(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Add> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS SUBTRACTED FROM DEST
template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		sub(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		sub(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-exp.lhs[i], exp.rhs[i], dest[i]);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Sub> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(-1.0 / exp.rhs[i], exp.lhs[i], dest[i]);
	};
};

// WHERE RESULT IS MULTIPLIED INTO DEST
template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], dest[i] * exp.rhs[i]);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] = std::fma(dest[i], exp.lhs[i], -dest[i] * exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		mul(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Mul> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		div(dest, exp.rhs);
		mul(dest, exp.lhs);
	};
};

// WHERE DEST IS DIVIDED BY RESULT
template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] + exp.rhs[i];
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
#pragma clang loop vectorize(enable) unroll(enable)
		for (ulong i = 0; i < exp.lhs.size(); i++)
			dest[i] /= exp.lhs[i] - exp.rhs[i];
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		div(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Div> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		mul(dest, exp.rhs);
		div(dest, exp.lhs);
	};
};

// WHERE RESULT IS ASSIGNED TO DEST
template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Add, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Add>& exp) {
		copyTo(dest, exp.lhs);
		add(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Sub, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Sub>& exp) {
		copyTo(dest, exp.lhs);
		sub(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Mul, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Mul>& exp) {
		copyTo(dest, exp.lhs);
		mul(dest, exp.rhs);
	};
};

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb>
struct MMOPImpl<T, ra, ca, rb, cb, OPType::Div, OPType::Assign> {
	template <ulong __rows, ulong __cols>
	static void apply(Matrix<T, __rows, __cols>& dest,
					  const MMOP<T, ra, ca, rb, cb, OPType::Div>& exp) {
		copyTo(dest, exp.lhs);
		div(dest, exp.rhs);
	};
};

// Unary Operators
template <typename T, ulong ra, ulong ca>
auto operator+(const Matrix<T, ra, ca>& a) {
	return a;
}

template <typename T, ulong ra, ulong ca>
auto operator-(const Matrix<T, ra, ca>& a) {
	return -T(1.0) * a;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong sa, ulong sb>
void Matrix<T, __rows, __cols>::matvec(const Vector<T, sa>& a,
									   Vector<T, sb>& out) const {
	matvecOp(*this, a, out);
};

// DISPATCH TO COMPUTATIONAL KERNELS ON ASSIGNMENT
// ADDITION
template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator+=(
	const Matrix<T, ra, ca>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	add(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator+=(const T& a) {
	add(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator+=(
	const SMOP<T, ra, ca, _op>& exp) {
	SMOPImpl<T, ra, ca, _op, OPType::Add>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator+=(
	const MMOP<T, ra, ca, rb, cb, _op>& exp) {
	MMOPImpl<T, ra, ca, rb, cb, _op, OPType::Add>::apply(*this, exp);
	return *this;
}

// SUBTRACTION
template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator-=(
	const Matrix<T, ra, ca>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	sub(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator-=(const T& a) {
	sub(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator-=(
	const SMOP<T, ra, ca, _op>& exp) {
	SMOPImpl<T, ra, ca, _op, OPType::Sub>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator-=(
	const MMOP<T, ra, ca, rb, cb, _op>& exp) {
	MMOPImpl<T, ra, ca, rb, cb, _op, OPType::Sub>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator*=(
	const Matrix<T, ra, ca>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	mul(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator*=(const T& a) {
	mul(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator*=(
	const SMOP<T, ra, ca, _op>& exp) {
	SMOPImpl<T, ra, ca, _op, OPType::Mul>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator*=(
	const MMOP<T, ra, ca, rb, cb, _op>& exp) {
	MMOPImpl<T, ra, ca, rb, cb, _op, OPType::Mul>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator/=(
	const Matrix<T, ra, ca>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	div(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator/=(const T& a) {
	div(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator/=(
	const SMOP<T, ra, ca, _op>& exp) {
	SMOPImpl<T, ra, ca, _op, OPType::Div>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator/=(
	const MMOP<T, ra, ca, rb, cb, _op>& exp) {
	MMOPImpl<T, ra, ca, rb, cb, _op, OPType::Div>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator=(
	const Matrix<T, __rows, __cols>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	copyTo(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator=(
	const Matrix<T, ra, ca>& a) {
	_LINALG_SELFMAT_SIZECHECK;
	copyTo(*this, a);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator=(
	const SMOP<T, ra, ca, _op>& exp) {
	SMOPImpl<T, ra, ca, _op, OPType::Assign>::apply(*this, exp);
	return *this;
}

template <typename T, ulong __rows, ulong __cols>
template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
Matrix<T, __rows, __cols>& Matrix<T, __rows, __cols>::operator=(
	const MMOP<T, ra, ca, rb, cb, _op>& exp) {
	MMOPImpl<T, ra, ca, rb, cb, _op, OPType::Assign>::apply(*this, exp);
	return *this;
}

#undef _LINALG_MATMAT_SIZECHECK

}  // namespace LinAlgebra
