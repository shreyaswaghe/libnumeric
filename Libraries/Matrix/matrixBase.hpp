#pragma once

#include "Libraries/Vector/vectorBase.hpp"

namespace LinAlgebra {

template <typename T, ulong ra, ulong ca, OPType _op>
struct SMOP;

template <typename T, ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
struct MMOP;

template <typename T = real, ulong __rows = 0, ulong __cols = 0>
class Matrix {
   protected:
	ulong _rows = __rows;
	ulong _cols = __cols;
	ulong _size = __rows * __cols;
	std::shared_ptr<memoryBlock<T, __rows * __cols>> _data = nullptr;

	static constexpr ulong comptimeRows = __rows;
	static constexpr ulong comptimeCols = __cols;

   public:
	Matrix() {
		if constexpr (__rows * __cols > 0)
			_data = std::make_shared<memoryBlock<T, __rows * __cols>>();
	}

	explicit Matrix(ulong rows, ulong cols) {
		if constexpr (__rows * __cols == 0)
			alloc(rows, cols);
		else
			_data = std::make_shared<memoryBlock<T, __rows * __cols>>();
	}

	Matrix(const Matrix<T, __rows, __cols>& a) {
		if constexpr (__rows * __cols == 0)
			alloc(a.rows(), a.cols());
		else
			_data = std::make_shared<memoryBlock<T, __rows * __cols>>();
		*this = a;
	}

	bool alloc(ulong rows, ulong cols) {
		if constexpr (__rows * __cols > 0) return false;
		if (isAlloc()) return false;
		_rows = rows;
		_cols = cols;
		_size = rows * cols;
		_data = std::make_shared<memoryBlock<T, __rows * __cols>>();
		_data->alloc(_size);
		return true;
	}

	Matrix(Matrix<T, __rows, __cols>&& a)
		: _rows(a._rows), _cols(a._cols), _size(a._size), _data(a._data) {}

	inline bool isAlloc() const { return static_cast<bool>(_data); }
	inline ulong rows() const { return __rows > 0 ? __rows : _rows; }
	inline ulong cols() const { return __cols > 0 ? __cols : _cols; }
	inline ulong size() const {
		return __rows * __cols > 0 ? __rows * __cols : _size;
	}
	inline ulong ldim() const { return _rows; }
	inline ulong idx(ulong i, ulong j) const { return j * _rows + i; }

	Matrix<T, __rows, __cols> copy() const {
		Matrix<T, __rows, __cols> cpy(_rows, _cols);
		std::memcpy(cpy(), this->operator()(), sizeof(T) * this->size());
		return std::move(cpy);
	}

	inline T* operator()() { return _data->raw(); }
	inline const T* operator()() const { return _data->raw(); }

	inline T* operator()(ulong i) { return _data->raw() + i; }
	inline const T* operator()(ulong i) const { return _data->raw() + i; }

	inline T& operator()(ulong i, ulong j) {
		return *(_data->raw() + idx(i, j));
	}
	inline const T& operator()(ulong i, ulong j) const {
		return *(_data->raw() + idx(i, j));
	}

	inline T& operator[](ulong i) { return _data->operator[](i); }
	inline const T& operator[](ulong i) const { return _data->operator[](i); }

	Matrix<T, __rows, __cols>& operator=(const Matrix<T, __rows, __cols>& a);
	template <ulong ra, ulong ca>
	Matrix<T, __rows, __cols>& operator=(const Matrix<T, ra, ca>& a);
	template <ulong ra, ulong ca, OPType _op>
	Matrix<T, __rows, __cols>& operator=(const SMOP<T, ra, ca, _op>& exp);
	template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
	Matrix<T, __rows, __cols>& operator=(
		const MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <ulong ra, ulong ca>
	Matrix<T, __rows, __cols>& operator+=(const Matrix<T, ra, ca>& a);
	Matrix<T, __rows, __cols>& operator+=(const T& a);
	template <ulong ra, ulong ca, OPType _op>
	Matrix<T, __rows, __cols>& operator+=(const SMOP<T, ra, ca, _op>& exp);
	template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
	Matrix<T, __rows, __cols>& operator+=(
		const MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <ulong ra, ulong ca>
	Matrix<T, __rows, __cols>& operator-=(const Matrix<T, ra, ca>& a);
	Matrix<T, __rows, __cols>& operator-=(const T& a);
	template <ulong ra, ulong ca, OPType _op>
	Matrix<T, __rows, __cols>& operator-=(const SMOP<T, ra, ca, _op>& exp);
	template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
	Matrix<T, __rows, __cols>& operator-=(
		const MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <ulong ra, ulong ca>
	Matrix<T, __rows, __cols>& operator*=(const Matrix<T, ra, ca>& a);
	Matrix<T, __rows, __cols>& operator*=(const T& a);
	template <ulong ra, ulong ca, OPType _op>
	Matrix<T, __rows, __cols>& operator*=(const SMOP<T, ra, ca, _op>& exp);
	template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
	Matrix<T, __rows, __cols>& operator*=(
		const MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <ulong ra, ulong ca>
	Matrix<T, __rows, __cols>& operator/=(const Matrix<T, ra, ca>& a);
	Matrix<T, __rows, __cols>& operator/=(const T& a);
	template <ulong ra, ulong ca, OPType _op>
	Matrix<T, __rows, __cols>& operator/=(const SMOP<T, ra, ca, _op>& exp);
	template <ulong ra, ulong ca, ulong rb, ulong cb, OPType _op>
	Matrix<T, __rows, __cols>& operator/=(
		const MMOP<T, ra, ca, rb, cb, _op>& exp);

	template <ulong sa, ulong sb>
	void matvec(const Vector<T, sa>& a, Vector<T, sb>& out) const;

	template <bool U, ulong sa, ulong sb>
	void trsv(const Vector<T, sa>& a, Vector<T, sb>& out) const;

	void setZero() { std::memset(_data->raw(), 0, sizeof(T) * this->size()); }
	void setOne() { std::fill_n(_data->raw(), this->size(), T(1.0)); }
};

typedef Matrix<real, 1, 1> Matrix11;
typedef Matrix<real, 1, 2> Matrix12;
typedef Matrix<real, 1, 3> Matrix13;
typedef Matrix<real, 1, 4> Matrix14;
typedef Matrix<real, 1, 5> Matrix15;
typedef Matrix<real, 1, 6> Matrix16;

typedef Matrix<real, 2, 1> Matrix21;
typedef Matrix<real, 2, 2> Matrix22;
typedef Matrix<real, 2, 3> Matrix23;
typedef Matrix<real, 2, 4> Matrix24;
typedef Matrix<real, 2, 5> Matrix25;
typedef Matrix<real, 2, 6> Matrix26;

typedef Matrix<real, 3, 1> Matrix31;
typedef Matrix<real, 3, 2> Matrix32;
typedef Matrix<real, 3, 3> Matrix33;
typedef Matrix<real, 3, 4> Matrix34;
typedef Matrix<real, 3, 5> Matrix35;
typedef Matrix<real, 3, 6> Matrix36;

typedef Matrix<real, 4, 1> Matrix41;
typedef Matrix<real, 4, 2> Matrix42;
typedef Matrix<real, 4, 3> Matrix43;
typedef Matrix<real, 4, 4> Matrix44;
typedef Matrix<real, 4, 5> Matrix45;
typedef Matrix<real, 4, 6> Matrix46;

typedef Matrix<real, 5, 1> Matrix51;
typedef Matrix<real, 5, 2> Matrix52;
typedef Matrix<real, 5, 3> Matrix53;
typedef Matrix<real, 5, 4> Matrix54;
typedef Matrix<real, 5, 5> Matrix55;
typedef Matrix<real, 5, 6> Matrix56;

typedef Matrix<real, 6, 1> Matrix61;
typedef Matrix<real, 6, 2> Matrix62;
typedef Matrix<real, 6, 3> Matrix63;
typedef Matrix<real, 6, 4> Matrix64;
typedef Matrix<real, 6, 5> Matrix65;
typedef Matrix<real, 6, 6> Matrix66;

typedef Matrix<single, 1, 1> MatrixF11;
typedef Matrix<single, 1, 2> MatrixF12;
typedef Matrix<single, 1, 3> MatrixF13;
typedef Matrix<single, 1, 4> MatrixF14;
typedef Matrix<single, 1, 5> MatrixF15;
typedef Matrix<single, 1, 6> MatrixF16;

typedef Matrix<single, 2, 1> MatrixF21;
typedef Matrix<single, 2, 2> MatrixF22;
typedef Matrix<single, 2, 3> MatrixF23;
typedef Matrix<single, 2, 4> MatrixF24;
typedef Matrix<single, 2, 5> MatrixF25;
typedef Matrix<single, 2, 6> MatrixF26;

typedef Matrix<single, 3, 1> MatrixF31;
typedef Matrix<single, 3, 2> MatrixF32;
typedef Matrix<single, 3, 3> MatrixF33;
typedef Matrix<single, 3, 4> MatrixF34;
typedef Matrix<single, 3, 5> MatrixF35;
typedef Matrix<single, 3, 6> MatrixF36;

typedef Matrix<single, 4, 1> MatrixF41;
typedef Matrix<single, 4, 2> MatrixF42;
typedef Matrix<single, 4, 3> MatrixF43;
typedef Matrix<single, 4, 4> MatrixF44;
typedef Matrix<single, 4, 5> MatrixF45;
typedef Matrix<single, 4, 6> MatrixF46;

typedef Matrix<single, 5, 1> MatrixF51;
typedef Matrix<single, 5, 2> MatrixF52;
typedef Matrix<single, 5, 3> MatrixF53;
typedef Matrix<single, 5, 4> MatrixF54;
typedef Matrix<single, 5, 5> MatrixF55;
typedef Matrix<single, 5, 6> MatrixF56;

typedef Matrix<single, 6, 1> MatrixF61;
typedef Matrix<single, 6, 2> MatrixF62;
typedef Matrix<single, 6, 3> MatrixF63;
typedef Matrix<single, 6, 4> MatrixF64;
typedef Matrix<single, 6, 5> MatrixF65;
typedef Matrix<single, 6, 6> MatrixF66;

}  // namespace LinAlgebra
