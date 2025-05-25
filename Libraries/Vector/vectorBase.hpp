#pragma once

#include "common.hpp"
#include "memory.hpp"

namespace LinAlgebra {

enum class OPType;

template <typename T, ulong sa, OPType _op>
struct SVOP;

template <typename T, ulong sa, ulong sb, OPType _op>
struct VVOP;

template <typename T = real, ulong __size = 0>
class Vector {
   protected:
	ulong _size = __size;
	std::shared_ptr<memoryBlock<T, __size>> _data = nullptr;

	bool alloc(ulong size) {
		if (isAlloc()) return false;
		_size = size;
		_data = std::make_shared<memoryBlock<T>>();
		_data->alloc(size);
		return true;
	}

   public:
	Vector() {
		if constexpr (__size > 0) {
			_data = std::make_shared<memoryBlock<T, __size>>();
		}
	}

	explicit Vector(ulong size) {
		if constexpr (__size == 0)
			alloc(size);
		else
			_data = std::make_shared<memoryBlock<T, __size>>();
	}

	inline bool isAlloc() const { return static_cast<bool>(_data); }
	inline ulong size() const { return __size > 0 ? __size : _size; }

	Vector<T, __size> copy() const {
		const ulong size = this->size();
		Vector<T, __size> cpy(size);
		std::memcpy(cpy(), this->operator()(), sizeof(T) * this->size());
		return cpy;
	}

	Vector<T, __size>& operator=(const Vector<T, __size>& a);
	template <ulong sa>
	Vector<T, __size>& operator=(const Vector<T, sa>& a);
	template <ulong sa, OPType _op>
	Vector<T, __size>& operator=(const SVOP<T, sa, _op>& exp);
	template <ulong sa, ulong sb, OPType _op>
	Vector<T, __size>& operator=(const VVOP<T, sa, sb, _op>& exp);

	inline T* operator()() { return _data->raw(); }
	inline const T* operator()() const { return _data->raw(); }

	inline T& operator[](ulong i) { return _data->operator[](i); }
	inline const T& operator[](ulong i) const { return _data->operator[](i); }

	template <ulong sa>
	Vector<T, __size>& operator+=(const Vector<T, sa>& a);
	Vector<T, __size>& operator+=(const T& a);
	template <ulong sa, OPType _op>
	Vector<T, __size>& operator+=(const SVOP<T, sa, _op>& a);
	template <ulong sa, ulong sb, OPType _op>
	Vector<T, __size>& operator+=(const VVOP<T, sa, sb, _op>& a);

	template <ulong sa>
	Vector<T, __size>& operator-=(const Vector<T, sa>& a);
	Vector<T, __size>& operator-=(const T& a);
	template <ulong sa, OPType _op>
	Vector<T, __size>& operator-=(const SVOP<T, sa, _op>& a);
	template <ulong sa, ulong sb, OPType _op>
	Vector<T, __size>& operator-=(const VVOP<T, sa, sb, _op>& a);

	template <ulong sa>
	Vector<T, __size>& operator*=(const Vector<T, sa>& a);
	Vector<T, __size>& operator*=(const T& a);
	template <ulong sa, OPType _op>
	Vector<T, __size>& operator*=(const SVOP<T, sa, _op>& a);
	template <ulong sa, ulong sb, OPType _op>
	Vector<T, __size>& operator*=(const VVOP<T, sa, sb, _op>& a);

	template <ulong sa>
	Vector<T, __size>& operator/=(const Vector<T, sa>& a);
	Vector<T, __size>& operator/=(const T& a);
	template <ulong sa, OPType _op>
	Vector<T, __size>& operator/=(const SVOP<T, sa, _op>& a);
	template <ulong sa, ulong sb, OPType _op>
	Vector<T, __size>& operator/=(const VVOP<T, sa, sb, _op>& a);

	template <ulong sa>
	T dot(const Vector<T, sa>& a);

	void setZero() { std::memset(_data->raw(), 0, sizeof(T) * this->size()); }
	void setOne() { std::fill_n(_data->raw(), this->size(), T(1.0)); }
};

typedef Vector<real, 2> Vector2;
typedef Vector<real, 3> Vector3;
typedef Vector<real, 4> Vector4;
typedef Vector<real, 5> Vector5;
typedef Vector<real, 6> Vector6;

typedef Vector<single, 2> FVector2;
typedef Vector<single, 3> FVector3;
typedef Vector<single, 4> FVector4;
typedef Vector<single, 5> FVector5;
typedef Vector<single, 6> FVector6;

}  // namespace LinAlgebra
