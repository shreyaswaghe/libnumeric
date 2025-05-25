#pragma once

#include <array>
#include <memory>

#include "common.hpp"

namespace LinAlgebra {

// Base template: dynamic allocation
template <typename T, ulong __size = 0>
struct memoryBlock {
	ulong _size = 0;
	std::unique_ptr<T[]> _data = nullptr;

	memoryBlock() = default;

	explicit memoryBlock(ulong size) : _size(size) {
		if (size == 0) throw std::runtime_error(LINALGSIZEERROR);
		_data = std::make_unique<T[]>(size);
	}

	bool alloc(ulong size) {
		if (size == 0 || _size != 0 || _data != nullptr) return false;
		_data = std::make_unique<T[]>(size);
		_size = size;
		return true;
	}

	T& operator[](ulong i) { return _data[i]; }
	const T& operator[](ulong i) const { return _data[i]; }

	T* raw() { return _data.get(); }
	const T* raw() const { return _data.get(); }
};

// Specialization: static allocation
template <typename T, ulong __size>
	requires(__size > 0)
struct memoryBlock<T, __size> {
	static constexpr ulong _size = __size;
	std::array<T, __size> _data{};

	memoryBlock() = default;

	// No alloc() needed: statically allocated

	T& operator[](ulong i) { return _data[i]; }
	const T& operator[](ulong i) const { return _data[i]; }

	T* raw() { return _data.data(); }
	const T* raw() const { return _data.data(); }
};

}  // namespace LinAlgebra
