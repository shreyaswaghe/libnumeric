#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../vector.hpp"
#include "vectorBase.hpp"

using namespace LinAlgebra;

constexpr double EPSILON = 1e-10;

template <typename T>
bool nearlyEqual(T a, T b, T epsilon = EPSILON) {
	return std::abs(a - b) < epsilon;
}

void test_default_construction_static() {
	Vector<double, 3> v;
	assert(v.size() == 3);
	assert(v.isAlloc());
}

void test_dynamic_construction() {
	Vector<double> v(10);
	assert(v.size() == 10);
	assert(v.isAlloc());

	Vector<double> v1;
	assert(!v1.isAlloc());
}

void test_allocation_guard() {
	Vector v(5);
	bool res = v.size() == 5;
	assert(res);
}

void test_element_access() {
	Vector<double> v(4);
	for (ulong i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i);
	for (ulong i = 0; i < v.size(); ++i) assert(v[i] == static_cast<double>(i));
}

void test_raw_pointer_access() {
	Vector<double> v(3);
	double* ptr = v();
	ptr[0] = 1.0;
	ptr[1] = 2.0;
	ptr[2] = 3.0;
	assert(v[0] == 1.0 && v[1] == 2.0 && v[2] == 3.0);
}

void test_vector_addition() {
	Vector<double> a(3), b(3);
	a[0] = 1;
	a[1] = 2;
	a[2] = 3;
	b[0] = 4;
	b[1] = 5;
	b[2] = 6;
	Vector c = a + 4.0;
	assert(c.size() == 3);
	assert(c[0] == 5 && c[1] == 6 && c[2] == 7);
}

void test_vector_subtraction() {
	Vector<double> a(3), b(3);
	a[0] = 5;
	a[1] = 7;
	a[2] = 9;
	b[0] = 1;
	b[1] = 2;
	b[2] = 3;
	auto c = a - b;
	assert(c[0] == 4 && c[1] == 5 && c[2] == 6);
}

void test_vector_multiplication() {
	Vector<double> a(3), b(3);
	a[0] = 2;
	a[1] = 3;
	a[2] = 4;
	b[0] = 5;
	b[1] = 6;
	b[2] = 7;
	auto c = a * b;
	assert(c[0] == 10 && c[1] == 18 && c[2] == 28);
}

void test_vector_division() {
	Vector<double> a(3), b(3);
	a[0] = 8;
	a[1] = 9;
	a[2] = 10;
	b[0] = 2;
	b[1] = 3;
	b[2] = 5;
	auto c = a / b;
	assert(c[0] == 4 && c[1] == 3 && c[2] == 2);
}

void test_scalar_operations() {
	Vector<double> a(3);
	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;

	auto add = a + 2.0;
	auto sub = a - 1.0;
	auto mul = a * 3.0;
	auto div = a / 2.0;

	assert(add[0] == 3.0 && add[1] == 4.0 && add[2] == 5.0);
	assert(sub[0] == 0.0 && sub[1] == 1.0 && sub[2] == 2.0);
	assert(mul[0] == 3.0 && mul[1] == 6.0 && mul[2] == 9.0);
	assert(div[0] == 0.5 && div[1] == 1.0 && div[2] == 1.5);
}

void test_unary_operators() {
	Vector<double> a(2);
	a[0] = 3.0;
	a[1] = -4.0;

	auto b = -a;
	assert(b[0] == -3.0);
	assert(b[1] == 4.0);

	auto c = +a;
	assert(c[0] == 3.0);
	assert(c[1] == -4.0);
}

void test_typedefs() {
	Vector2 v2;
	Vector3 v3;
	FVector6 vf6;

	assert(v2.size() == 2);
	assert(v3.size() == 3);
	assert(vf6.size() == 6);
}

void test_size_mismatch_exception() {
	Vector<double> a(2), b(3);
	bool threw = false;
	try {
		auto _ = a + b;
	} catch (const std::runtime_error&) {
		threw = true;
	}
	assert(threw);

	threw = false;
	Vector3 d;
	Vector2 e;
	try {
		auto _ = d + e;
	} catch (const std::runtime_error&) {
		threw = true;
	}
	assert(threw);
}

void test_self_operators() {
	Vector<double> a(4), b(4);
	a[0] = 1.0;
	a[1] = 1.1;
	a[2] = 1.2;
	a[3] = 1.3;

	b = a;
	a += b;

	assert(a[0] == 2 * 1.0 && a[2] == 2 * 1.2);
	assert(b[0] == 1.0 && b[1] == 1.1);
}

void test_copy() {
	Vector<double> a(4), b(4);
	a[0] = 1.0;
	a[1] = 1.1;
	a[2] = 1.2;
	a[3] = 1.3;

	assert(b() != a());
	b = a;
	assert(b[0] == 1.0 && b[1] == 1.1);
	assert(b() != a());

	b = a.copy();
	a += b;

	assert(a[0] == 2 * 1.0 && a[2] == 2 * 1.2);
	assert(b[0] == 1.0 && b[1] == 1.1);
}

void test_time() {
	Vector<double> a(500);
	a.setZero();
	a += 1.0;
	a += 0.838298398;
}

int main() {
	test_default_construction_static();
	test_dynamic_construction();
	test_allocation_guard();
	test_element_access();
	test_raw_pointer_access();
	test_vector_addition();
	test_vector_subtraction();
	test_vector_multiplication();
	test_vector_division();
	test_scalar_operations();
	test_unary_operators();
	test_typedefs();
	test_size_mismatch_exception();
	test_self_operators();
	test_copy();
	test_time();

	std::cout << "All Vector tests passed!\n";
	return 0;
}
