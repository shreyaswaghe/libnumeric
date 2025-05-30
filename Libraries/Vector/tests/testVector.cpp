#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Libraries/Vector/vector.hpp"

using namespace LinAlgebra;

constexpr double EPSILON = 1e-10;

template <typename T>
bool nearlyEqual(T a, T b, T epsilon = EPSILON) {
	return std::abs(a - b) < epsilon;
}

// Test counter for tracking progress
int test_count = 0;
int passed_tests = 0;

#define TEST(name)                                                     \
	do {                                                               \
		std::cout << "Running " << #name << "..." << std::endl;        \
		test_count++;                                                  \
		try {                                                          \
			name();                                                    \
			passed_tests++;                                            \
			std::cout << "  ✓ PASSED" << std::endl;                    \
		} catch (const std::exception& e) {                            \
			std::cout << "  ✗ FAILED: " << e.what() << std::endl;      \
		} catch (...) {                                                \
			std::cout << "  ✗ FAILED: Unknown exception" << std::endl; \
		}                                                              \
	} while (0)

// ==================== CONSTRUCTION TESTS ====================

void test_static_construction() {
	// Test compile-time sized vectors
	Vector<double, 3> v3;
	assert(v3.size() == 3);
	assert(v3.isAlloc());

	Vector<double, 5> v5;
	assert(v5.size() == 5);
	assert(v5.isAlloc());

	// Test that default values are zero
	for (size_t i = 0; i < v3.size(); ++i) {
		assert(nearlyEqual(v3[i], 0.0));
	}
}

void test_dynamic_construction() {
	// Test runtime-sized vectors
	Vector<double> v10(10);
	assert(v10.size() == 10);
	assert(v10.isAlloc());

	Vector<double> v100(100);
	assert(v100.size() == 100);
	assert(v100.isAlloc());

	// Test unallocated vector
	Vector<double> v_empty;
	assert(!v_empty.isAlloc());
}

void test_mixed_size_construction() {
	// Test that we can create vectors of different sizes
	Vector<float, 2> vf2;
	Vector<double, 4> vd4;
	Vector<int, 6> vi6;

	assert(vf2.size() == 2);
	assert(vd4.size() == 4);
	assert(vi6.size() == 6);

	assert(vf2.isAlloc());
	assert(vd4.isAlloc());
	assert(vi6.isAlloc());
}

void test_typedefs() {
	Vector2 v2;
	assert(v2.size() == 2);
	Vector3 v3;
	assert(v3.size() == 3);
	Vector4 v4;
	assert(v4.size() == 4);
	Vector5 v5;
	assert(v5.size() == 5);
	Vector6 v6;
	assert(v6.size() == 6);

	FVector2 fv2;
	assert(fv2.size() == 2);
	FVector3 fv3;
	assert(fv3.size() == 3);
	FVector4 fv4;
	assert(fv4.size() == 4);
	FVector5 fv5;
	assert(fv5.size() == 5);
	FVector6 fv6;
	assert(fv6.size() == 6);

	// All should be allocated
	assert(v2.isAlloc() && v3.isAlloc() && v4.isAlloc());
	assert(fv2.isAlloc() && fv3.isAlloc() && fv4.isAlloc());
}

// ==================== ACCESS TESTS ====================

void test_element_access() {
	Vector<double, 4> v;

	// Test assignment and access
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = static_cast<double>(i + 1);
	}

	for (size_t i = 0; i < v.size(); ++i) {
		assert(nearlyEqual(v[i], static_cast<double>(i + 1)));
	}

	// Test const access
	const Vector<double, 4>& cv = v;
	for (size_t i = 0; i < cv.size(); ++i) {
		assert(nearlyEqual(cv[i], static_cast<double>(i + 1)));
	}
}

void test_raw_pointer_access() {
	Vector<double, 3> v;

	// Test non-const pointer access
	double* ptr = v();
	ptr[0] = 1.5;
	ptr[1] = 2.5;
	ptr[2] = 3.5;

	assert(nearlyEqual(v[0], 1.5));
	assert(nearlyEqual(v[1], 2.5));
	assert(nearlyEqual(v[2], 3.5));

	// Test const pointer access
	const Vector<double, 3>& cv = v;
	const double* cptr = cv();
	assert(nearlyEqual(cptr[0], 1.5));
	assert(nearlyEqual(cptr[1], 2.5));
	assert(nearlyEqual(cptr[2], 3.5));
}

void test_dynamic_access() {
	Vector<double> v(5);

	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = static_cast<double>(i * 2);
	}

	for (size_t i = 0; i < v.size(); ++i) {
		assert(nearlyEqual(v[i], static_cast<double>(i * 2)));
	}
}

// ==================== COPY OPERATIONS TESTS ====================

void test_copy_function() {
	Vector<double, 3> original;
	original[0] = 1.0;
	original[1] = 2.0;
	original[2] = 3.0;

	Vector<double, 3> copied = original.copy();

	// Should have same values
	assert(nearlyEqual(copied[0], 1.0));
	assert(nearlyEqual(copied[1], 2.0));
	assert(nearlyEqual(copied[2], 3.0));

	// Should be different memory locations
	assert(copied() != original());

	// Modifying original shouldn't affect copy
	original[0] = 100.0;
	assert(nearlyEqual(copied[0], 1.0));
}

void test_assignment_operators() {
	Vector<double, 3> a, b;

	// Initialize a
	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;

	// Test assignment
	b = a;
	assert(nearlyEqual(b[0], 1.0));
	assert(nearlyEqual(b[1], 2.0));
	assert(nearlyEqual(b[2], 3.0));

	// Test that they're independent
	a[0] = 100.0;
	assert(nearlyEqual(b[0], 1.0));	 // b should be unchanged
}

void test_cross_size_assignment() {
	Vector<double, 3> v3;
	Vector<double, 4> v4;

	v3[0] = 1.0;
	v3[1] = 2.0;
	v3[2] = 3.0;

	// This should work if your implementation supports it
	// Otherwise, it should throw an exception
	bool threw = false;
	try {
		v4 = v3;  // Assigning smaller to larger
	} catch (const std::exception&) {
		threw = true;
	}

	// Either it works or it throws - both are valid behaviors
	if (!threw) {
		assert(nearlyEqual(v4[0], 1.0));
		assert(nearlyEqual(v4[1], 2.0));
		assert(nearlyEqual(v4[2], 3.0));
	}
}

// ==================== EXPRESSION TEMPLATE TESTS ====================

void test_vector_addition_expressions() {
	Vector<double, 3> a, b, result;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 4.0;
	b[1] = 5.0;
	b[2] = 6.0;

	// Test expression template assignment
	result = a + b;
	assert(nearlyEqual(result[0], 5.0));
	assert(nearlyEqual(result[1], 7.0));
	assert(nearlyEqual(result[2], 9.0));

	// Test with dynamic vectors
	Vector<double> da(3), db(3), dresult(3);
	da[0] = 1.0;
	da[1] = 2.0;
	da[2] = 3.0;
	db[0] = 10.0;
	db[1] = 20.0;
	db[2] = 30.0;

	dresult = da + db;
	assert(nearlyEqual(dresult[0], 11.0));
	assert(nearlyEqual(dresult[1], 22.0));
	assert(nearlyEqual(dresult[2], 33.0));
}

void test_vector_subtraction_expressions() {
	Vector<double, 3> a, b, result;

	a[0] = 10.0;
	a[1] = 20.0;
	a[2] = 30.0;
	b[0] = 1.0;
	b[1] = 2.0;
	b[2] = 3.0;

	result = a - b;
	assert(nearlyEqual(result[0], 9.0));
	assert(nearlyEqual(result[1], 18.0));
	assert(nearlyEqual(result[2], 27.0));
}

void test_vector_multiplication_expressions() {
	Vector<double, 3> a, b, result;

	a[0] = 2.0;
	a[1] = 3.0;
	a[2] = 4.0;
	b[0] = 5.0;
	b[1] = 6.0;
	b[2] = 7.0;

	result = a * b;
	assert(nearlyEqual(result[0], 10.0));
	assert(nearlyEqual(result[1], 18.0));
	assert(nearlyEqual(result[2], 28.0));
}

void test_vector_division_expressions() {
	Vector<double, 3> a, b, result;

	a[0] = 12.0;
	a[1] = 15.0;
	a[2] = 18.0;
	b[0] = 3.0;
	b[1] = 5.0;
	b[2] = 6.0;

	result = a / b;
	assert(nearlyEqual(result[0], 4.0));
	assert(nearlyEqual(result[1], 3.0));
	assert(nearlyEqual(result[2], 3.0));
}

void test_scalar_operations_expressions() {
	Vector<double, 3> a, result;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;

	// Test scalar addition
	result = a + 5.0;
	assert(nearlyEqual(result[0], 6.0));
	assert(nearlyEqual(result[1], 7.0));
	assert(nearlyEqual(result[2], 8.0));

	// Test scalar subtraction
	result = a - 1.0;
	assert(nearlyEqual(result[0], 0.0));
	assert(nearlyEqual(result[1], 1.0));
	assert(nearlyEqual(result[2], 2.0));

	// Test scalar multiplication
	result = a * 3.0;
	assert(nearlyEqual(result[0], 3.0));
	assert(nearlyEqual(result[1], 6.0));
	assert(nearlyEqual(result[2], 9.0));

	// Test scalar division
	result = a / 2.0;
	assert(nearlyEqual(result[0], 0.5));
	assert(nearlyEqual(result[1], 1.0));
	assert(nearlyEqual(result[2], 1.5));
}

void test_complex_expressions() {
	Vector<double, 3> a, b, c, result;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 2.0;
	b[1] = 3.0;
	b[2] = 4.0;
	c[0] = 1.0;
	c[1] = 1.0;
	c[2] = 1.0;

	// Test complex expression: (a + b) * c - 2.0
	result = (a + b);
	result *= c;
	result -= 2.0;

	// a + b = [3, 5, 7]
	// (a + b) * c = [3*1, 5*1, 7*1] = [3, 5, 7]
	// (a + b) * c - 2.0 = [1, 3, 5]
	assert(nearlyEqual(result[0], 1.0));
	assert(nearlyEqual(result[1], 3.0));
	assert(nearlyEqual(result[2], 5.0));
}

// ==================== IN-PLACE OPERATIONS TESTS ====================

void test_inplace_addition() {
	Vector<double, 3> a, b;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 4.0;
	b[1] = 5.0;
	b[2] = 6.0;

	a += b;
	assert(nearlyEqual(a[0], 5.0));
	assert(nearlyEqual(a[1], 7.0));
	assert(nearlyEqual(a[2], 9.0));

	// Test scalar addition
	a += 1.0;
	assert(nearlyEqual(a[0], 6.0));
	assert(nearlyEqual(a[1], 8.0));
	assert(nearlyEqual(a[2], 10.0));
}

void test_inplace_subtraction() {
	Vector<double, 3> a, b;

	a[0] = 10.0;
	a[1] = 20.0;
	a[2] = 30.0;
	b[0] = 1.0;
	b[1] = 2.0;
	b[2] = 3.0;

	a -= b;
	assert(nearlyEqual(a[0], 9.0));
	assert(nearlyEqual(a[1], 18.0));
	assert(nearlyEqual(a[2], 27.0));

	// Test scalar subtraction
	a -= 9.0;
	assert(nearlyEqual(a[0], 0.0));
	assert(nearlyEqual(a[1], 9.0));
	assert(nearlyEqual(a[2], 18.0));
}

void test_inplace_multiplication() {
	Vector<double, 3> a, b;

	a[0] = 2.0;
	a[1] = 3.0;
	a[2] = 4.0;
	b[0] = 5.0;
	b[1] = 6.0;
	b[2] = 7.0;

	a *= b;
	assert(nearlyEqual(a[0], 10.0));
	assert(nearlyEqual(a[1], 18.0));
	assert(nearlyEqual(a[2], 28.0));

	// Test scalar multiplication
	a *= 0.1;
	assert(nearlyEqual(a[0], 1.0));
	assert(nearlyEqual(a[1], 1.8));
	assert(nearlyEqual(a[2], 2.8));
}

void test_inplace_division() {
	Vector<double, 3> a, b;

	a[0] = 20.0;
	a[1] = 18.0;
	a[2] = 28.0;
	b[0] = 4.0;
	b[1] = 3.0;
	b[2] = 7.0;

	a /= b;
	assert(nearlyEqual(a[0], 5.0));
	assert(nearlyEqual(a[1], 6.0));
	assert(nearlyEqual(a[2], 4.0));

	// Test scalar division
	a /= 2.0;
	assert(nearlyEqual(a[0], 2.5));
	assert(nearlyEqual(a[1], 3.0));
	assert(nearlyEqual(a[2], 2.0));
}

void test_inplace_with_expressions() {
	Vector<double, 3> a, b, c;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 2.0;
	b[1] = 3.0;
	b[2] = 4.0;
	c[0] = 1.0;
	c[1] = 1.0;
	c[2] = 1.0;

	// Test += with expression
	a += b * c;
	// b * c = [2, 3, 4]
	// a = [1, 2, 3] + [2, 3, 4] = [3, 5, 7]
	assert(nearlyEqual(a[0], 3.0));
	assert(nearlyEqual(a[1], 5.0));
	assert(nearlyEqual(a[2], 7.0));
}

void test_chained_inplace_operations() {
	Vector<double, 3> a;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;

	// Test chaining: ((a += 1) *= 2) -= 3
	a += 1.0;  // [2, 3, 4]
	a *= 2.0;  // [4, 6, 8]
	a -= 3.0;  // [1, 3, 5]

	assert(nearlyEqual(a[0], 1.0));
	assert(nearlyEqual(a[1], 3.0));
	assert(nearlyEqual(a[2], 5.0));
}

// ==================== DOT PRODUCT TESTS ====================

void test_dot_product() {
	Vector<double, 3> a, b;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 4.0;
	b[1] = 5.0;
	b[2] = 6.0;

	double dot = a.dot(b);
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	assert(nearlyEqual(dot, 32.0));

	// Test orthogonal vectors
	Vector<double, 2> x, y;
	x[0] = 1.0;
	x[1] = 0.0;
	y[0] = 0.0;
	y[1] = 1.0;

	assert(nearlyEqual(x.dot(y), 0.0));
}

void test_dot_product_dynamic() {
	Vector<double> a(4), b(4);

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	a[3] = 4.0;
	b[0] = 2.0;
	b[1] = 3.0;
	b[2] = 4.0;
	b[3] = 5.0;

	double dot = a.dot(b);
	// 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
	assert(nearlyEqual(dot, 40.0));
}

void test_dot_product_size_mismatch() {
	Vector<double, 3> a;
	Vector<double, 4> b;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 1.0;
	b[1] = 2.0;
	b[2] = 3.0;
	b[3] = 4.0;

	bool threw = false;
	try {
		double dot = a.dot(b);
		(void)dot;	// Suppress unused variable warning
	} catch (const std::exception&) {
		threw = true;
	}

	assert(threw);	// Should throw for mismatched sizes
}

// ==================== UTILITY OPERATIONS TESTS ====================

void test_setZero() {
	Vector<double, 4> v;

	// Initialize with non-zero values
	v[0] = 1.0;
	v[1] = 2.0;
	v[2] = 3.0;
	v[3] = 4.0;

	v.setZero();

	for (size_t i = 0; i < v.size(); ++i) {
		assert(nearlyEqual(v[i], 0.0));
	}

	// Test with dynamic vector
	Vector<double> dv(5);
	for (size_t i = 0; i < dv.size(); ++i) {
		dv[i] = static_cast<double>(i + 10);
	}

	dv.setZero();
	for (size_t i = 0; i < dv.size(); ++i) {
		assert(nearlyEqual(dv[i], 0.0));
	}
}

void test_setOne() {
	Vector<double, 3> v;

	v.setOne();

	for (size_t i = 0; i < v.size(); ++i) {
		assert(nearlyEqual(v[i], 1.0));
	}

	// Test with different types
	Vector<float, 4> fv;
	fv.setOne();

	for (size_t i = 0; i < fv.size(); ++i) {
		assert(nearlyEqual(fv[i], 1.0f));
	}
}

// ==================== ERROR HANDLING TESTS ====================

void test_size_mismatch_errors() {
	Vector<double, 3> a;
	Vector<double, 4> b;

	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	b[0] = 1.0;
	b[1] = 2.0;
	b[2] = 3.0;
	b[3] = 4.0;

	bool threw = false;

	// Test vector addition size mismatch
	try {
		a += b;
	} catch (const std::exception&) {
		threw = true;
	}
	assert(threw);

	threw = false;
	try {
		Vector<double, 3> result;
		result = a + b;
	} catch (const std::exception&) {
		threw = true;
	}
	assert(threw);
}

void test_unallocated_vector_operations() {
	Vector<double> empty;  // Unallocated dynamic vector
	Vector<double> allocated(3);
	allocated[0] = 1.0;
	allocated[1] = 2.0;
	allocated[2] = 3.0;

	assert(!empty.isAlloc());
	assert(allocated.isAlloc());

	bool threw = false;
	try {
		empty += allocated;
	} catch (const std::exception&) {
		threw = true;
	}
	assert(threw);
}

// ==================== PERFORMANCE TESTS ====================

void test_expression_template_performance() {
	const size_t size = 4096;
	Vector<double> a(size), b(size), c(size), result(size);

	// Initialize vectors
	for (size_t i = 0; i < size; ++i) {
		a[i] = static_cast<double>(i);
		b[i] = static_cast<double>(i * 2);
		c[i] = static_cast<double>(i + 1);
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Complex expression that should be optimized by expression templates
	result = (a + b);
	result *= c;
	result -= 1.0;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Expression template evaluation took: " << duration.count()
			  << " microseconds" << std::endl;

	// Verify results
	for (size_t i = 0; i < 10; ++i) {  // Check first 10 elements
		double expected =
			(static_cast<double>(i) + static_cast<double>(i * 2)) *
				static_cast<double>(i + 1) -
			1.0;
		assert(nearlyEqual(result[i], expected));
	}
}

void test_large_vector_operations() {
	const size_t large_size = 10000;
	Vector<double> a(large_size), b(large_size);

	// Initialize
	for (size_t i = 0; i < large_size; ++i) {
		a[i] = static_cast<double>(i);
		b[i] = static_cast<double>(large_size - i);
	}

	// Test operations
	a += b;
	a *= 0.5;

	// Verify some results
	assert(nearlyEqual(a[0], static_cast<double>(large_size) * 0.5));
	assert(
		nearlyEqual(a[large_size - 1], static_cast<double>(large_size) * 0.5));
}

// ==================== MIXED TYPE TESTS ====================

void test_float_double_operations() {
	FVector3 fv;
	Vector3 dv;

	fv[0] = 1.0f;
	fv[1] = 2.0f;
	fv[2] = 3.0f;
	dv[0] = 1.0;
	dv[1] = 2.0;
	dv[2] = 3.0;

	// Test that operations work with both types
	fv += 1.0f;
	dv += 1.0;

	assert(nearlyEqual(fv[0], 2.0f));
	assert(nearlyEqual(dv[0], 2.0));

	fv.setZero();
	dv.setZero();

	for (size_t i = 0; i < 3; ++i) {
		assert(nearlyEqual(fv[i], 0.0f));
		assert(nearlyEqual(dv[i], 0.0));
	}
}

// ==================== EDGE CASES ====================

void test_zero_operations() {
	Vector<double, 3> a, b;

	a.setZero();
	b[0] = 1.0;
	b[1] = 2.0;
	b[2] = 3.0;

	// Test zero vector operations
	Vector<double, 3> result;
	result = a + b;

	assert(nearlyEqual(result[0], 1.0));
	assert(nearlyEqual(result[1], 2.0));
	assert(nearlyEqual(result[2], 3.0));

	result = a * b;
	assert(nearlyEqual(result[0], 0.0));
	assert(nearlyEqual(result[1], 0.0));
	assert(nearlyEqual(result[2], 0.0));

	assert(nearlyEqual(a.dot(b), 0.0));
}

void test_single_element_vector() {
	Vector<double, 1> v1;
	Vector<double, 1> v2;

	v1[0] = 5.0;
	v2[0] = 3.0;

	Vector<double, 1> result;
	result = v1 + v2;
	assert(nearlyEqual(result[0], 8.0));

	result = v1 * v2;
	assert(nearlyEqual(result[0], 15.0));

	assert(nearlyEqual(v1.dot(v2), 15.0));
}

// ==================== MAIN TEST RUNNER ====================

int main() {
	std::cout << "=== Comprehensive Vector Test Suite ===" << std::endl;
	std::cout << "Testing expression template Vector implementation"
			  << std::endl
			  << std::endl;

	// Construction tests
	std::cout << "--- Construction Tests ---" << std::endl;
	TEST(test_static_construction);
	TEST(test_dynamic_construction);
	TEST(test_mixed_size_construction);
	TEST(test_typedefs);

	// Access tests
	std::cout << "\n--- Access Tests ---" << std::endl;
	TEST(test_element_access);
	TEST(test_raw_pointer_access);
	TEST(test_dynamic_access);

	// Copy operations
	std::cout << "\n--- Copy Operations Tests ---" << std::endl;
	TEST(test_copy_function);
	TEST(test_assignment_operators);
	TEST(test_cross_size_assignment);

	// Expression template tests
	std::cout << "\n--- Expression Template Tests ---" << std::endl;
	TEST(test_vector_addition_expressions);
	TEST(test_vector_subtraction_expressions);
	TEST(test_vector_multiplication_expressions);
	TEST(test_vector_division_expressions);
	TEST(test_scalar_operations_expressions);
	TEST(test_complex_expressions);

	// In-place operations
	std::cout << "\n--- In-Place Operations Tests ---" << std::endl;
	TEST(test_inplace_addition);
	TEST(test_inplace_subtraction);
	TEST(test_inplace_multiplication);
	TEST(test_inplace_division);
	TEST(test_inplace_with_expressions);
	TEST(test_chained_inplace_operations);

	// Mathematical operations
	TEST(test_dot_product);
	TEST(test_dot_product_dynamic);
	TEST(test_dot_product_size_mismatch);

	// Utility operations
	TEST(test_setZero);
	TEST(test_setOne);

	// Error Handling Tests
	TEST(test_size_mismatch_errors);
	TEST(test_unallocated_vector_operations);

	// Performance Tests
	TEST(test_expression_template_performance);
	TEST(test_large_vector_operations);

	// Mixed Type tests
	TEST(test_float_double_operations);

	// Edge cases
	TEST(test_zero_operations);
	TEST(test_single_element_vector);
}
