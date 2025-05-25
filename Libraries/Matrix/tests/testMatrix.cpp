#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Libraries/Matrix/matrix.hpp"
#include "vectorBase.hpp"

using namespace LinAlgebra;

// Utility function for floating point comparison
template <typename T>
bool isApproxEqual(T a, T b,
				   T epsilon = std::numeric_limits<T>::epsilon() * 100) {
	return std::abs(a - b) < epsilon;
}

// Helper function to initialize matrix with values
template <typename T, ulong rows, ulong cols>
void initMatrix(Matrix<T, rows, cols>& m, const std::vector<T>& values) {
	assert(values.size() == m.size());
	for (ulong i = 0; i < values.size(); ++i) {
		m[i] = values[i];
	}
}

// Helper function to initialize vector with values
template <typename T, ulong size>
void initVector(Vector<T, size>& v, const std::vector<T>& values) {
	assert(values.size() == v.size());
	for (ulong i = 0; i < values.size(); ++i) {
		v[i] = values[i];
	}
}

// Test result tracking
struct TestResults {
	int passed = 0;
	int failed = 0;

	void pass(const std::string& test_name) {
		passed++;
		std::cout << "✓ " << test_name << " PASSED\n";
	}

	void fail(const std::string& test_name, const std::string& error) {
		failed++;
		std::cout << "✗ " << test_name << " FAILED: " << error << "\n";
	}

	void summary() {
		std::cout << "\n=== TEST SUMMARY ===\n";
		std::cout << "Passed: " << passed << "\n";
		std::cout << "Failed: " << failed << "\n";
		std::cout << "Total:  " << (passed + failed) << "\n";
		if (failed == 0) {
			std::cout << " ALL TESTS PASSED!\n";
		} else {
			std::cout << "XXXX " << failed << " TESTS FAILED\n";
		}
	}
};

TestResults results;

// Macro to run a test and handle exceptions
#define RUN_TEST(test_name, test_code)                    \
	do {                                                  \
		try {                                             \
			test_code;                                    \
			results.pass(test_name);                      \
		} catch (const std::exception& e) {               \
			results.fail(test_name, e.what());            \
		} catch (...) {                                   \
			results.fail(test_name, "Unknown exception"); \
		}                                                 \
	} while (0)

// =============================================================================
// Constructor and Basic Functionality Tests
// =============================================================================

void test_constructors() {
	std::cout << "\n--- Constructor Tests ---\n";

	RUN_TEST("Default Constructor", {
		Matrix22 m;
		assert(m.isAlloc());
		assert(m.rows() == 2);
		assert(m.cols() == 2);
		assert(m.size() == 4);
	});

	RUN_TEST("Dynamic Constructor", {
		Matrix<real> m(3, 4);
		assert(m.isAlloc());
		assert(m.rows() == 3);
		assert(m.cols() == 4);
		assert(m.size() == 12);
	});

	RUN_TEST("Index Access", {
		Matrix22 m;
		m[0] = 1.0;
		m[1] = 2.0;
		m[2] = 3.0;
		m[3] = 4.0;
		assert(m[0] == 1.0);
		assert(m[1] == 2.0);
		assert(m[2] == 3.0);
		assert(m[3] == 4.0);
	});

	RUN_TEST("Pointer Access", {
		Matrix22 m;
		real* ptr = m();
		ptr[0] = 10.0;
		assert(m[0] == 10.0);
	});

	RUN_TEST("Copy Function", {
		Matrix22 original;
		original[0] = 1.0;
		original[1] = 2.0;
		original[2] = 3.0;
		original[3] = 4.0;

		Matrix22 copy = original.copy();

		// Verify values are copied
		for (ulong i = 0; i < 4; ++i) {
			assert(copy[i] == original[i]);
		}

		// Verify independence
		copy[0] = 99.0;
		assert(copy[0] != original[0]);
	});
}

// =============================================================================
// Matrix Addition Tests
// =============================================================================

void test_addition() {
	std::cout << "\n--- Addition Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix Addition", {
		Matrix22 result = m22_a + m22_b;
		assert(result[0] == 6.0);	// 1 + 5
		assert(result[1] == 8.0);	// 2 + 6
		assert(result[2] == 10.0);	// 3 + 7
		assert(result[3] == 12.0);	// 4 + 8
	});

	RUN_TEST("Matrix Scalar Addition", {
		Matrix22 result = m22_a + 10.0;
		assert(result[0] == 11.0);
		assert(result[1] == 12.0);
		assert(result[2] == 13.0);
		assert(result[3] == 14.0);
	});

	RUN_TEST("Scalar Matrix Addition", {
		Matrix22 result = 10.0 + m22_a;
		assert(result[0] == 11.0);
		assert(result[1] == 12.0);
		assert(result[2] == 13.0);
		assert(result[3] == 14.0);
	});

	RUN_TEST("Matrix Addition Size Error", {
		Matrix22 m22;
		Matrix33 m33;
		bool exception_thrown = false;
		try {
			auto result = m22 + m33;
		} catch (const std::runtime_error&) {
			exception_thrown = true;
		}
		assert(exception_thrown);
	});
}

// =============================================================================
// Matrix Subtraction Tests
// =============================================================================

void test_subtraction() {
	std::cout << "\n--- Subtraction Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix Subtraction", {
		Matrix22 result = m22_b - m22_a;
		assert(result[0] == 4.0);  // 5 - 1
		assert(result[1] == 4.0);  // 6 - 2
		assert(result[2] == 4.0);  // 7 - 3
		assert(result[3] == 4.0);  // 8 - 4
	});

	RUN_TEST("Matrix Scalar Subtraction", {
		Matrix22 result = m22_a - 1.0;
		assert(result[0] == 0.0);
		assert(result[1] == 1.0);
		assert(result[2] == 2.0);
		assert(result[3] == 3.0);
	});

	RUN_TEST("Scalar Matrix Subtraction", {
		Matrix22 result = 10.0 - m22_a;
		assert(result[0] == 9.0);  // 10 - 1
		assert(result[1] == 8.0);  // 10 - 2
		assert(result[2] == 7.0);  // 10 - 3
		assert(result[3] == 6.0);  // 10 - 4
	});
}

// =============================================================================
// Matrix Multiplication Tests
// =============================================================================

void test_multiplication() {
	std::cout << "\n--- Multiplication Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix Element-wise Multiplication", {
		Matrix22 result = m22_a * m22_b;
		assert(result[0] == 5.0);	// 1 * 5
		assert(result[1] == 12.0);	// 2 * 6
		assert(result[2] == 21.0);	// 3 * 7
		assert(result[3] == 32.0);	// 4 * 8
	});

	RUN_TEST("Matrix Scalar Multiplication", {
		Matrix22 result = m22_a * 2.0;
		assert(result[0] == 2.0);
		assert(result[1] == 4.0);
		assert(result[2] == 6.0);
		assert(result[3] == 8.0);
	});

	RUN_TEST("Scalar Matrix Multiplication", {
		Matrix22 result = 2.0 * m22_a;
		assert(result[0] == 2.0);
		assert(result[1] == 4.0);
		assert(result[2] == 6.0);
		assert(result[3] == 8.0);
	});
}

// =============================================================================
// Matrix Division Tests
// =============================================================================

void test_division() {
	std::cout << "\n--- Division Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix Element-wise Division", {
		Matrix22 result = m22_b / m22_a;
		assert(result[0] == 5.0);					  // 5 / 1
		assert(result[1] == 3.0);					  // 6 / 2
		assert(isApproxEqual(result[2], 7.0 / 3.0));  // 7 / 3
		assert(result[3] == 2.0);					  // 8 / 4
	});

	RUN_TEST("Matrix Scalar Division", {
		Matrix22 result = m22_a / 2.0;
		assert(result[0] == 0.5);
		assert(result[1] == 1.0);
		assert(result[2] == 1.5);
		assert(result[3] == 2.0);
	});

	RUN_TEST("Scalar Matrix Division", {
		Matrix22 result = 12.0 / m22_a;
		assert(result[0] == 12.0);	// 12 / 1
		assert(result[1] == 6.0);	// 12 / 2
		assert(result[2] == 4.0);	// 12 / 3
		assert(result[3] == 3.0);	// 12 / 4
	});
}

// =============================================================================
// Unary Operator Tests
// =============================================================================

void test_unary_operators() {
	std::cout << "\n--- Unary Operator Tests ---\n";

	Matrix22 m22_a;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});

	RUN_TEST("Unary Plus", {
		Matrix22 result = +m22_a;
		for (ulong i = 0; i < 4; ++i) {
			assert(result[i] == m22_a[i]);
		}
	});

	RUN_TEST("Unary Minus", {
		Matrix22 result = -m22_a;
		assert(result[0] == -1.0);
		assert(result[1] == -2.0);
		assert(result[2] == -3.0);
		assert(result[3] == -4.0);
	});
}

// =============================================================================
// Self-Assignment Operator Tests
// =============================================================================

void test_self_assignment() {
	std::cout << "\n--- Self-Assignment Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Plus Equals", {
		Matrix22 m = m22_a.copy();
		m += m22_b;
		assert(m[0] == 6.0);   // 1 + 5
		assert(m[1] == 8.0);   // 2 + 6
		assert(m[2] == 10.0);  // 3 + 7
		assert(m[3] == 12.0);  // 4 + 8
	});

	RUN_TEST("Minus Equals", {
		Matrix22 m = m22_b.copy();
		m -= m22_a;
		assert(m[0] == 4.0);  // 5 - 1
		assert(m[1] == 4.0);  // 6 - 2
		assert(m[2] == 4.0);  // 7 - 3
		assert(m[3] == 4.0);  // 8 - 4
	});

	RUN_TEST("Times Equals", {
		Matrix22 m = m22_a.copy();
		m *= m22_b;
		assert(m[0] == 5.0);   // 1 * 5
		assert(m[1] == 12.0);  // 2 * 6
		assert(m[2] == 21.0);  // 3 * 7
		assert(m[3] == 32.0);  // 4 * 8
	});

	RUN_TEST("Divide Equals", {
		Matrix22 m = m22_b.copy();
		m /= m22_a;
		assert(m[0] == 5.0);					 // 5 / 1
		assert(m[1] == 3.0);					 // 6 / 2
		assert(isApproxEqual(m[2], 7.0 / 3.0));	 // 7 / 3
		assert(m[3] == 2.0);					 // 8 / 4
	});
}

// =============================================================================
// Matrix-Vector Operations Tests
// =============================================================================

void test_matrix_vector_ops() {
	std::cout << "\n--- Matrix-Vector Operations ---\n";

	RUN_TEST("Matrix Vector Multiplication", {
		Matrix23 m23;
		Vector3 v3;
		initMatrix(m23,
				   {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});	 // [[1, 3, 5], [2, 4, 6]]
		initVector(v3, {1.0, 2.0, 3.0});

		// Expected result: [1*1 + 3*2 + 5*3, 2*1 + 4*2 + 6*3] = [22, 28]
		Vector2 result = m23.matvec(v3);
		assert(result[0] == 22.0);
		assert(result[1] == 28.0);
	});

	RUN_TEST("Matrix Vector No Alloc", {
		Matrix23 m23;
		Vector3 v3;
		Vector2 result;
		initMatrix(m23, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
		initVector(v3, {1.0, 2.0, 3.0});

		m23.matvecNoAlloc(v3, result);
		assert(result[0] == 22.0);
		assert(result[1] == 28.0);
	});
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

void test_edge_cases() {
	std::cout << "\n--- Edge Cases ---\n";

	RUN_TEST("Zero Matrix Operations", {
		Matrix22 zero;
		Matrix22 m22_a;
		initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
		for (ulong i = 0; i < zero.size(); ++i) {
			zero[i] = 0.0;
		}

		Matrix22 result = m22_a + zero;
		for (ulong i = 0; i < 4; ++i) {
			assert(result[i] == m22_a[i]);
		}
	});

	RUN_TEST("Identity-like Operations", {
		Matrix22 ones;
		Matrix22 m22_a;
		initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
		for (ulong i = 0; i < ones.size(); ++i) {
			ones[i] = 1.0;
		}

		Matrix22 result = m22_a * ones;
		for (ulong i = 0; i < 4; ++i) {
			assert(result[i] == m22_a[i]);
		}
	});

	RUN_TEST("Large Values", {
		Matrix22 large;
		const real large_val = 1e10;
		for (ulong i = 0; i < large.size(); ++i) {
			large[i] = large_val;
		}

		Matrix22 result = large / large_val;
		for (ulong i = 0; i < 4; ++i) {
			assert(isApproxEqual(result[i], 1.0, 1e-6));
		}
	});
}

// =============================================================================
// Type-Specific Tests
// =============================================================================

void test_different_types() {
	std::cout << "\n--- Type-Specific Tests ---\n";

	RUN_TEST("Single Precision Matrix", {
		MatrixF22 mf_a;
		MatrixF22 mf_b;
		mf_a[0] = 1.0f;
		mf_a[1] = 2.0f;
		mf_a[2] = 3.0f;
		mf_a[3] = 4.0f;
		mf_b[0] = 5.0f;
		mf_b[1] = 6.0f;
		mf_b[2] = 7.0f;
		mf_b[3] = 8.0f;

		MatrixF22 result = mf_a + mf_b;
		assert(result[0] == 6.0f);
		assert(result[1] == 8.0f);
		assert(result[2] == 10.0f);
		assert(result[3] == 12.0f);
	});

	RUN_TEST("Dynamic Size Operations", {
		Matrix<real> dyn_a(2, 2);
		Matrix<real> dyn_b(2, 2);

		initMatrix(dyn_a, {1.0, 2.0, 3.0, 4.0});
		initMatrix(dyn_b, {5.0, 6.0, 7.0, 8.0});

		Matrix<real> result = dyn_a + dyn_b;
		assert(result[0] == 6.0);
		assert(result[1] == 8.0);
		assert(result[2] == 10.0);
		assert(result[3] == 12.0);
	});
}

// =============================================================================
// Integration Tests
// =============================================================================

void test_integration() {
	std::cout << "\n--- Integration Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Chained Operations", {
		Matrix22 result = (m22_a + m22_b) * 2.0 - m22_a;

		// (m22_a + m22_b) * 2.0 - m22_a = m22_a + 2*m22_b
		assert(result[0] == 11.0);	// 1 + 2*5
		assert(result[1] == 14.0);	// 2 + 2*6
		assert(result[2] == 17.0);	// 3 + 2*7
		assert(result[3] == 20.0);	// 4 + 2*8
	});

	RUN_TEST("Complex Expression", {
		Matrix22 a = m22_a;
		Matrix22 result = a * 2.0 + (-a) / 2.0;

		// a * 2.0 + (-a) / 2.0 = a * (2.0 - 0.5) = a * 1.5
		assert(result[0] == 1.5);
		assert(result[1] == 3.0);
		assert(result[2] == 4.5);
		assert(result[3] == 6.0);
	});
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
	std::cout << "🚀 Starting Linear Algebra Library Tests\n";
	std::cout << "==========================================\n";

	test_constructors();
	test_addition();
	test_subtraction();
	test_multiplication();
	test_division();
	test_unary_operators();
	test_self_assignment();
	test_matrix_vector_ops();
	test_edge_cases();
	test_different_types();
	test_integration();

	results.summary();

	return (results.failed == 0) ? 0 : 1;
}
