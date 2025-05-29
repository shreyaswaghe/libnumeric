#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../Vector/vector.hpp"
#include "../matrix.hpp"

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
			std::cout << "🎉 ALL TESTS PASSED!\n";
		} else {
			std::cout << "❌ " << failed << " TESTS FAILED\n";
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
// Matrix Addition Expression Tests
// =============================================================================

void test_addition_expressions() {
	std::cout << "\n--- Matrix Addition Expression Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix-Matrix Addition Expression", {
		auto expr = m22_a + m22_b;
		static_assert(std::is_same_v<decltype(expr),
									 MMOP<real, 2, 2, 2, 2, OPType::Add>>);
		assert(expr.op == OPType::Add);

		Matrix22 result;
		result = expr;
		assert(result[0] == 6.0);	// 1 + 5
		assert(result[1] == 8.0);	// 2 + 6
		assert(result[2] == 10.0);	// 3 + 7
		assert(result[3] == 12.0);	// 4 + 8
	});

	RUN_TEST("Matrix-Scalar Addition Expression", {
		auto expr = m22_a + 10.0;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Add>>);
		assert(expr.op == OPType::Add);
		assert(expr.scalar == 10.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 11.0);
		assert(result[1] == 12.0);
		assert(result[2] == 13.0);
		assert(result[3] == 14.0);
	});

	RUN_TEST("Scalar-Matrix Addition Expression", {
		auto expr = 10.0 + m22_a;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Add>>);
		assert(expr.op == OPType::Add);
		assert(expr.scalar == 10.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 11.0);
		assert(result[1] == 12.0);
		assert(result[2] == 13.0);
		assert(result[3] == 14.0);
	});

	RUN_TEST("Self-Assignment Addition", {
		Matrix22 result = m22_a.copy();
		result += m22_b;
		assert(result[0] == 6.0);	// 1 + 5
		assert(result[1] == 8.0);	// 2 + 6
		assert(result[2] == 10.0);	// 3 + 7
		assert(result[3] == 12.0);	// 4 + 8
	});

	RUN_TEST("Self-Assignment Scalar Addition", {
		Matrix22 result = m22_a.copy();
		result += 10.0;
		assert(result[0] == 11.0);
		assert(result[1] == 12.0);
		assert(result[2] == 13.0);
		assert(result[3] == 14.0);
	});
}

// =============================================================================
// Matrix Subtraction Expression Tests
// =============================================================================

void test_subtraction_expressions() {
	std::cout << "\n--- Matrix Subtraction Expression Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix-Matrix Subtraction Expression", {
		auto expr = m22_b - m22_a;
		static_assert(std::is_same_v<decltype(expr),
									 MMOP<real, 2, 2, 2, 2, OPType::Sub>>);
		assert(expr.op == OPType::Sub);

		Matrix22 result;
		result = expr;
		assert(result[0] == 4.0);  // 5 - 1
		assert(result[1] == 4.0);  // 6 - 2
		assert(result[2] == 4.0);  // 7 - 3
		assert(result[3] == 4.0);  // 8 - 4
	});

	RUN_TEST("Matrix-Scalar Subtraction Expression", {
		auto expr = m22_a - 1.0;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Sub>>);
		assert(expr.op == OPType::Sub);
		assert(expr.scalar == 1.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 0.0);
		assert(result[1] == 1.0);
		assert(result[2] == 2.0);
		assert(result[3] == 3.0);
	});

	RUN_TEST("Scalar-Matrix Subtraction Expression", {
		auto expr = 10.0 - m22_a;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::SubLeft>>);
		assert(expr.op == OPType::SubLeft);
		assert(expr.scalar == 10.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 9.0);  // 10 - 1
		assert(result[1] == 8.0);  // 10 - 2
		assert(result[2] == 7.0);  // 10 - 3
		assert(result[3] == 6.0);  // 10 - 4
	});

	RUN_TEST("Self-Assignment Subtraction", {
		Matrix22 result = m22_b.copy();
		result -= m22_a;
		assert(result[0] == 4.0);  // 5 - 1
		assert(result[1] == 4.0);  // 6 - 2
		assert(result[2] == 4.0);  // 7 - 3
		assert(result[3] == 4.0);  // 8 - 4
	});
}

// =============================================================================
// Matrix Multiplication Expression Tests
// =============================================================================

void test_multiplication_expressions() {
	std::cout << "\n--- Matrix Multiplication Expression Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix-Matrix Element-wise Multiplication Expression", {
		auto expr = m22_a * m22_b;
		static_assert(std::is_same_v<decltype(expr),
									 MMOP<real, 2, 2, 2, 2, OPType::Mul>>);
		assert(expr.op == OPType::Mul);

		Matrix22 result;
		result = expr;
		assert(result[0] == 5.0);	// 1 * 5
		assert(result[1] == 12.0);	// 2 * 6
		assert(result[2] == 21.0);	// 3 * 7
		assert(result[3] == 32.0);	// 4 * 8
	});

	RUN_TEST("Matrix-Scalar Multiplication Expression", {
		auto expr = m22_a * 2.0;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Mul>>);
		assert(expr.op == OPType::Mul);
		assert(expr.scalar == 2.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 2.0);
		assert(result[1] == 4.0);
		assert(result[2] == 6.0);
		assert(result[3] == 8.0);
	});

	RUN_TEST("Scalar-Matrix Multiplication Expression", {
		auto expr = 2.0 * m22_a;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Mul>>);
		assert(expr.op == OPType::Mul);
		assert(expr.scalar == 2.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 2.0);
		assert(result[1] == 4.0);
		assert(result[2] == 6.0);
		assert(result[3] == 8.0);
	});

	RUN_TEST("Self-Assignment Multiplication", {
		Matrix22 result = m22_a.copy();
		result *= m22_b;
		assert(result[0] == 5.0);	// 1 * 5
		assert(result[1] == 12.0);	// 2 * 6
		assert(result[2] == 21.0);	// 3 * 7
		assert(result[3] == 32.0);	// 4 * 8
	});
}

// =============================================================================
// Matrix Division Expression Tests
// =============================================================================

void test_division_expressions() {
	std::cout << "\n--- Matrix Division Expression Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Matrix-Matrix Element-wise Division Expression", {
		auto expr = m22_b / m22_a;
		static_assert(std::is_same_v<decltype(expr),
									 MMOP<real, 2, 2, 2, 2, OPType::Div>>);
		assert(expr.op == OPType::Div);

		Matrix22 result;
		result = expr;
		assert(result[0] == 5.0);					  // 5 / 1
		assert(result[1] == 3.0);					  // 6 / 2
		assert(isApproxEqual(result[2], 7.0 / 3.0));  // 7 / 3
		assert(result[3] == 2.0);					  // 8 / 4
	});

	RUN_TEST("Matrix-Scalar Division Expression", {
		auto expr = m22_a / 2.0;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::Div>>);
		assert(expr.op == OPType::Div);
		assert(expr.scalar == 2.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 0.5);
		assert(result[1] == 1.0);
		assert(result[2] == 1.5);
		assert(result[3] == 2.0);
	});

	RUN_TEST("Scalar-Matrix Division Expression", {
		auto expr = 12.0 / m22_a;
		static_assert(
			std::is_same_v<decltype(expr), SMOP<real, 2, 2, OPType::DivLeft>>);
		assert(expr.op == OPType::DivLeft);
		assert(expr.scalar == 12.0);

		Matrix22 result;
		result = expr;
		assert(result[0] == 12.0);	// 12 / 1
		assert(result[1] == 6.0);	// 12 / 2
		assert(result[2] == 4.0);	// 12 / 3
		assert(result[3] == 3.0);	// 12 / 4
	});

	RUN_TEST("Self-Assignment Division", {
		Matrix22 result = m22_b.copy();
		result /= m22_a;
		assert(result[0] == 5.0);					  // 5 / 1
		assert(result[1] == 3.0);					  // 6 / 2
		assert(isApproxEqual(result[2], 7.0 / 3.0));  // 7 / 3
		assert(result[3] == 2.0);					  // 8 / 4
	});
}

// =============================================================================
// Expression Composition Tests
// =============================================================================

void test_expression_composition() {
	std::cout << "\n--- Expression Composition Tests ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("Nested Expression Assignment", {
		// Test that expressions can be assigned to SMOP/MMOP assignment
		// operators
		Matrix22 result;
		result = m22_a + m22_b;	 // MMOP assignment
		assert(result[0] == 6.0);
		assert(result[1] == 8.0);
		assert(result[2] == 10.0);
		assert(result[3] == 12.0);

		result = m22_a * 2.0;  // SMOP assignment
		assert(result[0] == 2.0);
		assert(result[1] == 4.0);
		assert(result[2] == 6.0);
		assert(result[3] == 8.0);
	});

	RUN_TEST("Expression Self-Assignment", {
		Matrix22 result = m22_a.copy();
		result += m22_a + m22_b;  // Add MMOP expression
		// result = original + (original + m22_b) = 2*original + m22_b
		assert(result[0] == 7.0);	// 2*1 + 5
		assert(result[1] == 10.0);	// 2*2 + 6
		assert(result[2] == 13.0);	// 2*3 + 7
		assert(result[3] == 16.0);	// 2*4 + 8

		result = m22_a.copy();
		result *= m22_a * 2.0;	// Multiply by SMOP expression
		// result = original * (original * 2) = 2 * original^2
		assert(result[0] == 2.0);	// 2 * 1^2
		assert(result[1] == 8.0);	// 2 * 2^2
		assert(result[2] == 18.0);	// 2 * 3^2
		assert(result[3] == 32.0);	// 2 * 4^2
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
		Vector2 result;
		m23.matvec(v3, result);
		assert(result[0] == 22.0);
		assert(result[1] == 28.0);
	});

	RUN_TEST("Matrix Vector No Alloc", {
		Matrix23 m23;
		Vector3 v3;
		Vector2 result;
		initMatrix(m23, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
		initVector(v3, {1.0, 2.0, 3.0});

		m23.matvec(v3, result);
		assert(result[0] == 22.0);
		assert(result[1] == 28.0);
	});
}

// =============================================================================
// Type-Specific Expression Tests
// =============================================================================

void test_different_type_expressions() {
	std::cout << "\n--- Type-Specific Expression Tests ---\n";

	RUN_TEST("Single Precision Matrix Expressions", {
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

		auto expr = mf_a + mf_b;
		static_assert(std::is_same_v<decltype(expr),
									 MMOP<single, 2, 2, 2, 2, OPType::Add>>);

		MatrixF22 result;
		result = expr;
		assert(result[0] == 6.0f);
		assert(result[1] == 8.0f);
		assert(result[2] == 10.0f);
		assert(result[3] == 12.0f);
	});

	RUN_TEST("Dynamic Size Matrix Expressions", {
		Matrix<real> dyn_a(2, 2);
		Matrix<real> dyn_b(2, 2);

		initMatrix(dyn_a, {1.0, 2.0, 3.0, 4.0});
		initMatrix(dyn_b, {5.0, 6.0, 7.0, 8.0});

		auto expr = dyn_a + dyn_b;
		Matrix<real> result(2, 2);
		result = expr;
		assert(result[0] == 6.0);
		assert(result[1] == 8.0);
		assert(result[2] == 10.0);
		assert(result[3] == 12.0);
	});
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

void test_edge_cases() {
	std::cout << "\n--- Edge Cases ---\n";

	RUN_TEST("Zero Matrix Expression Operations", {
		Matrix22 zero;
		Matrix22 m22_a;
		initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
		for (ulong i = 0; i < zero.size(); ++i) {
			zero[i] = 0.0;
		}

		Matrix22 result;
		result = m22_a + zero;
		for (ulong i = 0; i < 4; ++i) {
			assert(result[i] == m22_a[i]);
		}
	});

	RUN_TEST("Identity-like Expression Operations", {
		Matrix22 ones;
		Matrix22 m22_a;
		initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
		for (ulong i = 0; i < ones.size(); ++i) {
			ones[i] = 1.0;
		}

		Matrix22 result;
		result = m22_a * ones;
		for (ulong i = 0; i < 4; ++i) {
			assert(result[i] == m22_a[i]);
		}
	});

	RUN_TEST("Large Values in Expressions", {
		Matrix22 large;
		const real large_val = 1e10;
		for (ulong i = 0; i < large.size(); ++i) {
			large[i] = large_val;
		}

		Matrix22 result;
		result = large / large_val;
		for (ulong i = 0; i < 4; ++i) {
			assert(isApproxEqual(result[i], 1.0, 1e-6));
		}
	});
}

// =============================================================================
// Expression Template Properties Tests
// =============================================================================

void test_expression_properties() {
	std::cout << "\n--- Expression Template Properties ---\n";

	Matrix22 m22_a, m22_b;
	initMatrix(m22_a, {1.0, 2.0, 3.0, 4.0});
	initMatrix(m22_b, {5.0, 6.0, 7.0, 8.0});

	RUN_TEST("SMOP Properties", {
		auto expr = m22_a + 5.0;
		assert(expr.scalar == 5.0);
		assert(&expr.mat == &m22_a);
		assert(expr.op == OPType::Add);
	});

	RUN_TEST("MMOP Properties", {
		auto expr = m22_a + m22_b;
		assert(&expr.lhs == &m22_a);
		assert(&expr.rhs == &m22_b);
		assert(expr.op == OPType::Add);
	});

	RUN_TEST("Expression Type Verification", {
		// Verify that expressions have the correct types
		auto smop_add = m22_a + 1.0;
		auto smop_sub = m22_a - 1.0;
		auto smop_mul = m22_a * 1.0;
		auto smop_div = m22_a / 1.0;
		auto smop_sub_left = 1.0 - m22_a;
		auto smop_div_left = 1.0 / m22_a;

		static_assert(
			std::is_same_v<decltype(smop_add), SMOP<real, 2, 2, OPType::Add>>);
		static_assert(
			std::is_same_v<decltype(smop_sub), SMOP<real, 2, 2, OPType::Sub>>);
		static_assert(
			std::is_same_v<decltype(smop_mul), SMOP<real, 2, 2, OPType::Mul>>);
		static_assert(
			std::is_same_v<decltype(smop_div), SMOP<real, 2, 2, OPType::Div>>);
		static_assert(std::is_same_v<decltype(smop_sub_left),
									 SMOP<real, 2, 2, OPType::SubLeft>>);
		static_assert(std::is_same_v<decltype(smop_div_left),
									 SMOP<real, 2, 2, OPType::DivLeft>>);

		auto mmop_add = m22_a + m22_b;
		auto mmop_sub = m22_a - m22_b;
		auto mmop_mul = m22_a * m22_b;
		auto mmop_div = m22_a / m22_b;

		static_assert(std::is_same_v<decltype(mmop_add),
									 MMOP<real, 2, 2, 2, 2, OPType::Add>>);
		static_assert(std::is_same_v<decltype(mmop_sub),
									 MMOP<real, 2, 2, 2, 2, OPType::Sub>>);
		static_assert(std::is_same_v<decltype(mmop_mul),
									 MMOP<real, 2, 2, 2, 2, OPType::Mul>>);
		static_assert(std::is_same_v<decltype(mmop_div),
									 MMOP<real, 2, 2, 2, 2, OPType::Div>>);
	});
}

// ==================== MATRIX PERFORMANCE TESTS ====================

void test_matrix_expression_template_performance() {
	const size_t rows = 64, cols = 64;
	Matrix<double, rows, cols> a, b, c, result;

	// Initialize matrices
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			a(i, j) = static_cast<double>(i * cols + j);
			b(i, j) = static_cast<double>((i * cols + j) * 2);
			c(i, j) = static_cast<double>(i * cols + j + 1);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Complex expression that should be optimized by expression templates
	result = (a + b);
	result *= c;
	result -= 1.0;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Matrix expression template evaluation took: "
			  << duration.count() << " microseconds" << std::endl;

	// Verify results (check first few elements)
	for (size_t i = 0; i < 3; ++i) {
		for (size_t j = 0; j < 3; ++j) {
			double idx = static_cast<double>(i * cols + j);
			double expected = (idx + idx * 2) * (idx + 1) - 1.0;
			assert(isApproxEqual(result(i, j), expected));
		}
	}
}

void test_large_matrix_operations() {
	const size_t large_rows = 100, large_cols = 100;
	Matrix<double, large_rows, large_cols> a, b;

	// Initialize matrices
	for (size_t i = 0; i < large_rows; ++i) {
		for (size_t j = 0; j < large_cols; ++j) {
			a(i, j) = static_cast<double>(i * large_cols + j);
			b(i, j) = static_cast<double>((large_rows * large_cols) -
										  (i * large_cols + j));
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Test operations
	a += b;
	a *= 0.5;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Large matrix operations took: " << duration.count()
			  << " microseconds" << std::endl;

	// Verify some results
	double expected_val = static_cast<double>(large_rows * large_cols) * 0.5;
	assert(isApproxEqual(a(0, 0), expected_val));
	assert(isApproxEqual(a(large_rows - 1, large_cols - 1), expected_val));

	// Check a few more elements
	assert(isApproxEqual(a(large_rows / 2, large_cols / 2), expected_val));
}

void test_matrix_multiplication_performance() {
	const size_t dim = 32;	// 32x32 matrices
	Matrix<double, dim, dim> a, b, result;

	// Initialize matrices
	for (size_t i = 0; i < dim; ++i) {
		for (size_t j = 0; j < dim; ++j) {
			a(i, j) = static_cast<double>(i + j + 1);
			b(i, j) = static_cast<double>(i * j + 1);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Matrix multiplication
	result = a * b;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Matrix multiplication (" << dim << "x" << dim
			  << ") took: " << duration.count() << " microseconds" << std::endl;

	// Verify a few elements by manual calculation
	// result(0,0) should be sum of a(0,k) * b(k,0) for k=0 to dim-1
	double expected_00 = a(0, 0) * b(0, 0);
	assert(isApproxEqual(result(0, 0), expected_00));

	// Check another element
	double expected_11 = a(1, 1) * b(1, 1);
	assert(isApproxEqual(result(1, 1), expected_11));
}

/**
void test_matrix_transpose_performance() {
	const size_t rows = 128, cols = 64;
	Matrix<double, rows, cols> a;

	// Initialize matrix
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			a(i, j) = static_cast<double>(i * cols + j + 1);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Transpose operation
	auto transposed = a.transpose();  // or however your transpose is
implemented

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -
start);

	std::cout << "  Matrix transpose (" << rows << "x" << cols
			  << ") took: " << duration.count() << " microseconds" << std::endl;

	// Verify transpose correctness
	for (size_t i = 0; i < std::min(rows, static_cast<size_t>(5)); ++i) {
		for (size_t j = 0; j < std::min(cols, static_cast<size_t>(5)); ++j) {
			assert(isApproxEqual(transposed(j, i), a(i, j)));
		}
	}
}
**/

void test_matrix_scalar_operations_performance() {
	const size_t size = 200;
	Matrix<double, size, size> a;

	// Initialize matrix
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			a(i, j) = static_cast<double>(i + j + 1);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Chain of scalar operations
	a += 5.0;
	a *= 2.0;
	a -= 3.0;
	a /= 1.5;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Matrix scalar operations chain (" << size << "x" << size
			  << ") took: " << duration.count() << " microseconds" << std::endl;

	// Verify a few elements
	// Original: (i + j + 1), after ops: ((i + j + 1 + 5) * 2 - 3) / 1.5
	double expected_00 = ((1.0 + 5.0) * 2.0 - 3.0) / 1.5;
	assert(isApproxEqual(a(0, 0), expected_00));

	double expected_11 = ((3.0 + 5.0) * 2.0 - 3.0) / 1.5;
	assert(isApproxEqual(a(1, 1), expected_11));
}

void test_matrix_copy_performance() {
	const size_t size = 150;
	Matrix<double, size, size> a, b;

	// Initialize matrix a
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			a(i, j) = static_cast<double>(i * size + j);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Copy operation
	b = a;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Matrix copy (" << size << "x" << size
			  << ") took: " << duration.count() << " microseconds" << std::endl;

	// Verify copy correctness
	for (size_t i = 0; i < 5; ++i) {
		for (size_t j = 0; j < 5; ++j) {
			assert(isApproxEqual(b(i, j), a(i, j)));
		}
	}
}

void test_matrix_vector_multiplication_performance() {
	const size_t dim = 128;	 // 128x128 matrix
	Matrix<double, dim, dim> mat;
	Vector<double, dim> vec, result;

	// Initialize matrix
	for (size_t i = 0; i < dim; ++i) {
		for (size_t j = 0; j < dim; ++j) {
			mat(i, j) = static_cast<double>(i + j + 1);
		}
	}

	// Initialize vector
	for (size_t i = 0; i < dim; ++i) {
		vec[i] = static_cast<double>(i + 1);
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Matrix-vector multiplication
	mat.matvec(vec, result);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Matrix-vector multiplication (" << dim << "x" << dim
			  << " * " << dim << ") took: " << duration.count()
			  << " microseconds" << std::endl;

	// Verify a few elements by manual calculation
	// result[i] should be sum of mat(i,j) * vec[j] for j=0 to dim-1
	double expected_0 = 0.0;
	for (size_t j = 0; j < dim; ++j) {
		expected_0 += mat(0, j) * vec[j];
	}
	assert(isApproxEqual(result[0], expected_0));

	// Check another element
	double expected_1 = 0.0;
	for (size_t j = 0; j < dim; ++j) {
		expected_1 += mat(1, j) * vec[j];
	}
	assert(isApproxEqual(result[1], expected_1));

	// Check last element
	double expected_last = 0.0;
	for (size_t j = 0; j < dim; ++j) {
		expected_last += mat(dim - 1, j) * vec[j];
	}
	assert(isApproxEqual(result[dim - 1], expected_last));
}

void test_large_matrix_vector_multiplication() {
	const size_t large_dim = 512;  // 512x512 matrix
	Matrix<double, large_dim, large_dim> mat;
	Vector<double, large_dim> vec, result;

	// Initialize matrix with a pattern
	for (size_t i = 0; i < large_dim; ++i) {
		for (size_t j = 0; j < large_dim; ++j) {
			mat(i, j) = static_cast<double>((i + 1) * (j + 1)) /
						static_cast<double>(large_dim);
		}
	}

	// Initialize vector
	for (size_t i = 0; i < large_dim; ++i) {
		vec[i] = static_cast<double>(i % 10 + 1);  // Repeating pattern 1-10
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Matrix-vector multiplication
	mat.matvec(vec, result);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Large matrix-vector multiplication (" << large_dim << "x"
			  << large_dim << ") took: " << duration.count() << " microseconds"
			  << std::endl;

	// Verify first and middle elements
	double expected_0 = 0.0;
	for (size_t j = 0; j < large_dim; ++j) {
		expected_0 += mat(0, j) * vec[j];
	}
	assert(isApproxEqual(result[0], expected_0));

	double expected_mid = 0.0;
	size_t mid = large_dim / 2;
	for (size_t j = 0; j < large_dim; ++j) {
		expected_mid += mat(mid, j) * vec[j];
	}
	assert(isApproxEqual(result[mid], expected_mid));
}

void test_rectangular_matrix_vector_multiplication() {
	const size_t rows = 64, cols = 96;	// Non-square matrix
	Matrix<double, rows, cols> mat;
	Vector<double, cols> vec;
	Vector<double, rows> result;

	// Initialize matrix
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			mat(i, j) = static_cast<double>(i * cols + j + 1) / 100.0;
		}
	}

	// Initialize vector
	for (size_t i = 0; i < cols; ++i) {
		vec[i] = static_cast<double>(i + 1);
	}

	auto start = std::chrono::high_resolution_clock::now();

	// Matrix-vector multiplication
	mat.matvec(vec, result);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration =
		std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "  Rectangular matrix-vector multiplication (" << rows << "x"
			  << cols << ") took: " << duration.count() << " microseconds"
			  << std::endl;

	// Verify first and last elements
	double expected_0 = 0.0;
	for (size_t j = 0; j < cols; ++j) {
		expected_0 += mat(0, j) * vec[j];
	}
	assert(isApproxEqual(result[0], expected_0));

	double expected_last = 0.0;
	for (size_t j = 0; j < cols; ++j) {
		expected_last += mat(rows - 1, j) * vec[j];
	}
	assert(isApproxEqual(result[rows - 1], expected_last));
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
	std::cout << "🚀 Starting Linear Algebra Expression Template Tests\n";
	std::cout << "===================================================\n";

	test_constructors();
	test_addition_expressions();
	test_subtraction_expressions();
	test_multiplication_expressions();
	test_division_expressions();
	test_expression_composition();
	test_matrix_vector_ops();
	test_different_type_expressions();
	test_edge_cases();
	test_expression_properties();

	test_matrix_expression_template_performance();
	test_large_matrix_operations();
	test_matrix_multiplication_performance();
	test_matrix_scalar_operations_performance();
	test_matrix_copy_performance();

	test_matrix_vector_multiplication_performance();
	test_large_matrix_vector_multiplication();
	test_rectangular_matrix_vector_multiplication();

	results.summary();

	return (results.failed == 0) ? 0 : 1;
}
