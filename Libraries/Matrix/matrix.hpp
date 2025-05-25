#include "matrixBase.hpp"

#if _USE_LINALG_BLAS
#include "operationsBlas.hpp"
#else
#include "operationsNoBlas.hpp"
#endif
