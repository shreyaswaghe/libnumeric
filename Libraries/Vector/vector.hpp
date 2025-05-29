#include "vectorBase.hpp"

#define _USE_LINALG_BLAS 0

#if _USE_LINALG_BLAS
#include "vectorOperations.hpp"
#else
#include "operationsNoBlas.hpp"
#endif
