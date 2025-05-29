#include "vectorBase.hpp"

#if USE_LINALG_BLAS
#include "operationsBlas.hpp"
#else
#include "operationsNoBlas.hpp"
#endif
