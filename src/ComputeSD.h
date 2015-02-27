#ifndef _SteepDescent_H_
#define __SteepDescent_H_

#include <valarray>
#include <math.h>
#include "omxState.h"
#include "omxFitFunction.h"
#include "omxExportBackendState.h"
#include "Compute.h"
#include "matrix.h"

bool FitCompare(GradientOptimizerContext &, double);

void SD_grad(GradientOptimizerContext &, double);

void steepDES(GradientOptimizerContext &, int);


#endif
