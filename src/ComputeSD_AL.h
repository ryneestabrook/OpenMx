#ifndef _SteepDescentAL_H_
#define __SteepDescentAL_H_

#include "ComputeSD.h"

void SD_grad(GradientOptimizerContext &, double, double, double *, double *);
bool FitCompare(GradientOptimizerContext &, double, double, double *, double *);
void steepDES(GradientOptimizerContext &rf, int maxIter, double rho, double *lambda, double *mu);
double auglagSD(GradientOptimizerContext &, double, double *, double *);
void auglag_minimize_SD(GradientOptimizerContext &);

#endif
