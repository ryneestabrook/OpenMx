/* Steepest descent optimizer test */

#include "computeSD.h"

bool FitCompare(GradientOptimizerContext &rf, double speed)
{
    Eigen::VectorXd prevEst(rf.fc->numParam), currEst(rf.fc->numParam);
    double refFit, newFit;

    memcpy(prevEst.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    refFit = rf.fc->fit;

    Eigen::VectorXd searchDir(rf.fc->grad);
    currEst = prevEst - speed / searchDir.norm() * searchDir;
    memcpy(rf.fc->est, currEst.data(), (rf.fc->numParam) * sizeof(double));
    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    newFit = rf.fc->fit;

    if(newFit < refFit) return newFit < refFit;
    memcpy(rf.fc->est, prevEst.data(), (rf.fc->numParam) * sizeof(double));
    return newFit < refFit;
}

void SD_grad(GradientOptimizerContext &rf)
{
    ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);

    const double refFit = rf.fc->fit;
    const double eps = 1e-5;
    Eigen::VectorXd p1(rf.fc->numParam), p2(rf.fc->numParam);

    memcpy(p1.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));

    for (int px = 0; px < rf.fc->numParam; px++) {
        memcpy(p2.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
        p2[px] += eps;
        memcpy(rf.fc->est, p2.data(), (rf.fc->numParam) * sizeof(double));
        ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
        rf.fc->grad[px] = (rf.fc->fit - refFit) / eps;
        memcpy(rf.fc->est, p1.data(), (rf.fc->numParam) * sizeof(double));
    }
}

void steepDES(GradientOptimizerContext &rf, int maxIter)
{
	int iter = 0;
	double priorSpeed = 1.0;

	while(iter < maxIter)
	{
        if(FitCompare(rf, priorSpeed)){
            iter++;
            SD_grad(rf);
        }
        else
        {
            int retries = 8;
            double speed = priorSpeed;
            bool findit = FALSE;
            while (--retries > 0){
                speed *= 0.5;
                if(FitCompare(rf, speed)){
                    iter++;
                    SD_grad(rf);
                    findit = TRUE;
                    break;
                }
            }
            if(!findit){
                mxLog("cannot find better estimation along the gradient direction");
                return;
            }
        }
    }
    return;
}

