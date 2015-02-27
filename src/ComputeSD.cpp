/* Steepest descent optimizer test */

#include "ComputeSD.h"

void SD_grad(GradientOptimizerContext &rf, double eps)
{
    rf.fc->copyParamToModel();
    ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);

    const double refFit = rf.fc->fit;
    //const double eps = 1e-9;
    Eigen::VectorXd p1(rf.fc->numParam), p2(rf.fc->numParam), grad(rf.fc->numParam);

    memcpy(p1.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));

    for (int px = 0; px < int(rf.fc->numParam); px++) {
        memcpy(p2.data(), rf.fc->est, (rf.fc->numParam) * sizeof(double));
        p2[px] += eps;
        memcpy(rf.fc->est, p2.data(), (rf.fc->numParam) * sizeof(double));
        rf.fc->copyParamToModel();
        ComputeFit("steep_fd", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
        grad[px] = (rf.fc->fit - refFit) / eps;
        memcpy(rf.fc->est, p1.data(), (rf.fc->numParam) * sizeof(double));
        rf.fc->fit = refFit;
        rf.fc->copyParamToModel();
    }
    rf.fc->grad = grad;
}

bool FitCompare(GradientOptimizerContext &rf, double speed)
{
    Eigen::Map< Eigen::VectorXd > currEst(rf.fc->est, rf.fc->numParam);
    Eigen::VectorXd prevEst = currEst;

    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    if (isnan(rf.fc->fit))
    {
        rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
        return FALSE;
    }
    double refFit = rf.fc->fit;

    Eigen::VectorXd searchDir = rf.fc->grad;

    currEst = prevEst - speed * searchDir / searchDir.norm();
    currEst = currEst.cwiseMax(rf.solLB).cwiseMin(rf.solUB);
    if(rf.verbose >= 2){
        for(int index = 0; index < int(rf.fc->numParam); index++)
        {
            if(currEst[index] == rf.solLB[index])
                mxLog("paramter %i hit lower bound %f", index, rf.solLB[index]);
            if(currEst[index] == rf.solUB[index])
                mxLog("paramter %i hit upper bound %f", index, rf.solUB[index]);
        }
    }

    rf.fc->copyParamToModel();
    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);
    double newFit = rf.fc->fit;

    if(newFit < refFit) return newFit < refFit;
    currEst = prevEst;
    rf.fc->copyParamToModel();
    return newFit < refFit;
}



void steepDES(GradientOptimizerContext &rf, int maxIter)
{
	int iter = 0;
	double priorSpeed = 1.0;//, grad_tol = 1e-30;
    rf.setupSimpleBounds();
    rf.informOut = INFORM_UNINITIALIZED;
    rf.fc->copyParamToModel();

    ComputeFit("steep", rf.fitMatrix, FF_COMPUTE_FIT, rf.fc);  // for fitmultigroup.R to check isErrorRaised()

    SD_grad(rf, 1e-9);
 //   Eigen::VectorXd oldgrad;// = rf.fc->grad / rf.fc->grad.norm();

	while(iter < maxIter && !isErrorRaised())
	{

        //
        //SD_grad(rf);



        if(rf.fc->grad.norm() == 0)
        {
            rf.informOut = INFORM_CONVERGED_OPTIMUM;
            mxLog("after %i iterations, gradient achieves zero!", iter);
            break;
        }
        bool findit = FitCompare(rf, priorSpeed);

//        if (!isnan(rf.fc->fit) && rf.fc->grad.norm() / fabs(rf.fc->fit) < grad_tol)
//        {
//            rf.informOut = INFORM_CONVERGED_OPTIMUM;
//            mxLog("after %i iterations, gradient tolerance achieved!", iter);
//            break;
//        }

//        if (findit)
//        {
//            priorSpeed *=1.1;
//            SD_grad(rf);
//            findit = FitCompare(rf, priorSpeed);
//        }

        int retries = 200;
        double speed = priorSpeed;
        while (--retries > 0 && !findit && !isErrorRaised()){
            speed *= 0.5;
            findit = FitCompare(rf, speed);
        }
        if(findit){
            priorSpeed = speed * 1.1;
            iter++;

          //  oldgrad = rf.fc->grad / rf.fc->grad.norm();
            if (speed < 1e-9)
            {
                SD_grad(rf, speed * 1e-2);
            //    mxLog("aha!");
            }
            else
            {
                SD_grad(rf, 1e-9);
            }

            //double graddot = ;
        //    mxLog("At %i iterations, the dot product of gradient is %f", iter, oldgrad.dot(rf.fc->grad / rf.fc->grad.norm()));

            if(iter == maxIter){
                rf.informOut = INFORM_ITERATION_LIMIT;
                mxLog("Maximum iteration achieved!");
                break;
            }
        }
        else{
            switch (iter)
            {
                case 0:
                    rf.informOut = INFORM_STARTING_VALUES_INFEASIBLE;
                    mxLog("Infeasbile starting values!");
                    break;
                case 99999:
                    rf.informOut = INFORM_ITERATION_LIMIT;
                    mxLog("Maximum iteration achieved!");
                    break;
                default:
                    rf.informOut = INFORM_CONVERGED_OPTIMUM;
                    mxLog("after %i iterations, cannot find better estimation along the gradient direction", iter);
            }
            break;
        }
    }
    mxLog("status code : %i", rf.informOut);
    return;
}
