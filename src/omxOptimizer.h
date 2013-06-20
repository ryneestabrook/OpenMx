/*
 *  Copyright 2007-2013 The OpenMx Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _OMXOPTIMIZER_H
#define _OMXOPTIMIZER_H

void cacheFreeVarDependencies();
void markFreeVarDependencies(omxState* os, int varNumber);
void handleFreeVarList(omxFitFunction* oo, double* x, int numVars);
void handleFreeVarListHelper(omxState* os, double* x, int numVars); // seems like this should be static TODO

#endif // _OMXOPTIMIZER_H
