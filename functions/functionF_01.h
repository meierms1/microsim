#ifndef FUNCTION_F_01_H_
#define FUNCTION_F_01_H_

#include "global_vars.h"
double function_F_01_free_energy(double *c, double T, long a) {
  long i,j;
  double sum=0.0;
  for (i=0;i<NUMCOMPONENTS-1;i++) {
    for (j=0;j<NUMCOMPONENTS-1;j++) {
      if (i<=j) {
        sum += A[a][i][j]*c[i]*c[j];
      }
    }
    if (!ISOTHERMAL) {
      B[a][i] = (Beq[a][i] + dBbdT[a][i]*(T-Teq));
    }
    sum += B[a][i]*c[i];
  }
  if (!ISOTHERMAL) {
    C[a] = function_C(T,a);
  }
  sum += C[a];
  return sum;
}
void function_F_01_Mu(double *c, double T, long a, double *Mu) {
  long j,k;
  double sum=0.0;
  for(k=0; k < NUMCOMPONENTS-1; k++) {
    sum  = 2.0*A[a][k][k]*c[k] + (Beq[a][k] + dBbdT[a][k]*(T-Teq));
    for (j=0; j<NUMCOMPONENTS-1;j++) {
      if (k!=j) {
        sum += A[a][k][j]*c[j];
      }
    }
    Mu[k] = sum;
  }
}
void function_F_01_c_mu(double *mu, double *c, double T, long a, double *c_guess) {
  long k,l;
  double sum=0.0;
  for(l=0; l < NUMCOMPONENTS-1; l++) {
    sum = 0.0;
    for (k=0;k < NUMCOMPONENTS-1;k++) {
      sum += cmu[a][l][k]*(mu[k]-(Beq[a][k] + dBbdT[a][k]*(T-Teq)));
    }
    c[l] = sum;
  }
}
void function_F_01_dc_dmu(double *mu, double *phase_comp, double T, long a, double **dcdmu) {
  long i, j;
  for (i=0; i<NUMCOMPONENTS-1; i++) {
    for (j=0; j<NUMCOMPONENTS-1; j++) {
      dcdmu[i][j] = cmu[a][i][j];
    }
  }
}
double function_F_01_dpsi(double *mu, double **phase_comp, double T, double *phi, long a) {
  double psi=0.0;
  double c[NUMCOMPONENTS-1];
  long k=0, b;
  double sum=0.0;
  
  for (b=0; b < NUMPHASES; b++) {
    psi = 0.0;
    for (k=0;k <NUMCOMPONENTS-1; k++) {
      c[k] = phase_comp[b][k];
      psi -= mu[k]*c[k];
    }
    psi += free_energy(c,T,b);
    sum += psi*dhphi(phi, b, a);
  }
  return sum;
}
// void function_F_01_function_A(double T, long a) {
//   long i, j;
//   for (i=0; i<NUMCOMPONENTS-1; i++) {
//     for (j=0; j<NUMCOMPONENTS-1; j++) {
//       A[a][i][j];
//     }
//   }
// }

double function_F_01_function_B(double T, long i, long a) {
  double c_liq[NUMCOMPONENTS-1];
  double c_sol[NUMCOMPONENTS-1];
  
  long k;
  double sum_s=0.0, sum_l=0.0, sum_c=0.0;
  double B_ai=0.0;
  
  if (a != (NUMPHASES-1)) {
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      c_liq[k] = ceq[a][NUMPHASES-1][k] - (DELTA_C[a][k])*(Teq-T)/(DELTA_T[a][NUMPHASES-1]);
      c_sol[k] = ceq[a][a][k]           - (DELTA_C[a][k])*(Teq-T)/(DELTA_T[a][a]);
      if (k!=i) {
        sum_c += A[NUMPHASES-1][k][i]*c_liq[k] - A[a][k][i]*c_sol[k];
      }
    }
    B_ai = (2.0*(A[NUMPHASES-1][i][i]*c_liq[i] - A[a][i][i]*c_sol[i]) + sum_c);
  }
  return B_ai;
}

double function_F_01_function_C(double T, long a) {
  double c_liq[NUMCOMPONENTS-1];
  double c_sol[NUMCOMPONENTS-1];
  
  long i, j, k;
  double sum_s=0.0, sum_l=0.0, sum_c=0.0;
  
  if (a != (NUMPHASES-1)) {
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      c_liq[k] = ceq[a][NUMPHASES-1][k] - (DELTA_C[a][k])*(Teq-T)/(DELTA_T[a][NUMPHASES-1]);
      c_sol[k] = ceq[a][a][k]           - (DELTA_C[a][k])*(Teq-T)/(DELTA_T[a][a]);
    }
    for (i=0; i < NUMCOMPONENTS-1; i++) {
      for (j=0; j < NUMCOMPONENTS-1; j++) {
        if (i <= j) {
          sum_c += (A[a][i][j]*c_sol[i]*c_sol[j] - A[NUMPHASES-1][i][j]*c_liq[i]*c_liq[j]);
        }
      }
    }
  }
  return sum_c;
}


void function_F_01_init_propertymatrices(double T) {
  //Initialize property matrices
  long a, b, i, j, k;
  for (a=0;a<NUMPHASES;a++) {
    for (i=0;i<NUMCOMPONENTS-1;i++) {
      for (j=0;j<NUMCOMPONENTS-1;j++) {
	if (i==j) {
	  muc[a][i][j]=2.0*A[a][i][j];
	} else {
	  muc[a][i][j]=A[a][i][j];
	}
      }
    }
    matinvnew(muc[a], cmu[a], NUMCOMPONENTS-1);
  }

  for (a=0; a < NUMPHASES-1; a++) {
    DELTA_T[a][NUMPHASES-1] = 0.0;
    DELTA_T[a][a]           = 0.0;
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      DELTA_T[a][a]           += slopes[a][a][k]*(ceq[a][NUMPHASES-1][k]              - ceq[a][a][k]);
      DELTA_T[a][NUMPHASES-1] += slopes[a][NUMPHASES-1][k]*(ceq[a][NUMPHASES-1][k]    - ceq[a][a][k]);
      DELTA_C[a][k]            = ceq[a][NUMPHASES-1][k]   - ceq[a][a][k];
    }
    for (k=0; k < NUMCOMPONENTS-1; k++) {
     dcbdT[a][a][k]           = DELTA_C[a][k]/DELTA_T[a][a];
     dcbdT[a][NUMPHASES-1][k] = DELTA_C[a][k]/DELTA_T[a][NUMPHASES-1];
    }
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      dBbdT[a][k]      = 2.0*(A[NUMPHASES-1][k][k]*dcbdT[a][NUMPHASES-1][k] - A[a][k][k]*dcbdT[a][a][k]);
      for (i=0; i < NUMCOMPONENTS-1; i++) {
        if (k!=i) {
          dBbdT[a][k] += (A[NUMPHASES-1][k][i]*dcbdT[a][NUMPHASES-1][i] - A[a][k][i]*dcbdT[a][a][i]);
        }
      }
    }
  }
  for (k=0; k < NUMCOMPONENTS-1; k++) {
    dBbdT[NUMPHASES-1][k] = 0.0;
  }
  for (a=0; a < NUMPHASES; a++) {
    for (i=0; i < NUMCOMPONENTS-1; i++) {
      Beq[a][i] = function_B(Teq, i, a); 
    }
  }
// #endif
  for (a=0; a < NUMPHASES; a++) {
    for (i=0; i < NUMCOMPONENTS-1; i++) {
      B[a][i] = function_B(T, i, a); 
    }
    C[a] = function_C(T,a);
  }
}

#endif