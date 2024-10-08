#ifndef CALCULATE_DIVERGENCE_CONCENTRATION_H_
#define CALCULATE_DIVERGENCE_CONCENTRATION_H_

void calculate_divergence_concentration_2D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  double sum_dcbdT[NUMCOMPONENTS-1];
  double sum_composition;
  double y[NUMCOMPONENTS];
  double sum_c;
  
  for (gidy=1; gidy <=(workers_mpi.end[Y]+2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      sum[k]        = 0.0;
      sum_dcbdT[k]  = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	      dcdmu[k][l] = 0.0;
      }
    }
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
      }
      if (OBSTACLE) {
        divjat[k]  = (grad->jat[X][k]-grad_back->jat[X][k])*(-0.25*M_PI*epsilon)/deltax;
        divjat[k] += (grad->jat[Y][k]-grad_left->jat[Y][k])*(-0.25*M_PI*epsilon)/deltay;
      }
      if (WELL) {
        divjat[k]  = (grad->jat[X][k]-grad_back->jat[X][k])*(-1.0*epsilon)/deltax;
        divjat[k] += (grad->jat[Y][k]-grad_left->jat[Y][k])*(-1.0*epsilon)/deltay;
      }
    }
    if (grad->interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k])-divjat[k];
      }
      Mu(gridinfo_w[center].composition, T, grad->bulk_phase, mu); 
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltamu[k]                  = mu[k] - gridinfo_w[center].compi[k];
	      gridinfo_w[center].compi[k] = mu[k];
      }
    } else if (grad->interface) {
      for (a=0; a < NUMPHASES; a++) { 
        sum_dhphi = 0.0;
        for (b=0; b < NUMPHASES; b++) {
          sum_dhphi += dhphi(gridinfo_w[center].phia, a, b)*gridinfo_w[center].deltaphi[b];
        }
        for (k=0; k < NUMCOMPONENTS-1; k++) {
          sum[k]       += grad->phase_comp[a][k]*sum_dhphi;
          sum_dcbdT[k] += grad->dcbdT_phase[a][k]*hphi(gridinfo_w[center].phia, a);
        }
      }
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k])-divjat[k];
        gridinfo_w[center].composition[k] += deltac[k];
        if (ISOTHERMAL) {
	        deltac[k] += -sum[k];
        } else {
          deltac[k] += -sum[k] -sum_dcbdT[k];
        }
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
            dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      if (BINARY) {
        inv_dcdmu[0][0]             = 1.0/dcdmu[0][0];
        deltamu[0]                  = deltac[0]*inv_dcdmu[0][0];
        gridinfo_w[center].compi[0] += deltamu[0];
      }
      
      if (TERNARY) {
        
        DET = dcdmu[0][0]*dcdmu[1][1] - dcdmu[0][1]*dcdmu[1][0];

        if (DET == 0){
          printf("\nDET WAS ZERO\n");
          DET = 0.000000001;
        }
        if (isnan(DET)){
          printf("\nDET WAS NAN\n");
          DET = 0.000000001;
          dcdmu[1][1] = 1. ; 
          dcdmu[0][0] = 1. ;
          dcdmu[0][1] = 1. ;
          dcdmu[1][0] = 0. ;
        }

        inv_dcdmu[0][0]  = dcdmu[1][1]/DET;
        inv_dcdmu[1][1]  = dcdmu[0][0]/DET;
        inv_dcdmu[0][1]  = -dcdmu[0][1]/DET;
        inv_dcdmu[1][0]  = -dcdmu[1][0]/DET;
        
        for (k=0; k < NUMCOMPONENTS-1; k++ ) {
          deltamu[k] = 0.0;
          for (l=0; l < NUMCOMPONENTS-1; l++) {
            deltamu[k] += inv_dcdmu[k][l]*deltac[l];
          }
        }
        vectorsum(deltamu, gridinfo_w[center].compi, gridinfo_w[center].compi,NUMCOMPONENTS-1);
      }
      if (DILUTE) {
        for (k=0; k < NUMCOMPONENTS-1; k++) {
          inv_dcdmu[k][k]             =  1.0/dcdmu[k][k];
          deltamu[k]                  =  deltac[k]*inv_dcdmu[k][k];
          gridinfo_w[center].compi[k] += deltamu[k];
        }
      }
      if (!(DILUTE || BINARY || TERNARY)) {
       matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
       multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
       vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
      }
    }
    //Update of phase-field
    for (b=0; b < NUMPHASES; b++) {
       gridinfo_w[center].phia[b] = gridinfo_w[center].phia[b] + gridinfo_w[center].deltaphi[b];
       //FINDING MAXIMUM AND MINIMUM VALUES 
       if ((gidy==1) && (x==1)) {
         workers_max_min.phi_max[b] = gridinfo_w[center].phia[b];
         workers_max_min.phi_min[b] = gridinfo_w[center].phia[b];
       } else if (gridinfo_w[center].phia[b] > workers_max_min.phi_max[b]) {
         workers_max_min.phi_max[b] = gridinfo_w[center].phia[b];
       } else if(gridinfo_w[center].phia[b] < workers_max_min.phi_min[b]) {
         workers_max_min.phi_min[b] = gridinfo_w[center].phia[b];
       }
       workers_max_min.rel_change_phi[b] += gridinfo_w[center].deltaphi[b]*gridinfo_w[center].deltaphi[b];
       //FINISHED
    }
    for (k=0; k<NUMCOMPONENTS-1; k++) {
        //FINDING MAXIMUM AND MINIMUM VALUES 
       if ((gidy == 1) && (x==1)) {
         workers_max_min.mu_max[k] = gridinfo_w[center].compi[k];
         workers_max_min.mu_min[k] = gridinfo_w[center].compi[k];
       } else if (gridinfo_w[center].compi[k] > workers_max_min.mu_max[k]) {
         workers_max_min.mu_max[k] = gridinfo_w[center].compi[k];
       } else if (gridinfo_w[center].compi[k] < workers_max_min.mu_min[k]) {
         workers_max_min.mu_min[k] = gridinfo_w[center].compi[k];
       }
       workers_max_min.rel_change_mu[k] += deltamu[k]*deltamu[k];
       //FINISHED
    }
  }
}
void calculate_divergence_concentration_3D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  double sum_dcbdT[NUMCOMPONENTS-1];
  double sum_composition;
  double y[NUMCOMPONENTS];
  double sum_c;
  
  for (gidy=workers_mpi.rows_y+1; gidy <= (workers_mpi.layer_size - workers_mpi.rows_y-2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;
    top           =  center + workers_mpi.rows_y;
    bottom        =  center - workers_mpi.rows_y;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if (((gidy-workers_mpi.rows_y)/workers_mpi.rows_y) >= 0) {
      grad_bottom  = &gradient[0][gidy-workers_mpi.rows_y];
    }
    
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      sum[k]        = 0.0;
      sum_dcbdT[k]  = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	      dcdmu[k][l] = 0.0;
      }
    }
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
        divflux[k] += (grad->Dmid[Z][k][l]*grad->gradchempot[Z][l] - grad_bottom->Dmid[Z][k][l]*grad_bottom->gradchempot[Z][l])/deltaz;
      }
      if (OBSTACLE) {
        divjat[k]  = (grad->jat[X][k]-grad_back->jat[X][k])*(-0.25*M_PI*epsilon)/deltax;
        divjat[k] += (grad->jat[Y][k]-grad_left->jat[Y][k])*(-0.25*M_PI*epsilon)/deltay;
        divjat[k] += (grad->jat[Z][k]-grad_bottom->jat[Z][k])*(-0.25*M_PI*epsilon)/deltaz;
      }
      if (WELL) {
        divjat[k]  = (grad->jat[X][k]-grad_back->jat[X][k])*(-1.0*epsilon)/deltax;
        divjat[k] += (grad->jat[Y][k]-grad_left->jat[Y][k])*(-1.0*epsilon)/deltay;
        divjat[k] += (grad->jat[Z][k]-grad_bottom->jat[Z][k])*(-1.0*epsilon)/deltaz;
      }
    }
    if (grad->interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k])-divjat[k];
      }
      Mu(gridinfo_w[center].composition, T, grad->bulk_phase, mu); 
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltamu[k]                  = mu[k] - gridinfo_w[center].compi[k];
        gridinfo_w[center].compi[k] = mu[k];
      }
    } else if (grad->interface) {
      for (a=0; a < NUMPHASES; a++) { 
        sum_dhphi = 0.0;
        for (b=0; b < NUMPHASES; b++) {
          sum_dhphi += dhphi(gridinfo_w[center].phia, a, b)*gridinfo_w[center].deltaphi[b];
        }
        for (k=0; k < NUMCOMPONENTS-1; k++) {
          sum[k]       += grad->phase_comp[a][k]*sum_dhphi;
          sum_dcbdT[k] += grad->dcbdT_phase[a][k]*hphi(gridinfo_w[center].phia, a);
        }
      }
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k])-divjat[k];
        gridinfo_w[center].composition[k] += deltac[k];
        if (ISOTHERMAL) {
	        deltac[k] += -sum[k];
        } else {
          deltac[k] += -sum[k] -sum_dcbdT[k];
        }
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
            dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      if (BINARY) {
        inv_dcdmu[0][0]             = 1.0/dcdmu[0][0];
        deltamu[0]                  = deltac[0]*inv_dcdmu[0][0];
        gridinfo_w[center].compi[0] += deltamu[0];
      }
      
      if (TERNARY) {
        DET = dcdmu[0][0]*dcdmu[1][1] - dcdmu[0][1]*dcdmu[1][0];
        inv_dcdmu[0][0]  = dcdmu[1][1]/DET;
        inv_dcdmu[1][1]  = dcdmu[0][0]/DET;
        inv_dcdmu[0][1]  = -dcdmu[0][1]/DET;
        inv_dcdmu[1][0]  = -dcdmu[1][0]/DET;
        
        for (k=0; k < NUMCOMPONENTS-1; k++ ) {
          deltamu[k] = 0.0;
          for (l=0; l < NUMCOMPONENTS-1; l++) {
            deltamu[k] += inv_dcdmu[k][l]*deltac[l];
          }
        }
        vectorsum(deltamu, gridinfo_w[center].compi, gridinfo_w[center].compi,NUMCOMPONENTS-1);
      }
      if (DILUTE) {
        for (k=0; k < NUMCOMPONENTS-1; k++) {
          inv_dcdmu[k][k]             =  1.0/dcdmu[k][k];
          deltamu[k]                  =  deltac[k]*inv_dcdmu[k][k];
          gridinfo_w[center].compi[k] += deltamu[k];
        }
      }
      if (!(DILUTE || BINARY || TERNARY)) {
       matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
       multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
       vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
      }
    }
    //Update of phase-field
    for (b=0; b < NUMPHASES; b++) {
       gridinfo_w[center].phia[b] = gridinfo_w[center].phia[b] + gridinfo_w[center].deltaphi[b];
       //FINDING MAXIMUM AND MINIMUM VALUES 
       if ((gidy==1) && (x==1)) {
         workers_max_min.phi_max[b] = gridinfo_w[center].phia[b];
         workers_max_min.phi_min[b] = gridinfo_w[center].phia[b];
       } else if (gridinfo_w[center].phia[b] > workers_max_min.phi_max[b]) {
         workers_max_min.phi_max[b] = gridinfo_w[center].phia[b];
       } else if(gridinfo_w[center].phia[b] < workers_max_min.phi_min[b]) {
         workers_max_min.phi_min[b] = gridinfo_w[center].phia[b];
       }
       workers_max_min.rel_change_phi[b] += gridinfo_w[center].deltaphi[b]*gridinfo_w[center].deltaphi[b];
       //FINISHED
    }
    for (k=0; k<NUMCOMPONENTS-1; k++) {
        //FINDING MAXIMUM AND MINIMUM VALUES 
       if ((gidy == 1) && (x==1)) {
         workers_max_min.mu_max[k] = gridinfo_w[center].compi[k];
         workers_max_min.mu_min[k] = gridinfo_w[center].compi[k];
       } else if (gridinfo_w[center].compi[k] > workers_max_min.mu_max[k]) {
         workers_max_min.mu_max[k] = gridinfo_w[center].compi[k];
       } else if (gridinfo_w[center].compi[k] < workers_max_min.mu_min[k]) {
         workers_max_min.mu_min[k] = gridinfo_w[center].compi[k];
       }
       workers_max_min.rel_change_mu[k] += deltamu[k]*deltamu[k];
       //FINISHED
    }
  }
}
void calculate_divergence_concentration_smooth_concentration_2D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  for (gidy=1; gidy <=(workers_mpi.end[Y]+2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
//       sum[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	      dcdmu[k][l] = 0.0;
      }
    }
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;

          
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
	
      }
    }
    //Time step iteration, concentration
    //Check whether you are inside the interface
    interface = 1;
    for (a=0; a < NUMPHASES; a++) {
      if (gridinfo_w[center].phia[a] == 1.0) {
        bulk_phase=a;
        interface = 0;
        break;
      }
    }
    if (interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k]);
      }
      Mu(gridinfo_w[center].composition, T, bulk_phase, mu); 
      for (k=0; k < NUMCOMPONENTS-1; k++) {
	      gridinfo_w[center].compi[k] = mu[k];
      }
    } else {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k]);
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
            dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
      multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
      vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
    }

  }
}
void calculate_divergence_concentration_smooth_concentration_3D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  for (gidy=workers_mpi.rows_y+1; gidy <= (workers_mpi.layer_size - workers_mpi.rows_y-2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;
    top           =  center + workers_mpi.rows_y;
    bottom        =  center - workers_mpi.rows_y;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if ((gidy-workers_mpi.rows_y ) >= 0) {
      grad_bottom  = &gradient[0][gidy-workers_mpi.rows_y];
    }
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	        dcdmu[k][l] = 0.0;
      }
    }

    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
        divflux[k] += (grad->Dmid[Z][k][l]*grad->gradchempot[Z][l] - grad_bottom->Dmid[Z][k][l]*grad_bottom->gradchempot[Z][l])/deltaz;
      }
    }
    //Time step iteration, concentration
    //Check whether you are inside the interface
    interface = 1;
    for (a=0; a < NUMPHASES; a++) {
      if (gridinfo_w[center].phia[a] == 1.0) {
        bulk_phase=a;
        interface = 0;
        break;
      }
    }
    if (interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k]);
      }
      Mu(gridinfo_w[center].composition, T, bulk_phase, mu); 
      for (k=0; k < NUMCOMPONENTS-1; k++) {
	       gridinfo_w[center].compi[k] = mu[k];
      }
    } else {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k]);
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
            dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
      multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
      vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
    }
  }
}


void calculate_divergence_concentration_smooth_2D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  for (gidy=1; gidy <=(workers_mpi.end[Y]+2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	      dcdmu[k][l] = 0.0;
      }
    }

    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;

          
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
      }
    }
    //Time step iteration, concentration
    //Check whether you are inside the interface
    interface = 1;
    for (a=0; a < NUMPHASES; a++) {
      if (gridinfo_w[center].phia[a] == 1.0) {
        bulk_phase=a;
        interface = 0;
        break;
      }
    }
    if (interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k]);
      } 
      Mu(gridinfo_w[center].composition, T, bulk_phase, mu);
      for (k=0; k < NUMCOMPONENTS-1; k++) {
	      gridinfo_w[center].compi[k] = mu[k];
      }
    } else {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k]);
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
            dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
      multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
      vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
    }
//     Update of phase-field
    for (b=0; b < NUMPHASES; b++) {
      gridinfo_w[center].phia[b] = gridinfo_w[center].phia[b] + grad->deltaphi[b];
    }
  }
}
void calculate_divergence_concentration_smooth_3D(long x, struct gradlayer **gradient) {
  long k, l, a;
  double mu[NUMCOMPONENTS-1];
  for (gidy=workers_mpi.rows_y+1; gidy <=(workers_mpi.layer_size - workers_mpi.rows_y-2); gidy++) {
    
    center        =  gidy   + (x)*workers_mpi.layer_size;
    front         =  gidy   + (x+1)*workers_mpi.layer_size;
    right         =  center + 1;
    left          =  center - 1;
    top           =  center + workers_mpi.rows_y;
    bottom        =  center - workers_mpi.rows_y;

    grad          =  &gradient[0][gidy];
    grad_left     =  &gradient[0][gidy-1];
    grad_back     =  &gradient[-1][gidy];
    
    if ((gidy-workers_mpi.rows_y) > 0) {
      grad_bottom  = &gradient[0][gidy-workers_mpi.rows_y];
    }
    
    if (!ISOTHERMAL) {
      T = gridinfo_w[center].temperature;
    }
    
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      for (l=0; l < NUMCOMPONENTS-1; l++) {
	      dcdmu[k][l] = 0.0;
      }
    }

    for (k=0; k < NUMCOMPONENTS-1; k++) {
      divflux[k] = 0.0;
      for (l=0; l < NUMCOMPONENTS-1; l++) {
        divflux[k] += (grad->Dmid[X][k][l]*grad->gradchempot[X][l] - grad_back->Dmid[X][k][l]*grad_back->gradchempot[X][l])/deltax;
        divflux[k] += (grad->Dmid[Y][k][l]*grad->gradchempot[Y][l] - grad_left->Dmid[Y][k][l]*grad_left->gradchempot[Y][l])/deltay;
        divflux[k] += (grad->Dmid[Z][k][l]*grad->gradchempot[Z][l] - grad_bottom->Dmid[Z][k][l]*grad_bottom->gradchempot[Z][l])/deltay;
      }
    }
    //Time step iteration, concentration
    //Check whether you are inside the interface
    interface = 1;
    for (a=0; a < NUMPHASES; a++) {
      if (gridinfo_w[center].phia[a] == 1.0) {
        bulk_phase=a;
        interface = 0;
        break;
      }
    }
    if (interface==0) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        gridinfo_w[center].composition[k] += deltat*(divflux[k]);
      }
      Mu(gridinfo_w[center].composition, T, bulk_phase, mu);
      for (k=0; k < NUMCOMPONENTS-1; k++) {
	      gridinfo_w[center].compi[k] = mu[k];
      }
    } else {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        deltac[k] = deltat*(divflux[k]);
        for (l=0; l < NUMCOMPONENTS-1; l++) {
          for (a=0; a < NUMPHASES; a++) {
             dcdmu[k][l] += grad->dcdmu_phase[a][k][l]*hphi(gridinfo_w[center].phia, a);
          }
        }
      }
      matinvnew(dcdmu,inv_dcdmu,NUMCOMPONENTS-1);
      multiply(inv_dcdmu,deltac,deltamu,NUMCOMPONENTS-1);
      vectorsum(deltamu,gridinfo_w[center].compi,gridinfo_w[center].compi,NUMCOMPONENTS-1);
    }
//     Update of phase-field
    for (b=0; b < NUMPHASES; b++) {
      gridinfo_w[center].phia[b] = gridinfo_w[center].phia[b] + grad->deltaphi[b];
    }
  }
}

#endif
