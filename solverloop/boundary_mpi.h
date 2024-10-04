#ifndef BOUNDARY_MPI_H_
#define BOUNDARY_MPI_H_

void copyXZ(struct bc_scalars *boundary, long x_start, long x_end, struct fields* gridinfo_w, char *field_type);
void copyXY(struct bc_scalars *boundary, long x_start, long x_end, struct fields* gridinfo_w, char *field_type);
void copyYZ(struct bc_scalars *boundary, struct fields* gridinfo_w, char *field_type);

void copyXZ(struct bc_scalars *boundary, long x_start, long x_end, struct fields* gridinfo_w, char *field_type) {
  long gidy_from, gidy_to, y, a, k;
  long copy_from, copy_to;
  long j;
  long x, z;
  int m;
  if (strcmp(field_type, "PHI") == 0) {
   if(((boundary[0].type ==1) && (workers_mpi.firsty || workers_mpi.lasty)) || ((boundary[0].type == 3) && (workers_mpi.firsty && workers_mpi.lasty))) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
        copy_from = boundary[0].proxy[j];
        copy_to   = boundary[0].points[j];
        for (x=x_start; x<=x_end; x++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_from;
            gidy_to            = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_to;
            for (a=0; a < NUMPHASES; a++) {
              gridinfo_w[gidy_to].phia[a]     = gridinfo_w[gidy_from].phia[a];
              gridinfo_w[gidy_to].deltaphi[a] = gridinfo_w[gidy_from].deltaphi[a];
            }
          }
        }
      }
    }
  }
  if (strcmp(field_type, "MU") == 0) {
   if(((boundary[1].type == 1) && (workers_mpi.firsty || workers_mpi.lasty)) || ((boundary[1].type == 3) && (workers_mpi.firsty && workers_mpi.lasty))) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
        copy_from = boundary[1].proxy[j];
        copy_to   = boundary[1].points[j];
        for (x=x_start; x<=x_end; x++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_from;
            gidy_to            = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_to;
            for (k=0; k< NUMCOMPONENTS-1; k++) {
              gridinfo_w[gidy_to].compi[k]       = gridinfo_w[gidy_from].compi[k];
              gridinfo_w[gidy_to].composition[k] = gridinfo_w[gidy_from].composition[k];
            }
          }
        }
      }
    }
  }
  if (strcmp(field_type, "T") == 0) {
   if(((boundary[2].type == 1) && (workers_mpi.firsty || workers_mpi.lasty)) || ((boundary[2].type == 3) && (workers_mpi.firsty && workers_mpi.lasty))) {
     for (j=0; j < 3; j++) { //Loop over three-buffer points
        copy_from = boundary[2].proxy[j];
        copy_to   = boundary[2].points[j];
        for (x=x_start; x<=x_end; x++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from                       = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_from;
            gidy_to                         = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_to;
            gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
          }
        }
      }
    }
  }
  if (strcmp(field_type, "U") == 0) {
   if(((boundary[3].type == 1) && (workers_mpi.firsty || workers_mpi.lasty)) || ((boundary[3].type == 3) && (workers_mpi.firsty && workers_mpi.lasty))) {
     for (j=0; j < 3; j++) { //Loop over three-buffer points
        copy_from = boundary[3].proxy[j];
        copy_to   = boundary[3].points[j];
        for (x=x_start; x<=x_end; x++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from                       = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_from;
            gidy_to                         = x*workers_mpi.layer_size + z*workers_mpi.rows_y + copy_to;
//             gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
            for (m=0; m < 3; m++) {
              for (n=0; n < 3; n++) {
                iter_gridinfo_w[gidy_to].disp[m][n] = iter_gridinfo_w[gidy_from].disp[m][n];
              }
            }
          }
        }
      }
    }
  }
}
void copyYZ(struct bc_scalars *boundary, struct fields* gridinfo_w, char *field_type) {
  long gidy_from, gidy_to, y, a, k;
  long copy_from, copy_to;
  long j;
  long x, z;
  int m;
  if (strcmp(field_type, "PHI") == 0) {
   for (j=0; j < 3; j++) { //Loop over three-buffer points
     if(((boundary[0].type ==1) && (workers_mpi.firstx || workers_mpi.lastx)) || ((boundary[0].type == 3) && (workers_mpi.firstx && workers_mpi.lastx))) {
        copy_from = boundary[0].proxy[j];
        copy_to   = boundary[0].points[j];
        for (y=0; y < workers_mpi.rows_y; y++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = copy_from*workers_mpi.layer_size + z*workers_mpi.rows_y  + y;
            gidy_to            = copy_to*workers_mpi.layer_size   + z*workers_mpi.rows_y  + y;
            for (a=0; a < NUMPHASES; a++) {
              gridinfo_w[gidy_to].phia[a]     = gridinfo_w[gidy_from].phia[a];
              gridinfo_w[gidy_to].deltaphi[a] = gridinfo_w[gidy_from].deltaphi[a];
            }
          }
        }
      }
    }
  }
  if (strcmp(field_type, "MU") == 0) {
   for (j=0; j < 3; j++) { //Loop over three-buffer points
     if(((boundary[1].type ==1) && (workers_mpi.firstx || workers_mpi.lastx)) || ((boundary[1].type == 3) && (workers_mpi.firstx && workers_mpi.lastx))) {
        copy_from = boundary[1].proxy[j];
        copy_to   = boundary[1].points[j];
        for (y=0; y < workers_mpi.rows_y; y++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = copy_from*workers_mpi.layer_size  +  z*workers_mpi.rows_y + y;
            gidy_to            = copy_to*workers_mpi.layer_size    +  z*workers_mpi.rows_y + y;
            for (k=0; k < NUMCOMPONENTS-1; k++) {
              gridinfo_w[gidy_to].compi[k] = gridinfo_w[gidy_from].compi[k];
              gridinfo_w[gidy_to].composition[k] = gridinfo_w[gidy_from].composition[k];
            }
          }
        }
      }
    }
  }
  if (strcmp(field_type, "T") == 0) {
   for (j=0; j < 3; j++) { //Loop over three-buffer points
     if(((boundary[2].type == 1) && (workers_mpi.firstx || workers_mpi.lastx)) || ((boundary[2].type == 3) && (workers_mpi.firstx && workers_mpi.lastx))) {
        copy_from = boundary[2].proxy[j];
        copy_to   = boundary[2].points[j];
        for (y=0; y < workers_mpi.rows_y; y++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = copy_from*workers_mpi.layer_size + z*workers_mpi.rows_y + y;
            gidy_to            = copy_to*workers_mpi.layer_size   + z*workers_mpi.rows_y + y;
            gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
          }
        }
      }
    }
  }
  if (strcmp(field_type, "U") == 0) {
   for (j=0; j < 3; j++) { //Loop over three-buffer points
     if(((boundary[3].type == 1) && (workers_mpi.firstx || workers_mpi.lastx)) || ((boundary[3].type == 3) && (workers_mpi.firstx && workers_mpi.lastx))) {
        copy_from = boundary[3].proxy[j];
        copy_to   = boundary[3].points[j];
        for (y=0; y < workers_mpi.rows_y; y++) {
          for (z=0; z < workers_mpi.rows_z; z++) {
            gidy_from          = copy_from*workers_mpi.layer_size + z*workers_mpi.rows_y + y;
            gidy_to            = copy_to*workers_mpi.layer_size   + z*workers_mpi.rows_y + y;
//             gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
            for (m=0; m < 3; m++) {
              for (n=0; n < 3; n++) {
                iter_gridinfo_w[gidy_to].disp[m][n] = iter_gridinfo_w[gidy_from].disp[m][n];
              }
            }
          }
        }
      }
    }
  }
}
void copyXY(struct bc_scalars *boundary, long x_start, long x_end, struct fields* gridinfo_w, char *field_type) {
  long gidy_from, gidy_to, y, a, k;
  long copy_from, copy_to;
  long j;
  long x, z;
  int m;
  if (strcmp(field_type, "PHI") == 0) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
      if(((boundary[0].type ==1) && (workers_mpi.firstz || workers_mpi.lastz)) || ((boundary[0].type == 3) && (workers_mpi.firstz && workers_mpi.lastz))) {
          copy_from = boundary[0].proxy[j];
          copy_to   = boundary[0].points[j];
          for (x=x_start; x <= x_end; x++) {
            for (y=0; y < workers_mpi.rows_y; y++) {
              gidy_from          = x*workers_mpi.layer_size + copy_from*workers_mpi.rows_y + y;
              gidy_to            = x*workers_mpi.layer_size + copy_to*workers_mpi.rows_y   + y;
              for (a=0; a < NUMPHASES; a++) {
                gridinfo_w[gidy_to].phia[a]     = gridinfo_w[gidy_from].phia[a];
                gridinfo_w[gidy_to].deltaphi[a] = gridinfo_w[gidy_from].deltaphi[a];
              }
            }
          }
        }
      }
   }
   if (strcmp(field_type, "MU") == 0) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
        if(((boundary[1].type ==1) && (workers_mpi.firstz || workers_mpi.lastz)) || ((boundary[1].type == 3) && (workers_mpi.firstz && workers_mpi.lastz))) {
          copy_from = boundary[1].proxy[j];
          copy_to   = boundary[1].points[j];
          for (x=x_start; x <= x_end; x++) {
            for (y=0; y < workers_mpi.rows_y; y++) {
              gidy_from          = x*workers_mpi.layer_size + copy_from*workers_mpi.rows_y + y;
              gidy_to            = x*workers_mpi.layer_size + copy_to*workers_mpi.rows_y   + y;
              for(k=0; k<NUMCOMPONENTS-1; k++) {
                gridinfo_w[gidy_to].compi[k]       = gridinfo_w[gidy_from].compi[k];
                gridinfo_w[gidy_to].composition[k] = gridinfo_w[gidy_from].composition[k];
              }
            }
          }
        }
      }
    }
   if (strcmp(field_type, "T") == 0) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
      if(((boundary[2].type ==1) && (workers_mpi.firstz || workers_mpi.lastz)) || ((boundary[2].type == 3) && (workers_mpi.firstz && workers_mpi.lastz))) {
          copy_from = boundary[2].proxy[j];
          copy_to   = boundary[2].points[j];
          for (x=x_start; x <= x_end; x++) {
            for (y=0; y < workers_mpi.rows_y; y++) {
              gidy_from                      = x*workers_mpi.layer_size + copy_from*workers_mpi.rows_y + y;
              gidy_to                        = x*workers_mpi.layer_size + copy_to*workers_mpi.rows_y   + y;
              gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
            }
          }
        }
      }
   }
   if (strcmp(field_type, "U") == 0) {
    for (j=0; j < 3; j++) { //Loop over three-buffer points
      if(((boundary[3].type ==1) && (workers_mpi.firstz || workers_mpi.lastz)) || ((boundary[3].type == 3) && (workers_mpi.firstz && workers_mpi.lastz))) {
          copy_from = boundary[3].proxy[j];
          copy_to   = boundary[3].points[j];
          for (x=x_start; x <= x_end; x++) {
            for (y=0; y < workers_mpi.rows_y; y++) {
              gidy_from                      = x*workers_mpi.layer_size + copy_from*workers_mpi.rows_y + y;
              gidy_to                        = x*workers_mpi.layer_size + copy_to*workers_mpi.rows_y   + y;
//               gridinfo_w[gidy_to].temperature = gridinfo_w[gidy_from].temperature;
              for (m=0; m < 3; m++) {
                for (n=0; n < 3; n++) {
                  iter_gridinfo_w[gidy_to].disp[m][n] = iter_gridinfo_w[gidy_from].disp[m][n];
                }
              }
            }
          }
        }
      }
   }
}
void apply_boundary_conditions(long taskid) {
  int i, j, field_num;
   if(boundary_worker) {
    //PERIODIC is the default boundary condition  
    if (!((workers_mpi.firstx ==1) && (workers_mpi.lastx ==1))) {
      mpiboundary_left_right(taskid);
    }
    if (!((workers_mpi.firsty ==1) && (workers_mpi.lasty ==1))) {
      mpiboundary_top_bottom(taskid);
    }
    if (DIMENSION == 3) {
//       printf("firstz=%d, lastz=%d, rank_z=%d, front_node=%d, back_node=%d\n", workers_mpi.firstz, workers_mpi.lastz, workers_mpi.rank_z, workers_mpi.front_node, workers_mpi.back_node);
      if (!((workers_mpi.firstz ==1) && (workers_mpi.lastz ==1))) {
        mpiboundary_front_back(taskid);
      }
    }
    //PERIODIC is the default boundary condition  

    for (i=0; i<6; i++) {
      if ((i==0) || (i==1)) {
        if (workers_mpi.firstx) {
          copyYZ(boundary[0], gridinfo_w, "PHI");
          copyYZ(boundary[0], gridinfo_w, "MU");
          if (!ISOTHERMAL) {
            copyYZ(boundary[0], gridinfo_w, "T");
          }
        }
        if (workers_mpi.lastx) {
          copyYZ(boundary[1], gridinfo_w, "PHI");
          copyYZ(boundary[1], gridinfo_w, "MU");
          if (!ISOTHERMAL) {
            copyYZ(boundary[1], gridinfo_w, "T");
          }
        }
      }
      if ((i==2) || (i==3)) {
//         copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "PHI");
//         copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "MU");
//         if (!ISOTHERMAL) {
//           copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "T");
//         }
        if (workers_mpi.lasty) {
          copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
          copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
          if (!ISOTHERMAL) {
            copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
          }
        }
        if (workers_mpi.firsty) {
          copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
          copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
          if (!ISOTHERMAL) {
            copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
          }
        }
      }
      if (DIMENSION == 3) {
//         if ((i==4)||(i==5)) {
//           copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "PHI");
//           copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "T");
//           }
//         }
        if ((i==4)||(i==5)) {
          if (workers_mpi.lastz) {
            copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
            copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
            if (!ISOTHERMAL) {
              copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
            }
          }
          if (workers_mpi.firstz) {
            copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
            copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
            if (!ISOTHERMAL) {
              copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
            }
          }
        }
      }
    }
  }
}
void apply_boundary_conditions_stress(long taskid) {
  int i, j, field_num;
   if(boundary_worker) {
    //PERIODIC is the default boundary condition  
    if (!((workers_mpi.firstx ==1) && (workers_mpi.lastx ==1))) {
      mpiboundary_left_right_stress(taskid);
    }
    if (!((workers_mpi.firsty ==1) && (workers_mpi.lasty ==1))) {
      mpiboundary_top_bottom_stress(taskid);
    }
    if (DIMENSION == 3) {
//       printf("firstz=%d, lastz=%d, rank_z=%d, front_node=%d, back_node=%d\n", workers_mpi.firstz, workers_mpi.lastz, workers_mpi.rank_z, workers_mpi.front_node, workers_mpi.back_node);
      if (!((workers_mpi.firstz ==1) && (workers_mpi.lastz ==1))) {
        mpiboundary_front_back_stress(taskid);
      }
    }
    //PERIODIC is the default boundary condition  

    for (i=0; i<6; i++) {
      if ((i==0) || (i==1)) {
        if (workers_mpi.firstx) {
//           copyYZ(boundary[0], gridinfo_w, "PHI");
//           copyYZ(boundary[0], gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyYZ(boundary[0], gridinfo_w, "T");
//           }
          copyYZ(boundary[0], gridinfo_w, "U");
        }
        if (workers_mpi.lastx) {
//           copyYZ(boundary[1], gridinfo_w, "PHI");
//           copyYZ(boundary[1], gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyYZ(boundary[1], gridinfo_w, "T");
//           }
          copyYZ(boundary[1], gridinfo_w, "U");
        }
      }
      if ((i==2) || (i==3)) {
//         copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "PHI");
//         copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "MU");
//         if (!ISOTHERMAL) {
//           copyXZ(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "T");
//         }
        if (workers_mpi.lasty) {
//           copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
//           copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
//           }
          copyXZ(boundary[2], 0, workers_mpi.rows_x-1, gridinfo_w, "U");
        }
        if (workers_mpi.firsty) {
//           copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
//           copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
//           }
          copyXZ(boundary[3], 0, workers_mpi.rows_x-1, gridinfo_w, "U");
        }
      }
      if (DIMENSION == 3) {
//         if ((i==4)||(i==5)) {
//           copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "PHI");
//           copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "MU");
//           if (!ISOTHERMAL) {
//             copyXY(boundary[i], workers_mpi.start[X], workers_mpi.end[X], gridinfo_w, "T");
//           }
//         }
        if ((i==4)||(i==5)) {
          if (workers_mpi.lastz) {
//             copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
//             copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
//             if (!ISOTHERMAL) {
//               copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
//             }
            copyXY(boundary[4], 0, workers_mpi.rows_x-1, gridinfo_w, "U");
          }
          if (workers_mpi.firstz) {
//             copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "PHI");
//             copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "MU");
//             if (!ISOTHERMAL) {
//               copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "T");
//             }
            copyXY(boundary[5], 0, workers_mpi.rows_x-1, gridinfo_w, "U");
          }
        }
      }
    }
  }
}



// void apply_boundary_conditions(long taskid){
// #ifdef PERIODIC
//   if (!((workers_mpi.firstx ==1) && (workers_mpi.lastx ==1))) {
//     mpiboundary_left_right(taskid);
//   }
// #endif
// #ifndef PERIODIC
//   if (workers_mpi.firstx ==1) {
//     copyYZ(0,5,gridinfo_w);
//     copyYZ(1,4,gridinfo_w);
//     copyYZ(2,3,gridinfo_w);
//   }
//   if (workers_mpi.lastx ==1) {
//     copyYZ(end[X]+1,end[X],gridinfo_w);
//     copyYZ(end[X]+2,end[X]-1,gridinfo_w);
//     copyYZ(end[X]+3,end[X]-2,gridinfo_w);
//   }
// #endif
// #ifdef PERIODIC_Y
//   if (!((workers_mpi.firsty ==1) && (workers_mpi.lasty ==1))) {
//      mpiboundary_top_bottom(taskid);
//   }
// //   copyXZ(2,MESH_Y-4,start,end,gridinfo_w);
// //   copyXZ(1,MESH_Y-5,start,end,gridinfo_w);
// //   copyXZ(0,MESH_Y-6,start,end,gridinfo_w);
// //   
// //   copyXZ(MESH_Y-3,3,start,end,gridinfo_w);
// //   copyXZ(MESH_Y-2,4,start,end,gridinfo_w);
// //   copyXZ(MESH_Y-1,5,start,end,gridinfo_w);
// //   
// #endif
// #ifdef ISOLATE_Y
//  if (workers_mpi.firsty ==1) {
//     copyXZ(2, 3, start[X], end[X], gridinfo_w);
//     copyXZ(1, 4, start[X], end[X], gridinfo_w);
//     copyXZ(0, 5, start[X], end[X], gridinfo_w);
//   }
//   if(workers_mpi.lasty ==1) {
//     copyXZ(rows_y-3, rows_y-4, start[X], end[X], gridinfo_w);
//     copyXZ(rows_y-2, rows_y-5, start[X], end[X], gridinfo_w);
//     copyXZ(rows_y-1, rows_y-6, start[X], end[X], gridinfo_w);
//   }
// #endif
// }
#endif
