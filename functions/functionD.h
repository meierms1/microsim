double D(struct fields *gridinfo_w, struct gradlayer *grad, double T, long index, long i, long j) {
  long a, b, k;
  double sum=0.0;
  
  interface = 1;
    
  for (a=0; a < NUMPHASES; a++) {
    if (gridinfo_w[index].phia[a] == 1.0) {
      bulk_phase=a;
      interface = 0;
      break;
    }
  }

  if (interface) {
    for (b=0; b < NUMPHASES; b++) {
      for (k=0; k < NUMCOMPONENTS-1; k++) {
        sum += Diffusivity[b][i][k]*grad->dcdmu_phase[b][k][j]*gridinfo_w[index].phia[b];
      }
    }
  } else {
    for (k=0; k < NUMCOMPONENTS-1; k++) {
      sum += Diffusivity[bulk_phase][i][k]*grad->dcdmu_phase[bulk_phase][k][j];
    }
  }
  return sum;
}
