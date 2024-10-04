#ifndef FILL_DOMAIN_H_
#define FILL_DOMAIN_H_

void fill_domain(char *argv[]) {
  FILE *fr;
  int i;
  char tempbuff[1000];

  char tmpstr1[100];
  char tmpstr2[100];
  char **tmp;

  bool decision;

  char *str1, *str2, *token, *subtoken;
  char *saveptr1, *saveptr2;

  long k, j;
  long index;
  long length;
  long phase;

  fr = fopen(argv[2], "rt");

  if(fr == NULL) {
    printf("file %s not found", argv[2]);
  }
  while(fgets(tempbuff,1000,fr)) {
    sscanf(tempbuff, "%100s = %100[^;];", tmpstr1, tmpstr2);
//     printf("%s\n",  tmpstr1);
//     printf("%s\n",  tmpstr2);
    if(tmpstr1[0] != '#') {
      if ((strcmp(tmpstr1, "FILLCUBE") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) >0)) {
        tmp = (char**)malloc(sizeof(char*)*7);
        for (i = 0; i < 7; ++i) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }
        phase = atol(tmp[0]);

        fill_cube_parameters.x_start = atol(tmp[1]) + start[X];
        fill_cube_parameters.x_end   = atol(tmp[4]) + start[X];
        fill_cube_parameters.y_start = atol(tmp[2]) + start[Y];
        fill_cube_parameters.y_end   = atol(tmp[5]) + start[Y];
        fill_cube_parameters.z_start = atol(tmp[3]) + start[Z];
        fill_cube_parameters.z_end   = atol(tmp[6]) + start[Z];

        fill_phase_cube(fill_cube_parameters, gridinfo, phase);
        fill_phase_cube(fill_cube_parameters, gridinfo, NUMPHASES-1);

        for (i = 0; i < 7; ++i) {
          free(tmp[i]);
        }
        free(tmp);
      }
      else if ((strcmp(tmpstr1, "FILLCYLINDER") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) >0)) {
        printf("Filling cylinder\n");
        tmp = (char**)malloc(sizeof(char*)*6);
        for (i = 0; i < 6; ++i) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }
        phase = atol(tmp[0]);

        fill_cylinder_parameters.x_center = atol(tmp[1]) + start[X];
        fill_cylinder_parameters.y_center = atol(tmp[2]) + start[Y];
        fill_cylinder_parameters.z_start  = atol(tmp[3]) + start[Z];
        fill_cylinder_parameters.z_end    = atol(tmp[4]) + start[Z];
        fill_cylinder_parameters.radius   = atof(tmp[5]);

        fill_phase_cylinder(fill_cylinder_parameters, gridinfo, phase);
        fill_phase_cylinder(fill_cylinder_parameters, gridinfo, NUMPHASES-1);

        for (i = 0; i < 6; ++i) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling cylinder\n");
      }
      else if ((strcmp(tmpstr1, "FILLSPHERE") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) >0)) {
        tmp = (char**)malloc(sizeof(char*)*5);
        for (i = 0; i < 5; ++i) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }
        phase = atol(tmp[0]);

        fill_sphere_parameters.x_center = atol(tmp[1]) + start[X];
        fill_sphere_parameters.y_center = atol(tmp[2]) + start[Y];
        fill_sphere_parameters.z_center = atol(tmp[3]) + start[Z];
        fill_sphere_parameters.radius   = atof(tmp[4]);

        fill_phase_sphere(fill_sphere_parameters, gridinfo, phase);
        fill_phase_sphere(fill_sphere_parameters, gridinfo, NUMPHASES-1);

        for (i = 0; i < 5; ++i) {
          free(tmp[i]);
        }
        free(tmp);
      }

      else if ((strcmp(tmpstr1, "FILLELLIPSE") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) >0)) {
        tmp = (char**)malloc(sizeof(char*)*7);
        for (i = 0; i < 7; ++i) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }
        phase = atol(tmp[0]);

        fill_ellipse_parameters.x_center     = atol(tmp[1]) + start[X];
        fill_ellipse_parameters.y_center     = atol(tmp[2]) + start[Y];
        fill_ellipse_parameters.z_center     = atol(tmp[3]) + start[Z];
        fill_ellipse_parameters.major_axis   = atol(tmp[4]);
        fill_ellipse_parameters.eccentricity = atol(tmp[5]);
        fill_ellipse_parameters.rot_angle    = atol(tmp[6]);

        fill_phase_ellipse(fill_ellipse_parameters, gridinfo, phase);
        fill_phase_ellipse(fill_ellipse_parameters, gridinfo, NUMPHASES-1);

        for (i = 0; i < 7; ++i) {
          free(tmp[i]);
        }
        free(tmp);

      }
      else if ((strcmp(tmpstr1, "FILLCYLINDERRANDOM") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) > 0)) {
        printf("Filling cylinders at random\n");
        tmp = (char**)malloc(sizeof(char*)*5);
        for (i = 0; i < 5; i++) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }
        phase                   = atol(tmp[0]);
        long ppt_radius         = atol(tmp[1]);
        double volume_fraction  = atof(tmp[2]);
        long shield_dist        = atol(tmp[3]);
        double spread           = atof(tmp[4]);

        if (shield_dist > 8)
            shield_dist = 8;
        else if (shield_dist == 1)
            shield_dist = 2;

        fill_phase_cylinder_random(phase, ppt_radius, volume_fraction, shield_dist, spread);

        for (i = 0; i < 5; i++) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling cylinders at random\n");
      }
      else if ((strcmp(tmpstr1, "FILLVORONOI2D") == 0) && (NUMPHASES > 0)) {
        printf("Filling Voronoi 2D\n");
        tmp = (char**)malloc(sizeof(char*)*6);
        for (i = 0; i < 6; i++) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }

        long x_start                 = atol(tmp[0]) + start[X];
        long x_end                   = atol(tmp[1]) + start[X];
        long y_start                 = atol(tmp[2]) + start[Y];
        long y_end                   = atol(tmp[3]) + start[Y];
        long NUMPOINTS               = atol(tmp[4]);
        double SIZE                  = atof(tmp[5]);

        fill_cube_parameters.x_start = x_start;
        fill_cube_parameters.x_end   = x_end;
        fill_cube_parameters.y_start = y_start;
        fill_cube_parameters.y_end   = y_end;

        fill_phase_voronoi_2D(fill_cube_parameters, gridinfo, NUMPOINTS, SIZE);


        for (i = 0; i < 6; i++) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling Voronoi 2D");
      }
      else if ( (strcmp(tmpstr1, "FILLCUBEPATTERN")==0) && (NUMPHASES>0) )
      {
        printf("Filling cube pattern.\n");
        tmp = (char**)malloc(sizeof(char*)*8);
        for ( i=0; i<8; i++)
          tmp[i] = (char*)malloc(sizeof(char)*10);
        for ( i=0, str1=tmpstr2; ; i++, str1=NULL )
        {
          token = strtok_r(str1, "{,}", &saveptr1);
          if ( token==NULL )
            break;
          strcpy(tmp[i], token);
        }
        long variants  = atol(tmp[0]);
        long sx        = atol(tmp[1]);
        long sy        = atol(tmp[2]);
        long sz        = atol(tmp[3]);
        double sfrac   = atof(tmp[4]);
        long gap       = atol(tmp[5]);
        double gfrac   = atof(tmp[6]);
        long outof     = atol(tmp[7]);
        fill_cube_pattern(variants, sx, sy, sz, sfrac, gap, gfrac, outof);
        for ( i=0; i<8; i++)
          free(tmp[i]);
        free(tmp);
        printf("End filling cube pattern.\n");
      }
      else if ((strcmp(tmpstr1, "FILLCUBERANDOM") == 0) && (NUMPHASES > 0)) {
        printf("Filling random cubes.\n");
        tmp = (char**)malloc(sizeof(char*)*7);
        for (i = 0; i < 7; i++) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
            break;
          strcpy(tmp[i],token);
        }
        long variants   = atol(tmp[0]);
        long sx         = atol(tmp[1]);
        long sy         = atol(tmp[2]);
        long sz         = atol(tmp[3]);
        double sfrac    = atof(tmp[4]);
        double vol_frac = atof(tmp[5]);
        long shield     = atol(tmp[6]);
        fill_phase_cube_random_variants(variants, sx, sy, sz, sfrac, vol_frac, shield);
        for (i = 0; i < 7; i++) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling random cubes.\n");
      }
      else if ((strcmp(tmpstr1, "FILLVORONOI3D") == 0) && (NUMPHASES > 0)) {
        printf("Filling Voronoi 3D\n");
        tmp = (char**)malloc(sizeof(char*)*8);
        for (i = 0; i < 8; i++) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }

        long x_start                 = atol(tmp[0]) + start[X];
        long x_end                   = atol(tmp[1]) + start[X];
        long y_start                 = atol(tmp[2]) + start[Y];
        long y_end                   = atol(tmp[3]) + start[Y];
        long z_start                 = atol(tmp[4]) + start[Z];
        long z_end                   = atol(tmp[5]) + start[Z];
        long NUMPOINTS               = atol(tmp[6]);
        double SIZE                  = atof(tmp[7]);

        fill_cube_parameters.x_start = x_start;
        fill_cube_parameters.x_end   = x_end;
        fill_cube_parameters.y_start = y_start;
        fill_cube_parameters.y_end   = y_end;
        fill_cube_parameters.z_start = z_start;
        fill_cube_parameters.z_end   = z_end;

        fill_phase_voronoi_3D(fill_cube_parameters, gridinfo, NUMPOINTS, SIZE);

        for (i = 0; i < 8; i++) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling Voronoi 3D");
      }
      else if ((strcmp(tmpstr1, "FILLSPHERERANDOM") == 0) && (NUMPHASES > 0) && ((NUMCOMPONENTS-1) > 0)) {
        printf("Filling spheres at random\n");
        tmp = (char**)malloc(sizeof(char*)*5);
        for (i = 0; i < 5; i++) {
          tmp[i] = (char*)malloc(sizeof(char)*10);
        }
        for (i = 0, str1 = tmpstr2; ; i++, str1 = NULL) {
          token = strtok_r(str1, "{,}", &saveptr1);
          if (token == NULL)
              break;
          strcpy(tmp[i],token);
        }

        phase                   = atol(tmp[0]);
        long ppt_radius         = atol(tmp[1]);
        double volume_fraction  = atof(tmp[2]);
        long shield_dist        = atol(tmp[3]);
        double spread           = atof(tmp[4]);

        if (shield_dist > 8)
            shield_dist = 8;
        else if (shield_dist == 1)
            shield_dist = 2;

        fill_phase_sphere_random(phase, ppt_radius, volume_fraction, shield_dist, spread);

        for (i = 0; i < 5; i++) {
          free(tmp[i]);
        }
        free(tmp);
        printf("End filling spheres at random\n");
      }
    }
  }
  fclose(fr);
  printf("Filling composition\n");
  if ((FUNCTION_F !=5) && (!GRAIN_GROWTH)) {
    fill_composition_cube(gridinfo);
  }
}
#endif
