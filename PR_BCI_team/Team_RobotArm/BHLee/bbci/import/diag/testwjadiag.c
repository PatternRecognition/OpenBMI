/* Programme de test de jadiag */
#define	REAL	double
#include <stdio.h>
#include <math.h>

extern REAL jadiagw(REAL c[], REAL w[], int n, int m,
		    REAL a[], REAL *logdet, REAL *decr);

main() {

  REAL	c[18][6] = {
    { 45, 10,   0,  5,   0,   0  },
    { 10, 45,   5,  0,   0,   0  },
    {  0,  5,  45, 10,   0,   0  },
    {  5,  0,  10, 45,   0,   0  },
    {  0,  0,   0,  0, 16.4, -4.8},
    {  0,  0,   0,  0, -4.8, 13.6},
    { 27.5, -12.5,  -.5,  -4.5, -2.04,  3.72},
    {-12.5,  27.5,  -4.5, -.5,   2.04, -3.72},
    {  -.5,  -4.5,  24.5, -9.5, -3.72, -2.04},
    { -4.5,  -.5,   -9.5, 24.5,  3.72,  2.04},
    {-2.04,  2.04, -3.72, 3.72, 54.76, -4.68},
    { 3.72, -3.72, -2.04, 2.04, -4.68, 51.24},
    { 1, 0, 0, 0, 0, 0},
    { 0, 1, 0, 0, 0, 0},
    { 0, 0, 1, 0, 0, 0},
    { 0, 0, 0, 1, 0, 0},
    { 0, 0, 0, 0, 1, 0},
    { 0, 0, 0, 0, 0, 1}};

  REAL a[6][6], w[3];
  int i, j, m = 3, n = 6;
  char	oui_non[3], line[80];
  REAL tmp, crit, logdet = log(5.184000e+17), decr;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) a[j][i] = 0;
    a[i][i] = 1;
  }

  w[0] = w[1] = 1;
  printf("Poid de la matrice identite ? ");
  fgets(line, sizeof line, stdin);
  sscanf(line, "%lf", &w[2]);
  printf("Poid de la matrice identite = %f\n", w[2]);

  do {
    crit = jadiagw(c, w, n, m, a, &logdet, &decr);

    printf("Matrice A\n");
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
	printf("%10.4f", a[j][i]);
      printf("\n");
    }
    printf("Matrice C1 (initiale en partie triangulaire superieure)\n");
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
	printf("%10.4f", c[j][i]);
      printf("\n");
    }
    printf("Matrice C2 (initiale en partie triangulaire superieure)\n");
    for (i = 0; i < n; i++) {
      for (j = n; j < 2*n; j++)
	printf("%10.4f", c[j][i]);
      printf("\n");
    }
    printf("Matrice C3 (initiale en partie triangulaire superieure)\n");
    for (i = 0; i < n; i++) {
      for (j = 2*n; j < 2*n+n; j++)
	printf("%10.4f", c[j][i]);
      printf("\n");
    }

    printf("logdet= %g, crit= %g, decroissance= %g\n", logdet, crit, decr);
    printf("Continue ? (o/n) ");
    fgets(oui_non, sizeof oui_non, stdin);
  } while (oui_non[0] == 'o');

  for (i = 0; i < n; i++) {
    tmp = sqrt(c[2*n+i][i]);
    for (j = 0; j < n; j++)
      a[j][i] /= tmp; 
    for (j = i; j < n; j++) {
      c[j][i] /= tmp;
      c[n+j][i] /= tmp;
      c[2*n+j][i] /= tmp;
      }
    for (j = 0; j <= i; j++) {
      c[i][j] = (c[j][i] /= tmp);
      c[n+i][j] = (c[n+j][i] /= tmp);
      c[2*n+i][j] = (c[2*n+j][i] /= tmp);
    }
  }

  printf("Matrice A normalisee\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      printf("%10.4f", a[j][i]);
    printf("\n");
  }
  printf("Matrice C1 normalisee\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      printf("%10.4f", c[j][i]);
    printf("\n");
  }
  printf("Matrice C2 normalisee\n");
  for (i = 0; i < n; i++) {
    for (j = n; j < 2*n; j++)
      printf("%10.4f", c[j][i]);
    printf("\n");
  }
  printf("Matrice C3 normalisee\n");
  for (i = 0; i < n; i++) {
    for (j = 2*n; j < 2*n+n; j++)
      printf("%10.4f", c[j][i]);
    printf("\n");
  }
}
