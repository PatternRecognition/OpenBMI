/* Programme de test de jadiag */
#define	REAL	double
#include <stdio.h>
#include <math.h>

extern REAL jadiag(REAL c[], int n, int m, REAL a[],
		   REAL *logdet, REAL *decr);

main() {

  REAL	c[12][6] = {
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
    { 3.72, -3.72, -2.04, 2.04, -4.68, 51.24}};

  REAL a[6][6];
  int i, j, n = 6;
  char	oui_non[3];
  REAL crit, logdet = log(5.184e17), decr, norm;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) a[j][i] = 0;
    a[i][i] = 1;
  }

  do {
    crit = jadiag(c, n, 2, a, &logdet, &decr);

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
    
    printf("logdet= %g, crit= %g, decroisance= %g\n", logdet, crit, decr);
    printf("Continue ? (o/n) ");
    fgets(oui_non, sizeof oui_non, stdin);
  } while (oui_non[0] == 'o');

  /* normalisation */
  printf("Matrice A normalisee\n");
  for (i = 0; i < n; i++) {
    for (norm = 0, j = 0; j < n; j++)
      norm += a[j][i]*a[j][i];
    norm = sqrt(norm);
    for (j = 0; j < n; j++)
      printf("%10.4f", a[j][i]/norm);
    printf("\n");

    for (j = i; j < n; j++) c[i][j] /= norm;
    for (j = 0; j <= i; j++) c[i][j] = (c[j][i] /= norm);

    for (j = i; j < n; j++) c[n+i][j] /= norm;
    for (j = 0; j <= i; j++) c[n+i][j] = c[n+j][i] /= norm;
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
}
