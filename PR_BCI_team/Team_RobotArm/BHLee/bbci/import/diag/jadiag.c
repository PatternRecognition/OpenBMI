#define	REAL	double

#include <math.h>
#define	TINY	10e-20
REAL jadiag(REAL c[], int n, int m, REAL a[], REAL *logdet, REAL *decr)
/*
  Performs joint approximate diagonalization of m matrices.
  These matrices are concatenated into a n by (n*m) matrix and are
  stored column wise as a vector c[n*n*m]. They are tranformed to
  more diagonal in one iteration sweep. Only the lower triangular 
  elements of these matrices are used and changed. The transformation
  is also appliied (through premultipliation only) to the matrix a
  (which is also stored column wise as a vector) and the variable
  logdet is added by 2m times the logarithm of the determinant of the
  transformation. The function returns the criterion, which is the
  logarithm of the product of the diagonal elements of all the
  transformed matrices minus the new value of logdet. Finally, decr
  yields an estimate of the decrease of the criterion near the solution
  (it is more accurate for small decrease since taking the difference of
  the criterion can result in large rounding error)
*/
{
  int	i1,j1;
  int	n2 = n*n, mn2 = m*n2,
	i, ic, ii, ij, j, jc, jj, k;
  REAL p2, q1, p, q,
       alpha, beta, gamma, a12, a21, det;
  register REAL tmp1, tmp2, tmp;

  det = 1;
  *decr = 0;
  for (i = 1, ic = n; i < n ; i++, ic += n)
    for (j = jc = 0; j < i; j++, jc += n) {
      ii = i + ic;
      jj = j + jc;
      ij = i + jc;
      for (q1 = p2 = p = q = 0, k = 0; k < mn2; k += n2) {
	tmp1 = c[ii+k];
	tmp2 = c[jj+k];
	tmp = c[ij+k];
	p += tmp/tmp1;
	q += tmp/tmp2;
	q1 += tmp1/tmp2;
	p2 += tmp2/tmp1;
      }
      q1 /= m;
      p2 /= m;
      p /= m;
      q /= m;
      beta = 1 - p2*q1;			/* p1 = q2 = 1 */
      if (q1 <= p2) {			/* the same as q1*q2 <= p1*p2 */
	alpha = p2*q - p;		/* q2 = 1 */
	if (fabs(alpha) - beta < TINY) {	/* beta <= 0 always */
	  beta = -1;
	  gamma = p/p2;
	} else gamma = - (p*beta + alpha)/p2;
	*decr += m*(p*p - alpha*alpha/beta)/p2;
      } else {
	gamma = p*q1 - q;		/* p1 = 1 */
	if (fabs(gamma) - beta < tiny) {	/* beta <= 0 always */
	  beta = -1;
	  alpha = q/q1;
	} else alpha = - (q*beta + gamma)/q1;
	*decr += m*(q*q - gamma*gamma/beta)/q1;
      }
      tmp = (beta - sqrt(beta*beta - 4*alpha*gamma))/2;
      a12 = gamma/tmp;
      a21 = alpha/tmp;

      for (k = 0; k < mn2; k += n2) {
	for (ii = i, jj = j; ii < ij; ii += n, jj += n) {
	  tmp = c[ii+k];
	  c[ii+k] += a12*c[jj+k];
	  c[jj+k] += a21*tmp;
	}			/* at exit ii = ij = i + jc */
	tmp = c[i+ic+k];
	c[i+ic+k] += a12*(2*c[ij+k] + a12*c[jj+k]);
	c[jj+k] += a21*c[ij+k];
	c[ij+k] += a21*tmp;	/* = element of index j,i */
	for (; ii < ic; ii += n, jj++) {
	  tmp = c[ii+k];
	  c[ii+k] += a12*c[jj+k];
	  c[jj+k] += a21*tmp;
	}
	for (; ++ii, ++jj < jc+n; ) {
	  tmp = c[ii+k];
	  c[ii+k] += a12*c[jj+k];
	  c[jj+k] += a21*tmp;
	}
      }

      for (k = 0; k < n2; k += n) {
	tmp = a[i+k];
	a[i+k] += a12*a[j+k];
	a[j+k] += a21*tmp;
      }
      det *= 1 - a12*a21;		/* compute determinant */
    }

  *logdet += 2*m*log(det);
  for (det = 1, k = 0; k < mn2; k += n2) {
    for (ii = 0; ii < n2; ii += n+1)
      det *= c[ii+k];
  }
  return log(det) - *logdet;
}
