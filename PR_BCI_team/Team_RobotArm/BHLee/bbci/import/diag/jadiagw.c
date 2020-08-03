/* Same as jadiag but with weighs */

#define	REAL	double

#include <math.h>
#define	TINY	10e-20
REAL jadiagw(REAL c[], REAL w[], int n, int m, REAL a[],
	     REAL *logdet, REAL *decr)
/*
  Performs joint approximate diagonalization of m matrices with
  weighs. These matrices are concatenated into a n by (n*m) matrix and
  are stored column wise as a vector c[n*n*m]. The set of weighs used
  in the criterion is given by the vectors w[m]. They are tranformed
  to more diagonal in one iteration sweep. Only the lower triangular
  elements of these matrices are used and changed. The transformation
  is also appliied (through premultipliation only) to the matrix a
  (which is also stored column wise as a vector) and the variable
  logdet is added by twice the sum of weighs times the logarithm of
  the determinant of the transformation. The function returns the
  criterion, which is the weighted sum of the logarithm of the product
  of the diagonal elements of each transformed matrices, minus the new
  value of logdet. Finally, decr yields an estimate of the decrease of
  the criterion near the solution (it is more accurate for small
  decrease since taking the difference of the criterion can result in
  large rounding error)
*/
{
  int	i1,j1;
  int	n2 = n*n, mn2 = m*n2,
	i, ic, ii, ij, j, jc, jj, k, k0;
  REAL  sumweigh, p2, q1, p, q,
	alpha, beta, gamma, a12, a21, tiny, det;
  register REAL tmp1, tmp2, tmp, weigh;

  for (sumweigh = 0, i = 0; i < m; i++)
    sumweigh += w[i];

  det = 1;
  *decr = 0;
  for (i = 1, ic = n; i < n ; i++, ic += n)
    for (j = jc = 0; j < i; j++, jc += n) {
      ii = i + ic;
      jj = j + jc;
      ij = i + jc;
      for (q1 = p2 = p = q = 0, k0 = k = 0; k0 < m; k0++, k += n2) {
	weigh = w[k0];
	tmp1 = c[ii+k];
	tmp2 = c[jj+k];
	tmp = c[ij+k];
	p += weigh*tmp/tmp1;
	q += weigh*tmp/tmp2;
	q1 += weigh*tmp1/tmp2;
	p2 += weigh*tmp2/tmp1;
      }
      q1 /= sumweigh;
      p2 /= sumweigh;
      p /= sumweigh;
      q /= sumweigh;
      beta = 1 - p2*q1;			/* p1 = q2 = 1 */
      if (q1 <= p2) {			/* the same as q1*q2 <= p1*p2 */
	alpha = p2*q - p;		/* q2 = 1 */
	if (fabs(alpha) - beta < TINY) {	/* beta <= 0 always */
	  beta = -1;
	  gamma = p/p2;
	} else gamma = - (p*beta + alpha)/p2;	/* p1 = 1 */
	*decr += sumweigh*(p*p - alpha*alpha/beta)/p2;
      } else {
	gamma = p*q1 - q;		/* p1 = 1 */
	if (fabs(gamma) - beta < tiny) {	/* beta <= 0 always */
	  beta = -1;
	  alpha = q/q1;
	} else alpha = - (q*beta + gamma)/q1;	/* q2 = 1 */
	*decr += sumweigh*(q*q - gamma*gamma/beta)/q1;
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

  *logdet += 2*sumweigh*log(det);
  for (tmp = 0, k0 = k = 0; k0 < m; k0++, k += n2) {
    for (det = 1, ii = 0; ii < n2; ii += n+1)
      det *= c[ii+k];
    tmp += w[k0]*log(det);
  }
  return tmp - *logdet;
}
