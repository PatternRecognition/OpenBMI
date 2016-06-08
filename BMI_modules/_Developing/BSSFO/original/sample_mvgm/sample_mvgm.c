/* sample_mvgm.c
 
 
  Draw samples from a mixture of multivariate Gaussian PDF
 
 Usage      : [Z , index] = sample_mvgm([N] , [mu] , [sigma] , [p] , [v1] , ... , [vp])
 ------

 Inputs
 -------
 
     N      : Number of samples to draw (default N = 1)
     mu     : Mean vector (d x 1 x M x [n1] , ... , [nl]) (default mu = 0.0)
     sigma  : Covariance  (d x d x M x [n1] , ... , [nl]) (default sigma = eye(d))
     p      : Weights vector (1 x 1 x M x [n1] , ... , [nl]) such that sum(p , 3) = 1 (default p = 1)
	 vi     : Dimension of slice i

 
 
 Outputs
 -------
 
    Z        : Gaussian mixture samples (d x N x [n1] , ... , [nl] x [v1] , ... , [vp])
    index    : Index of each compounds (1,...,M) (d x N x [n1] , ... , [nl] x [v1] , ... , [vp])
 
 
 Example1
 --------
 
 mu          = cat(3 , [-5 ; -5] , [0 ; 0] ,[ 5 ; 5]);                %(d x 1 x M)
 sigma       = cat(3 , [2 0; 0 1] , [2 -.2; -.2 2] , [1 .9; .9 1]  ); %(d x d x M)
 p           = cat(3 , [0.3] , [0.2]  , [0.5]);                       %(1 x 1 x M)
 N           = 500;
 [Z , index] = sample_mvgm(N , mu , sigma , p);
 [x , y]     = ndellipse(mu , sigma);
 plot(Z(1 , :) , Z(2 , :) , 'k+', x , y , 'g' , 'markersize' , 2 , 'linewidth' , 2);
 hold on
 plot(reshape(mu(1 , : , :) , 1 , 3) , reshape(mu(2 , : , :) , 1 , 3) , 'r+' , 'markersize' , 6);
 hold off
 
 
 Example2
 --------
 
 N           = 100;
 V           = 3;
 L           = 2;
 mu          = repmat(cat(3 , [-5 ; -5] , [0 ; 0] ,[ 5 ; 5]) , [1 , 1 , 1 , V]) + 1*randn(2 , 1 ,3 , V);                %(d x 1 x M)
 sigma       = repmat(cat(3 , [2 0; 0 1] , [2 -.2; -.2 2] , [1 .9; .9 1]) , [1 , 1 , 1 , V]);   %(d x d x M)
 p           = repmat(cat(3 , [0.3] , [0.2]  , [0.5]) , [1 , 1 , 1 , V]);                       %(1 x 1 x M)
 [Z , index] = sample_mvgm(N , mu , sigma , p , L);


 Example3
 --------

 Z           = sample_mvgm;
 
 To compile
 ----------
 
 mex -DranSHR3 -output sample_mvgm.dll sample_mvgm.c or mex -DranKISS -outputsample_mvgm.dll sample_mvgm.c
 
 Myself, I use Intel CPP compiler as :
 
 
 mex -DranKISS -f mexopts_intel10.bat -output sample_mvgm.dll sample_mvgm.c
 
 or
 
 mex -DranSHR3 -f mexopts_intel10.bat -output sample_mvgm.dll sample_mvgm.c
 
 Ver 1.5
           V 1.5 (19/10/10)   Fixed a bug for Linux 64
 
           V 1.4 (10/07/09)   A more general Call syntax, i.e. Z  = sample_mvgm for a one simple random gaussian

           V 1.3 (11/12/07)   A more general Call syntax, i.e. Z  = sample_mvgm(N , mu , sigma) for a simple multivariate distribution
 
           V 1.2 (06/05/05)   General Call syntax.
 
           V 1.1 (03/26/05)   Bug fix :  Now Z  = sample_mvgm(N , mu , sigma , []); works if mu is (d x 1) & sigma is (d x d)
                               It permits to draw samples from a simple Multivariate Gaussian pdf instead of a Mixture of Multivariate Gaussian pdf
 
           V 1.0 (03/04/05)   Initial release
 
 
  Author : Sébastien PARIS  © (sebastien.paris@lsis.org)
 
 
 */


#include <math.h>
#include <time.h>
#include "mex.h"



/*---------------- Basic generators definition ------------------- */

#define mix(a , b , c) \
{ \
a -= b; a -= c; a ^= (c>>13); \
b -= c; b -= a; b ^= (a<<8); \
c -= a; c -= b; c ^= (b>>13); \
a -= b; a -= c; a ^= (c>>12);  \
b -= c; b -= a; b ^= (a<<16); \
c -= a; c -= b; c ^= (b>>5); \
a -= b; a -= c; a ^= (c>>3);  \
b -= c; b -= a; b ^= (a<<10); \
c -= a; c -= b; c ^= (b>>15); \
}

#define zigstep 128 /* Number of Ziggurat'Steps */


#define znew   (z = 36969*(z&65535) + (z>>16) )

#define wnew   (w = 18000*(w&65535) + (w>>16) )

#define MWC    ((znew<<16) + wnew )

#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )

#define CONG   (jcong = 69069*jcong + 1234567)

#define KISS   ((MWC^CONG) + SHR3)




#ifdef ranKISS

#define randint KISS

#define rand() (randint*2.328306e-10)

#endif



#ifdef ranSHR3

#define randint SHR3

#define rand() (0.5 + (signed)randint*2.328306e-10)

#endif

/*--------------------------------------------------------------- */


#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif


/*--------------------------------------------------------------- */


static UL jsrseed = 31340134 , jsr;

#ifdef ranKISS

static UL z=362436069, w=521288629, jcong=380116160;

#endif


static UL iz , kn[zigstep];

static long hz;

static double wn[zigstep] , fn[zigstep];



 /*--------------------------------------------------------------- */


void randini(void);

void randnini(void);


double nfix(void);

double randn(void);


void matvect(double * , double * , double *, int , int , int);


void chol(double * , double * , int  , int);



void sample_mvgm(double * , double * , double * , int , int , int , int , int ,
double * , double * ,
double * , double * , double *);



 /*--------------------------------------------------------------- */



void mexFunction( int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[] )
{
    
    
    double *mu , *sigma , *p;
    
    double *Z , *index;
    
    double *choles ,*v , *b;
    
    const int  *dimsmu , *dimssigma , *dimsp;
    
    int *dimsZ;
    
    int  numdimsmu , numdimssigma , numdimsp;
    
    int numdimsZ = 2;
    
    int  i , d , N=1 , K=1 , V =1 , M = 1 , cteM = 0;
    
    
     /* Check input */


     /* Input 1 */

       
    if(nrhs > 0)
        
    {

		N               = (int) mxGetScalar(prhs[0]);

    }
    
       
    
    
    
    
     /* Input 2 */

	if(nrhs < 2)
	{
		
		d           = 1;

		M           = 1;

		K           = 1;

		numdimsmu   = 2;

		mu          = (double *)mxMalloc(sizeof(double));

		mu[0]       = 0.0;
		
	}
    else
	{
		
		mu           = mxGetPr(prhs[1]);
		
		numdimsmu    = mxGetNumberOfDimensions(prhs[1]);
		
		dimsmu       = mxGetDimensions(prhs[1]);
		
		if ( (dimsmu[1] != 1))
			
		{
			mexErrMsgTxt("mu must be (d x 1 x M x n1 x ... x nl)");
		}
		
		d                 = dimsmu[0];
		
		if (numdimsmu > 2)
			
		{
			
			M              = dimsmu[2];
			
			numdimsZ      += (numdimsmu - 3);
			
		}
		
		for(i = 3 ; i < numdimsmu ; i++)
			
		{
			
			K              *= dimsmu[i];
			
			
		}
	}
    
    
     /* Input 3 */
    
	if(nrhs < 3)
	{
		
		sigma           = (double *)mxMalloc(d*d*sizeof(double));

		numdimssigma    = 2;

		for (i = 0 ; i < d*d ; i++)
		{

			sigma[i]      = 0.0;

		}

		for (i = 0 ; i < d ; i++)
		{
				
			sigma[i*(d+1)] = 1.0;

		}

		
	}
	
	else
	{
		
		sigma           = mxGetPr(prhs[2]);
		
		numdimssigma    = mxGetNumberOfDimensions(prhs[2]);
		
		dimssigma       = mxGetDimensions(prhs[2]);
		
		if (  (dimssigma[0] !=d ) && (dimssigma[1] != d) )
			
		{
			mexErrMsgTxt("sigma must be (d x d x M x n1 x ... x nl)");
		}
		
	}
    
    
    
     /* Input 4 */
    
    
    if (nrhs > 3)
        
    {
              
        numdimsp        = mxGetNumberOfDimensions(prhs[3]);
        
        dimsp           = mxGetDimensions(prhs[3]);
        
        
        if ( numdimsp != numdimsmu) 
            
        {
            
            mexErrMsgTxt("p must be (1 x 1 x M x n1 x ... x nl)");
            
            
        }
        
        
        if ( (dimsp[0] == 0) && (dimsp[1] == 0 ) ) 
            
        {
            
            p               = (double *)mxMalloc(sizeof(double));
            
            p[0]            = 1.0;
            
            M               = 1;
                     
            
        }
        
        if  ( (dimsp[0] == 1) && (dimsp[1] == 1 ) )
            
        {
            
            p               = mxGetPr(prhs[3]);
            
        }
        
        
    }
    
    else
        
    {
              
        p    = (double *)mxMalloc(sizeof(double));
        
        p[0] = 1.0;
        
        M    = 1;
        
    }
    
    if (nrhs > 4)
    {
        
        numdimsZ      += (nrhs - 4);
        
    }
    
     /* Output 1 */
    
    
    
    dimsZ         = (int *)mxMalloc(numdimsZ*sizeof(int));
    
    dimsZ[0]      = d;
    
    dimsZ[1]      = N;
    
    
    for(i = 3 ; i < numdimsmu ; i++)
        
    {
        
        dimsZ[i - 1] = dimsmu[i];
        
    }
    
    
    if (M > 1)
    {
        
        cteM = 1;
        
    }
    
    
    for (i = 4 ; i < nrhs ; i++)
        
    {
        
        dimsZ[(numdimsmu - 2 - cteM) + i  - 2  ] = (int) mxGetScalar(prhs[i]) ;
        
        V                                       *= dimsZ[(numdimsmu - 2 - cteM) + i  - 2  ];
        
    }
    
    
    
    /* Output 1 */
    
    
    
    plhs[0]        = mxCreateNumericArray(numdimsZ , dimsZ, mxDOUBLE_CLASS, mxREAL);
    
    Z              = mxGetPr(plhs[0]);
    
    
    dimsZ[0]       = 1;
    
    plhs[1]        = mxCreateNumericArray(numdimsZ , dimsZ, mxDOUBLE_CLASS, mxREAL);
    
    index          = mxGetPr(plhs[1]);
    
    
     /* vecteur temporaire */
    
       
    choles         = (double *)mxMalloc((d*d*M*K)*sizeof(double));   
    
    v              = (double *)mxMalloc((d)*sizeof(double));
    
    b              = (double *)mxMalloc((d)*sizeof(double));
    
    
    
     /* Rand ~U[0,1] Seed initialization */
    
    
    randini();
    
    /* Initialize Ziggurat Table with zigstep steps for Normal(0,1) */
    
    randnini();
    
    /* Main call */
    
    
    sample_mvgm(mu , sigma , p , d , M , N , K , V , Z , index , choles , v , b);
    
      
     /* Free ressources */
    
    
    mxFree(choles);
    
    mxFree(v);
    
    mxFree(b);
    
    mxFree(dimsZ);
    
    if (nrhs > 3)
        
    {
        
        
        if ( (dimsp[0] == 0) && (dimsp[1] == 0 ) ) 
            
        {
            
            mxFree(p);
            
        }
		
	}
	
	else
		
	{
		
		mxFree(p);
		
	}
	
	if(nrhs < 3)
	{
		mxFree(sigma);
		
	}
	
	
	if(nrhs < 2)
	{
		mxFree(mu);
		
	}
	
    
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */



void sample_mvgm(double *mu , double *sigma , double *p , int d , int M , int N , int K , int V ,
double *Z , double *index ,
double *choles , double *v , double *b)


{
    
    int h , l , i , j , jd , d2 = d*d, val , KN = K*N , hKN , hdKN , lM , lN , ldN , ii , dN = d*N , dM = d*M , ldM , lddM;
    
    double temp , cP;
    
    
    /* Compute choles=chol(sigma)'; */
    
    
    chol(sigma , choles , d , M*K);
    
    for (h = 0 ; h < V ; h++)
        
    {
        
        hKN  = h*KN;
        
        hdKN = d*hKN;
        
        for (l = 0 ; l < K ; l++)
            
        {
            
            lM   = l*M;
            
            lN   = l*N + hKN;
            
            ldN  = l*dN + hdKN;
            
            ldM  = l*dM;
            
            lddM = d*ldM;
            
            for (j = 0 ; j < N ; j++)
                
            {
                
                temp = rand();
                
                val  = 1;
                
                cP   = p[0 + lM];
                
                while( (temp > cP) && (val < M))
                {
                    
                    cP  +=p[val + lM];
                    
                    val++;
                    
                }
                
                index[j + lN] = val;
                
                
                for (i = 0 ; i < d ; i++)
                    
                {
                    
                    v[i] = randn();
                    
                }
                
                matvect(choles , v , b , d , d , (val - 1)*d2 + lddM);
                
                ii     = (val - 1)*d + ldM;
                
                jd     = j*d + ldN;
                
                
                for (i = 0 ; i < d ; i++)
                    
                {
                    
                    Z[i + jd] = b[i] + mu[i + ii];
                    
                }
                
            }
            
        }
        
    }
}

/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */

void matvect(double *A , double *v , double *w, int d , int n , int off)

 /*
  
   w = Av, A(d x n), v(n x 1)
  
  */

{
    
    int t , i ;
    
    register double temp;
    
    
    for (t = 0 ; t < d ; t++)
    {
        
        temp   = 0.0;
        
        for(i = 0 ; i < n ; i++)
        {
            
            temp += A[t + i*d + off]*v[i];
        }
        
        w[t] = temp;
        
    }
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */

void chol(double *Q , double *D , int d , int M)

{
    
    int i , j , r , d2=d*d;
    
    int id , d1 = d - 1 , i1 , i1d , l , knnn , jd , jv , v , iv;
    
    double sum , p , inv_p;
    
    
    for (r = 0 ; r  < M ; r++)
        
    {
        
        v = r*d2;
        
        for (i = 0 ; i < d2 ; i++)
            
        {
            
            D[i + v]    = Q[i + v];
            
        }
        
        
        p           = sqrt(D[0 + v]);
        
        inv_p       = 1.0/p;
        
        D[0 + v]    = p;
        
        
        for(i = 1 ; i < d; i++)
            
        {
            
            D[d*i + v]  *= inv_p;
            
        }
        
        
        for(i = 1 ; i < d; i++)
            
        {
            id   = i*d;
            
            i1d  = id - d;
            
            i1   = i - 1;
            
            iv   = i + v;
            
            sum  = D[iv + id];    /* sum = B[i][i] */
            
            for(l = 0; l < i; ++l)
                
            {
                knnn = id + l;
                
                sum -= D[knnn + v]*D[knnn + v];
            }
            
            p     = sqrt(sum);
            
            inv_p = 1.0/p;
            
            for(j = d1; j > i ; --j)
            {
                jd   = j*d;
                
                sum  = D[jd + iv];
                
                for(l = 0; l < i ; ++l)
                    
                {
                    sum   -= D[jd + l + v]*D[id + l + v];
                }
                
                
                D[jd + iv] = sum*inv_p;
                
            }
            
            D[iv + id] = p;
            
            for(l = d1  ; l>i1 ; l--)
            {
                
                D[l + i1d + v] = 0.0;
            }
        }
        
        /* D = D'; */
        
        for (j = 0 ; j < d ; j++)
            
        {
            jd = j*d + v;
            
            jv = j + v;
            
            for(i = j + 1 ; i < d ; i++)
                
            {
                
                D[i + jd]   = D[jv + i*d];
                
                D[jv + i*d] = 0.0;
                
            }
        }
        
    }
    
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */



void randini(void)

{
    
     /* SHR3 Seed initialization */
    
    jsrseed  = (UL) time( NULL );
    
    jsr     ^= jsrseed;
    
    
     /* KISS Seed initialization */
    
    #ifdef ranKISS
    
    z        = (UL) time( NULL );
    
    w        = (UL) time( NULL );
    
    jcong    = (UL) time( NULL );
    
    mix(z , w , jcong);
    
    #endif
    
    
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */

void randnini(void)
{
    register const double m1 = 2147483648.0;
    
    register double  invm1;
    
    register double dn = 3.442619855899 , tn = dn , vn = 9.91256303526217e-3 , q;
    
    int i;
    
    
    /* Ziggurat tables for randn */
    
    invm1             = 1.0/m1;
    
    q                 = vn/exp(-0.5*dn*dn);
    
    kn[0]             = (dn/q)*m1;
    
    kn[1]             = 0;
    
    wn[0]             = q*invm1;
    
    wn[zigstep - 1 ]  = dn*invm1;
    
    fn[0]             = 1.0;
    
    fn[zigstep - 1]   = exp(-0.5*dn*dn);
    
    for(i = (zigstep - 2) ; i >= 1 ; i--)
    {
        dn              = sqrt(-2.*log(vn/dn + exp(-0.5*dn*dn)));
        
        kn[i+1]         = (dn/tn)*m1;
        
        tn              = dn;
        
        fn[i]           = exp(-0.5*dn*dn);
        
        wn[i]           = dn*invm1;
    }
    
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */


double nfix(void)
{
    const double r = 3.442620; 	/* The starting of the right tail */
    
    static double x, y;
    
    for(;;)
        
    {
        
        x = hz*wn[iz];
        
        if(iz == 0)
            
        {	/* iz==0, handle the base strip */
            
            do
            {
                x = -log(rand())*0.2904764;  /* .2904764 is 1/r */
                
                y = -log(rand());
            }
            
            while( (y + y) < (x*x));
            
            return (hz > 0) ? (r + x) : (-r - x);
        }
        
        if( (fn[iz] + rand()*(fn[iz-1] - fn[iz])) < ( exp(-0.5*x*x) ) )
            
        {
            
            return x;
            
        }
        
        
        hz = randint;
        
        iz = (hz & (zigstep - 1));
        
        if(abs(hz) < kn[iz])
            
        {
            return (hz*wn[iz]);
            
        }
        
        
    }
    
}


/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */


double randn(void)

{
    
    hz = randint;
    
    iz = (hz & (zigstep - 1));
    
    return (abs(hz) < kn[iz]) ? (hz*wn[iz]) : ( nfix() );
    
};



/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------- */
