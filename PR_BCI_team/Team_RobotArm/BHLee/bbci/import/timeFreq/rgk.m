
function [tfr,Phi,sigma,its] = rgk(s,alpha) ;

%RGK  Optimal radially Gaussian kernel time-frequency representation
%
%  Useage:    [tfr,Phi,sigma,its] = rgk(s,alpha)
%
%  Input:   - s     : column or row vector containing the signal to be
%                     analyzed
%           - alpha : normalized volume of the optimal kernel
%                     reasonable values:  1 < alpha < 5
%                     alpha = 1 => optimal kernel has same volume as a
%                     spectrogram kernel
%
%  Output:  - tfr   : optimal radially Gaussian time-frequency representation
%           - Phi   : optimal radially Gaussian kernel
%           - sigma : spread function parametrized by the radial angle in the
%                     ambiguity domain
%           - its   : number of iterations of the step-projection algorithm
%                     to converge to a (local) maximum
%
%  Example:   Two parallel linear chirps
%             t = (0:127);
%             s1 = hamming(128)' .* cos(0.2*t + 0.008*t.^2);
%             s2 = hamming(128)' .* cos(0.6*t + 0.008*t.^2);
%             s = s1 + s2;
%             tfr = rgk(s,2);
%             contour(tfr); xlabel('time'); ylabel('frequency')
%
%  See also:  AMBNB


%----------------------------------------------------------------------------%
%File Name: rgk.m
%Last Modification Date: 1/26/96        18:30:22
%Current Version: rgk.m      1.2
%File Creation Date: Sun Jan 21 16:36:09 1996
%Author: Paulo Goncalves  <gpaulo@ece.rice.edu>
%Extra Verbiage: Richard Baraniuk <richb@rice.edu>
%
%Copyright: All software, documentation, and related files in this distribution
%           are Copyright (c) 1996 Rice University
%
%Permission is granted for use and non-profit distribution providing that this
%notice be clearly maintained. The right to distribute any portion for profit
%or as part of any commercial product is specifically reserved for the author.
%
%Change History:
%
%----------------------------------------------------------------------------%



%----------------------------------------------------------------------------%
% SIGNAL-DEPENDENT TIME-FREQUENCY ANALYSIS USING A RADIAL GAUSSIAN KERNEL    %
%----------------------------------------------------------------------------%
%
% This Matlab function implements the ``Optimal Radially Gaussian Kernel
% Time-Frequency Representation.''  For details, please consult either
% the paper
%
%    R. G. Baraniuk and D. L. Jones, ``Signal-Dependent Time-Frequency
%    Analysis Using a Radially Gaussian Kernel,'' Signal Processing,
%    Vol. 32, No. 3, pp. 263-284, June 1993.
%
% or the thesis
%
%    R. G. Baraniuk, ``Shear Madness: Signal-Dependent and Metaplectic
%    Time-Frequency Representations,'' Ph.D. Thesis, Department of
%    Electrical and Computer Engineering, University of Illinois at
%    Urbana-Champaign, August 1992.  Also Coordinated Science Laboratory
%    Technical Report No. UILU-ENG-92-2226, 1992.  See Chapter 6 and
%    Appendices B and G.
%
% Equation numbers in the comments below refer to the paper unless
% otherwise noted.  We have tried to keep as close as possible to the
% notation of these documents.
%
%
% FLOW OF THE ALGORITHM:
%
% Step 1:   Compute the rectangularly sampled ambiguity function (AF)
%           of the signal
%
%           *** Uses the separate function AMBNB ***
%                (included in this distribution)
%
% Step 2:   Interpolate the AF to polar coordinates
%
% Step 3:   Solve for the optimal kernel spread vector using the

%           so-called "step-project" algorithm [Eqs. (40)-(42)]
%
% Step 4:   Compute the optimal kernel in polar coordinates
%
% Step 5:   Interpolate the optimal kernel to rectangular coordinates
%
% Step 6:   Inverse FFT the optimal-kernel x AF product to get the
%           optimal time-frequency representation
%
%
% QUESTIONS?  COMMENTS?  Drop us a line:
%
%             Paulo Goncalves    gpaulo@rice.edu
%             Richard Baraniuk   richb@rice.edu
%                                http://www-dsp.rice.edu
%
%----------------------------------------------------------------------------%




%----------------------------------------------------------------------------%
% Step 1:   Compute the rectangularly sampled ambiguity function (AF)
%           of the signal using the separate function 'ambnb'
%----------------------------------------------------------------------------%

% s is the time signal sampled at rate T.  It has L points.

s = s(:) ;
L = size(s,1) ;

% 'recamb' is the rectangularly sampled ambiguity function.
% 'tau' runs from -L.T to (L-1).T at a sample rate dtau = 2T 
% with (L) points (vertical axis)
% 'theta' runs from -pi/T to pi/T with L points at 
% a rate dtheta = (2.pi)/(L.T) (horizontal axis). T is arbitrary, 
% but we can fix it such that dtheta = dtau (square sampling) 
% which avoid us dealing with the units when calculating the 
% angle psi = Atan (m.dtau/n.dtheta) = Atan (m/n).
% (See Section 3.1 in the paper and Appendix B in the thesis.)
% Therefore dtau = dtheta ==> T = sqrt(pi/L) and
% tau = -sqrt(pi*L) ... (L-1).sqrt(pi/L) with L points and
% dtau = 2.sqrt(pi/L)
% theta = -sqrt(pi.L) ... sqrt(pi.L) with L points and 
% dtheta = dtau = 2.sqrt(pi/L).

recamb = ambnb(hilbert(s)) ;
tau = linspace(-sqrt(pi*L),(L-1)*sqrt(pi/L),L) ;
   TAU = tau(ones(L,1),:)' ; 
theta = linspace(-sqrt(pi*L),(L-1)*sqrt(pi/L),L) ;
   THETA = theta(ones(L,1),:) ; 

   
   
%----------------------------------------------------------------------------%
% Step 2:   Interpolate the AF to polar coordinates 
%           (on the upper half of the ambiguity plane!)
%----------------------------------------------------------------------------%
 
% 'polamb2' is the square of the polar coordinate ambiguity function
% estimated on the sample grid (rho,psi) where:
% 0 <= psi < pi with Q = pi.L points and dpsi = 1/L
% rho = 0 ... sqrt(2*pi*L) with P = L/sqrt(2) points and dr = 2.sqrt(pi/L)
% The max of rho corresponds to the distance between the origin and the 
% upper right corner of 'recamb'. 'polamb2' spans the circle containing the
% square spaned by 'recamb' : tau_max ~ rho_max.cos(pi/4) = sqrt(pi.L)
%                             theta_max = rho_max.sin(pi/4) = sqrt(pi/L)
% Every values of (rho,psi) corresponding to a point outside the square
% on which 'recamb' is defined, is set to zero.  The ideal value for 
% the number of polar angle samples Q is 'pi*L', but we have found that
% just Q=L works fine in practice.

P = round(L/sqrt(2)) ;
rho = linspace(0,sqrt(2*pi*L),P) ;
dr = 2*sqrt(pi/L) ;
Q = L ; 				 % ideal value would be round(pi*L) 
psi = linspace(0,pi,Q) ; %  psi = psi(1:Q) ;
dpsi = psi(2)-psi(1) ;

% 'polamb2' is obtained by a 2-d bilinear (or cubic if you choose so below)
% interpolation of the three closest 'recamb' neighbours defined on the 
% rectangular grid.  Then we square it.

polamb2 = zeros(P,Q) ;
X = rho'*cos(psi) ;
Y = rho'*sin(psi) ;

polamb2 = interp2(THETA,TAU,abs(recamb),X,Y) ;
outsquare = find(isnan(polamb2) == 1) ;
polamb2(outsquare) = zeros(size(outsquare)) ;
polamb2 = abs(polamb2/max(max(abs(polamb2)))).^2;


%----------------------------------------------------------------------------%
% Step 3:   Solve for the optimal kernel spread vector using the
%           "step-project" algorithm 
%           [See Eqs. (40)-(42) of paper and Section G.1 of thesis.]
%----------------------------------------------------------------------------%

% Parameter initialization (see Appendix G of the thesis).

Td = clock ; 
gamma = sqrt(2*pi*alpha/dpsi) ;
sigma1 = ones(1,Q)*(gamma/sqrt(Q)) ;
sigma2 = sigma1 ;
muk = 1 ; 			% Step size - controls how fast we climb
eps1 = 0.1 ;                    % Step size test parameter 1
eps2 = 1-eps1 ;                 % Step size test parameter 2
delta = 10 ;                    % Step size mulitplicative inc/dec-rement
M = 1001 ;                      % Largest possible step size
eps3 = 1e-5 ; 			% Controls how soon we quit climbing

Pindices = 0:P-1 ;
P1  = (Pindices(ones(1,Q),:)).' ;
P3  = (Pindices(ones(1,Q),2:P).^3).' ;
criterion1 = inf ; 
criterion2 = inf ;
n_iter = 0 ;
max_iter = 50 ;                 % Number of iterations before we stop 
                                % trying to reach a local max

%  Step-Project loop with adaptive step size
%  Tests step size 'muk' at each iteration.  If too large, the algorithm
%  retreats and takes a shorter step.  If too small, the algorithm takes
%  a larger step next iteration.  This method seems to work nearly all
%  the time.  When it fails, it seems to be when the kernel volume is
%  such that part of the kernel must overlay the cross-components, which
%  makes the performance surface really nasty.  Let us know if it crashes
%  on you, and we will make it work!

% Get ready for the loop
Phi2 = exp(-((rho').^2)*(sigma2.^2).^(-1)) ;
f2 = sum(sum(P1.*polamb2.*Phi2)) ;

while ((criterion1>0) | (criterion2>0)) & n_iter<=max_iter

  sigma1 = sigma2 ;
  f1 = f2 ;  
  Phi1 = exp(-((rho').^2)*(sigma1.^2).^(-1)) ; 
  grad = sum(P3.*polamb2(2:P,:).*Phi1(2:P,:)).*(2*dr^2*(sigma1).^(-3)) ;
  sigma2 = (sigma1 + muk*grad).*gamma./norm(sigma1 + muk*grad) ;
  Phi2 = exp(-((rho').^2)*(sigma2.^2).^(-1)) ;
  f2 = sum(sum(P1.*polamb2.*Phi2)) ;
  g = (f2-f1)./((sigma2-sigma1)*grad.') ;
  if g < eps1
    muk = muk/delta ;
    sigma2 = sigma1 ;
  elseif g > eps2 	
    muk = min(delta*muk,M) ;
  else
    criterion1 = norm(sigma2-sigma1)-sqrt(eps3)*(1+gamma) ;
    criterion2 = (f2-f1)-eps3*(1+f1) ;
  end
  n_iter = n_iter + 1 ;
  
end

if n_iter > max_iter
  disp(' ')
  disp('Step-project algorithm stopped before convergence'); disp(' ')
end


%----------------------------------------------------------------------------%
% Step 4:   Interpolate the spread vector sigma from polar coordinates 
%           to rectangular (+ extend to the entire ambiguity plane)
%----------------------------------------------------------------------------%

% INTERPOLATE THE SPREAD VECTOR INSTEAD OF THE KERNEL

psi = [psi psi(2:Q)+pi] ;
sigma = [sigma2 sigma2(2:Q)] ;

recANGLE = atan2(TAU,THETA) ; 		% -pi < recANGLE < pi
negangle = find(recANGLE < 0) ; 
recANGLE(negangle) = recANGLE(negangle)+2*pi ; % 0 < recANGLE < 2*pi
recRADIUS = TAU.^2+THETA.^2 ;

recsigma = interp1(psi,sigma,recANGLE(:)) ;
recsigma = reshape(recsigma,L,L) ;


%----------------------------------------------------------------------------%
% Step 5:   Compute the optimal kernel in rectangular coordinates      
%----------------------------------------------------------------------------%

recPhi_opt = exp(-recRADIUS.*((2*recsigma.^2).^(-1))) ;


%----------------------------------------------------------------------------%
% Step 6:   Inverse FFT the optimal-kernel x AF product to get the
%           optimal time-frequency representation
%----------------------------------------------------------------------------%

% Characteristic function = optimal kernel times ambiguity function
% in rectangular coordinates

recamb_opt = recamb.*recPhi_opt ;
recamb_opt = fftshift(recamb_opt) ;
  
% Optimal kernel time-frequency distribution is the 2-D FFT of the
% characteristic function.  As it stands, TFR matrix will have zero
% frequency at the bottom of the matrix -- set up for use with
% "contour," "mesh," and "pcolor".  If you like to use the "image"
% command, which flips columns, then use the commented line that 
% performs an additional "flipud".

% Zero frequency in the last column of the tfr matrix
tfr = fliplr(real(fft2(recamb_opt))) ;

% Zero frequency in the first column of the tfr matrix
%% tfr = flipud(fliplr(real(fft2(recamb_opt)))) ;

Phi = recPhi_opt ;
its = n_iter ;

