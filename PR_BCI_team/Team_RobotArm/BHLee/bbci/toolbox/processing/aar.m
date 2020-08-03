function [a,e,REV,TOC,CPUTIME,ESU] = aar(y, Mode, arg3, arg4, arg5, arg6, arg7, arg8, arg9); 
% [a,e,REV] = aar(y, mode, MOP, UC, a0, A); 
% Calculates adaptive autoregressive (AAR) and adaptive autoregressive moving average estimates (AARMA)
% of real-valued data series with Kalman filter algorithm.
%
% The AAR process is described as following  
% 	y(k) - a(k,1)*y(t-1) -...- a(k,p)*y(t-p) = e(k);
% The AARMA process is described as following  
% 	y(k) - a(k,1)*y(t-1) -...- a(k,p)*y(t-p) = e(k) + b(k,1)*e(t-1) + ... + b(k,q)*e(t-q);
%
% Input:
%       y	Signal (AR-Process)
%	Mode    is a two-element vector [aMode, vMode], 
%		aMode determines 1 (out of 12) methods for updating the co-variance matrix (see also [1])
%		vMode determines 1 (out of 7) methods for estimating the innovation variance (see also [1])
%               aMode=1, vmode=2 is the RLS algorithm as used in [2]
%		aMode=-1, LMS algorithm (signal normalized)
%		aMode=-2, LMS algorithm with adaptive normalization  
% 				      
%       MOP     model order, default [10,0] 
%               MOP=[p]		AAR(p) model. p AR parameters
%		MOP=[p,q] 	AARMA(p,q) model, p AR parameters and q MA coefficients
%	UC	Update Coefficient, default 0
%	a0	Initial AAR parameters [a(0,1), a(0,2), ..., a(0,p),b(0,1),b(0,2), ..., b(0,q)]
%		 (row vector with p+q elements, default zeros(1,p) )
%	A	Initial Covariance matrix (positive definite pxp-matrix, default eye(p))
%      
% Output:
%	a	AAR(MA) estimates [a(k,1), a(k,2), ..., a(k,p),b(k,1),b(k,2), ..., b(k,q]
%	e	error process (Adaptively filtered process)
%       REV     relative error variance MSE/MSY
%
%
% Hint:
% The mean square (prediction) error of different variants is useful for determining the free parameters (Mode, MOP, UC) 
%
% REFERENCE(S): 
% [1] A. Schloegl (2000), The electroencephalogram and the adaptive autoregressive model: theory and applications. 
%     ISBN 3-8265-7640-3 Shaker Verlag, Aachen, Germany. 
% [2] A. Schloegl, D. Flotzinger, G. Pfurtscheller, 
%     Adaptive autoregressive modelling for single trial EEG classification.
%     Biomedizinische Technik, 42: 162-167, 1997.   
% [3] A. Schloegl, S.J. Roberts, G. Pfurtscheller, 
%     A criterion for adaptive autoregressive models. 
%     Proc. of the World conference 2000 for Medical Physics and biomedical engineering. Chicago, IL, 2000. 
%
%
% More references can be found at 
%     http://www.dpmi.tu-graz.ac.at/~schloegl/publications/

%
%	Version 2.01
%       17. Oct. 2000
%	Copyright (c) 1998-2000 by  Alois Schloegl
%	a.schloegl@ieee.org	
%
% revisions:
% V2.00 17.08.2000 major revision 
%		- Mode enumeration according to [1]
% V2.01 19.10.2000 
%               - aMode 13 & 14 included

%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Library General Public
% License as published by the Free Software Foundation; either
% version 2 of the License, or (at your option) any later version.
% 
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Library General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the
% Free Software Foundation, Inc., 59 Temple Place - Suite 330,
% Boston, MA  02111-1307, USA.


% for compilation with the Matlab compiler mcc -V1.2 -ir
%#realonly 
%#inbounds


[nc,nr]=size(y);
%if nc<nr y=y'; end; tmp=nr;nc=nr; nr=tmp;end;

if nargin<2 Mode=0; end;
% check Mode (argument2)
if prod(size(Mode))==2
        aMode=Mode(1);
        vMode=Mode(2);
end;
if any(aMode==(1:12)) & any(vMode==(1:7)), 
%	fprintf(1,['a' int2str(aMode) 'e' int2str(vMode) ' ']);
else
	
fprintf(2,'Error AAR.M: invalid Mode argument\n');
	return;
end;

% check model order (argument3)
if nargin<3 MOP=[10,0]; else MOP= arg3; end;
if length(MOP)==0 p=10; q=0; MOP=p;
elseif length(MOP)==1 p=MOP(1); q=0; MOP=p;
elseif length(MOP)>=2 p=MOP(1); q=MOP(2); MOP=p+q;
end;
       
       
if nargin<4 UC=0; else UC= arg4; end;
if nargin<5 
	a0=zeros(1,MOP); 
else 
	a0=arg5;  
	if any(size(a0)~=[1,MOP]),end;
end;
if nargin<6 A0=eye(MOP); else A0= arg6;  end;
if nargin<7 TH=3; else TH = arg7;  end;
%       TH=TH*var(y);
%        TH=TH*mean(detrend(y,0).^2);
        MSY=mean(detrend(y,0).^2);

e=zeros(nc,1);
Q=zeros(nc,1);
V=zeros(nc,1);
T=zeros(nc,1);
%DET=zeros(nc,1);
SPUR=zeros(nc,1);
ESU=zeros(nc,1);
a=a0(ones(nc,1),:);
%a=zeros(nc,MOP);
%b=zeros(nc,q);

mu=1-UC; % Patomaeki 1995
lambda=(1-UC); % Schloegl 1996
arc=poly((1-UC*2)*[1;1]);b0=sum(arc); % Whale forgettting factor for Mode=258,(Bianci et al. 1997)

dW=UC/MOP*eye(MOP);                % Schloegl



%------------------------------------------------
%	First Iteration
%------------------------------------------------
Y=zeros(MOP,1);
C=zeros(MOP);
%X=zeros(q,1);
at=a0;
A=A0;
E=y(1);
e(1)=E;
V(1)=(1-UC)+UC*E*E;
ESU(1)= 1; %Y'*A*Y;

A1=zeros(MOP);A2=A1;
tic;CPUTIME=cputime;
%------------------------------------------------
%	Update Equations
%------------------------------------------------
T0=2;
        
for t=T0:nc,
        
        %Y=[y(t-1); Y(1:p-1); E ; Y(p+1:MOP-1)]
        
        if t<=p Y(1:t-1)=y(t-1:-1:1);           % Autoregressive 
        else    Y(1:p)=y(t-1:-1:t-p); 
	end;
        
        if t<=q Y(p+(1:t-1))=e(t-1:-1:1);       % Moving Average
        else    Y(p+1:MOP)=e(t-1:-1:t-q); 
	end;
        
        % Prediction Error 
        E = y(t) - a(t-1,:)*Y;
        e(t) = E;
        E2=E*E;
        
        AY=A*Y; 
        V(t) = V(t-1)*(1-UC)+UC*E2;        
        esu=Y'*AY;
        ESU(t)=esu;
                  
        if aMode == -1, % LMS 
                %	V(t) = V(t-1)*(1-UC)+UC*E2;        
                a(t,:)=a(t-1,:) + (UC/MSY)*E*Y';
        elseif aMode == -2, % LMS with adaptive estimation of the variance 
                a(t,:)=a(t-1,:) + UC/V(t)*E*Y';

        else    % Kalman filtering (including RLS) 
                if vMode==1, 		%eMode==4
                        Q(t) = (esu + V(t));      
                elseif vMode==2, 	%eMode==2
                        Q(t) = (esu + 1);          
                elseif vMode==3, 	%eMode==3
                        Q(t) = (esu + lambda);     
                elseif vMode==4, 	%eMode==1
                        Q(t) = (esu + V(t-1));           
                elseif vMode==5, 	%eMode==6
                        if E2>esu 
                                V(t)=(1-UC)*V(t-1)+UC*(E2-esu);
                        else 
                                V(t)=V(t-1);
                        end;
                        Q(t) = (esu + V(t));           
                elseif vMode==6, 	%eMode==7
                        if E2>esu 
                                V(t)=(1-UC)*V(t-1)+UC*(E2-esu);
                        else 
                                V(t)=V(t-1);
                        end;
                        Q(t) = (esu + V(t-1));           
                elseif vMode==7, 	%eMode==8
                        Q(t) = esu;
                end;
        
                k = AY / Q(t);		% Kalman Gain
                a(t,:) = a(t-1,:) + k'*E;
                
                if aMode==1, 			%AMode=1
                        A = (1+UC)*(A - k*AY');                   % Schloegl et al. 1997
                elseif aMode==2, 		%AMode=11
                        A = A - k*AY';
                        A = A + sum(diag(A))*dW;
                elseif aMode==3, 		%AMode=5
                        A = A - k*AY' + sum(diag(A))*dW;
                elseif aMode==4, 		%AMode=6
                        A = A - k*AY' + UC*eye(MOP);               % Schloegl 1998
                elseif aMode==5, 		%AMode=2
                        A = A - k*AY' + UC*UC*eye(MOP);
                elseif aMode==6, 		%AMode=2
                        T(t)=(1-UC)*T(t-1)+UC*(E2-Q(t))/(Y'*Y);  
                        A=A*V(t-1)/Q(t);  
                        if T(t)>0 A=A+T(t)*eye(MOP); end;          
                elseif aMode==7, 		%AMode=6
                        T(t)=(1-UC)*T(t-1)+UC*(E2-Q(t))/(Y'*Y);      
                        A=A*V(t)/Q(t);  
                        if T(t)>0 A=A+T(t)*eye(MOP); end;          
                elseif aMode==8, 		%AMode=5
                        Q_wo = (Y'*C*Y + V(t-1));                
                        C=A-k*AY';
                        T(t)=(1-UC)*T(t-1)+UC*(E2-Q_wo)/(Y'*Y);      
                        if T(t)>0 A=C+T(t)*eye(MOP); else A=C; end;          
                elseif aMode==9, 		%AMode=3
                        A = A - (1+UC)*k*AY';
                        A = A + sum(diag(A))*dW;
                elseif aMode==10, 		%AMode=7
                        A = A - (1+UC)*k*AY' + sum(diag(A))*dW;
                elseif aMode==11, 		%AMode=8
                        A = A - (1+UC)*k*AY' + UC*eye(MOP);        % Schloegl 1998
                elseif aMode==12, 		%AMode=4
                        A = A - (1+UC)*k*AY' + UC*UC*eye(MOP);
                elseif aMode==13
                        A = A - k*AY' + UC*diag(diag(A));
                elseif aMode==14
                        A = A - k*AY' + (UC*UC)*diag(diag(A));
                end;
        end;
end;
%a=a(end,:);
TOC=toc;
CPUTIME=cputime-CPUTIME;

REV = (e'*e)/(y'*y);

