% This program performs the Empirical Mode Decomposition accordingly to the paper 
%       “On the HHT, its problems, and some solutions”, Reference: Rato, R. T., Ortigueira, M. D., and Batista, A. G., 
%       Mechanical Systems and Signal Processing , vol. 22, no. 6, pp. 1374-1394, August 2008.
%
%
%   Authors: Raul Rato (rtr@uninova.DOT.pt) and Manuel Ortigueira (mdortigueira@uninova.pt or mdo@fct.unl.pt) 
%--------------------------------------------------------------------------
%
%rParabEmd__L: Emd parabolic decomposition with extrapolated extrema           v1.01
%                                                                   Build 20070717001
%
%   Usage:  rParabEmd= rParabEmd__L(x,qResol, qResid, qAlfa);
%           x      - input signal - must be a real vector
%           qResol - Resolution (in DBs: 10*log(WSignal/Bias energy))- normally between 40 and 60 dB 
%           qResid - Residual energy (in DBs: 10*log (WSignal/WqResidual))- normally between 40 and 60 dB
%           qAlfa  - Gradient step size (normally is set to 1)
%           rParabEmd    - relation matrix of IMF modes  (each as a line)
%                   with residual in last line.
%
%   Limitations:    NaN is not trapped
%
%   History:    V1.00 First version
%               V1.01 Count mismatch detection (Line 44) increased from 1 to 2
%
% WARNING: This software is a result of our research work and is supplied without any garanties.
%           We would like to receive comments on the results and report on bugs.
%
%           /* NoSPAM: Replace .DOT. with a dot (.) */
%                   (c) LaPAS-2007

function rParabEmd = rParabEmd__L (x, qResol, qResid, qAlfa)

dbstop if warning
if(nargin~=4), error('rParabEmd__L: Use with 4 inputs.'), end
if(nargout>1), error('rParabEmd__L: Use with just one output.'), end
ArgCheck_s(x, qResol, qResid, qAlfa)

% Actual computation -------------------------------------
kc = x(:);                  % ket copy of the input signal
Wx= kc'*kc;                 % Original signal energy
quntN = length(kc);         % Signal length
% loop to decompose the input signal into successive IMFs
rParabEmd= [];    % Matrix which will contain the successive IMFs, and the residue
rParabEmdCnt= 0;
qDbResid= 0;                 %Equal energies at start
quntOscCnt= quntNOsc_s(kc);
while ((qDbResid<qResid) && (quntOscCnt>2) )   % c has some energy and oscilates
    kImf = kc; % at the beginning of the sifting process, kImf is the signal
    rPMOri= rGetPMaxs_s(kImf);     % rPM= [xM(M), yM(M)];
    rPmOri= rGetPMins_s(kImf);     % rPm= [xm(m), ym(m)];
    rPM= rPMaxExtrapol_s(rPMOri, rPmOri, quntN);
    rPm= rPMinExtrapol_s(rPMOri, rPmOri, quntN);
    quntLM= length(rPM);   quntLm= length(rPm);
%    if (abs(quntLM-quntLm)>2), disp('Debug: Max-Min count mismatch.'),keyboard,end;
    if (abs(quntLM-quntLm)>2), disp('Debug: Max-Min count mismatch.'),end;
    if(sum(abs(diff(sign(rPM(1:min(quntLM,quntLm),1)- rPm(1:min(quntLM,quntLm),1)))))>0)
%        disp('Debug: Max-Min sequence mismatch.'),keyboard;
        disp('Debug: Max-Min sequence mismatch.');
    end
    if(sum(abs(diff(sign(rPm(1:min(quntLM,quntLm),1)- rPM(1:min(quntLM,quntLm),1)))))>0)
%        disp('Debug: Max-Min reverse sequence mismatch.'),keyboard;
        disp('Debug: Max-Min reverse sequence mismatch.');
    end
    bTenv= spline(rPM(:,1), rPM(:,2), 1:quntN);          %  Top envelop: bTenv[n];
    bDenv= spline(rPm(:,1), rPm(:,2), 1:quntN);          % Down envelop: bDenv[n];
    bBias= (bTenv+bDenv)/2;               %  first bias estimate
    while true(1)             % inner loop to find each IMF
        WImf= kImf'*kImf;                %current IMF  energy
        WBias= bBias*bBias';                  %bias energy
        if WBias*WImf<0 , warning('rParabEmd__L: Ooops, negative energy detected.'), end
        if WBias> 0, DbqResol= 10*log10(WImf/WBias); else DbqResol= Inf; end
        if (DbqResol>qResol),  break, end %Resolution reached
        %Resolution not reached. More work is needed
        kImf = kImf- qAlfa*bBias';                % subtract qAlfa bias from kImf
        rPMOri= rGetPMaxs_s(kImf);     % rPM= [xM(M), yM(M)];
        rPmOri= rGetPMins_s(kImf);     % rPm= [xm(m), ym(m)];
        rPM= rPMaxExtrapol_s(rPMOri, rPmOri, quntN);
        rPm= rPMinExtrapol_s(rPMOri, rPmOri, quntN);
        bTenv= spline(rPM(:,1), rPM(:,2), 1:quntN);          % Top envelop: bTenv[n];
        bDenv= spline(rPm(:,1), rPm(:,2), 1:quntN);          % Down envelop: bDenv[n];
        bBias= (bTenv+bDenv)/2;               %  new bias estimate
    end % Wend true
    %
    rParabEmd = [rParabEmd; kImf'];          % store the extracted rParabEmd in the matrix rParabEmd
    kc = kc - kImf;             % subtract the extracted rParabEmd from the signal
    quntOscCnt= quntNOsc_s(kc);

    rParabEmdCnt=rParabEmdCnt+1;
    if (kc'*kc)>0
        qDbResid= 10*log10(Wx/(kc'*kc));
    else
        qDbResid = Inf
    end
    %
end % Wend ((DbR... ))
if ((kc'*kc)/Wx)>(10^-12)
    rParabEmd=[rParabEmd; kc'];        %The residual is the last IMF
    rParabEmdCnt=rParabEmdCnt+1;
    NumOscqResiduais= quntNOsc_s(kc);
end
 rParabEmd= rParabEmd';
 
end %main function

%SubFunctions ------------------------------------------------------------
%-------------------------------------------------------------------------

function ArgCheck_s(x, qResol, qResid, qAlfa)

[qL, qC] = size(x);
if ((qL*qC)~= max(qL,qC)), error('rParabEmd__L: Input signal must be a one dim vector.'), end
if ((qL*qC)<= 1), error('rParabEmd__L: Input signal must be a vector.'), end

[qL,qC] = size(qResol);
if ( ~((qL==1)&(qC==1)) ), error('rParabEmd__L: Input resolution must be a scalar.'), end
if ( qResol<=0 ), error('rParabEmd__L: Input resolution must strictly positive.'), end

[qL,qC] = size(qResid);
if ( ~((qL==1)&(qC==1)) ),  error('rParabEmd__L: Input residual must be a scalar.'),  end
if ( qResid<=0 ), error('rParabEmd__L: Input residual must strictly positive.'), end

[qL,qC] = size(qAlfa);
if ( ~((qL==1)&(qC==1)) ), error('rParabEmd__L: qAlfa step must be a scalar.'), end
if ( qAlfa<=0 ), error('rParabEmd__L: qAlfa step  must be strictly positive.'),  end
end

%--------------------------------------------------------------------------
%---------- make at 17-Jul-07 10:16:59.44 
% quntNOsc_s           v1.01
%                   build  20070409001
%      Returns the oscilation count, no steps
function quntNOsc = quntNOsc_s (x)

y=0;    qisTop= false; qisDown= false;


for i=2:(length(x)-1)
    if( ((x(i-1)) < (x(i))) && ((x(i+1))< (x(i))) )  %Max /-\
        y=y+1;
    end
    if( ((x(i-1)) > (x(i))) && ((x(i+1))> (x(i))) )  %min \_/
        y=y+1;
    end
    
%Top     
    if( ((x(i-1)) < (x(i))) && ((x(i+1))== (x(i))) ) %StepL /-
         qisTop= true; qisDown= false;
    end
    if( ((x(i-1)) == (x(i))) && ((x(i+1))< (x(i))) ) %stepR -\
        if qisTop;     y=y+1; end;
        qisTop= false;
    end
    
%Downs   
    if( ((x(i-1)) > (x(i))) && ((x(i+1))== (x(i))) ) %stepL \_
        qisTop= false; qisDown= true;
    end
    if( ((x(i-1)) == (x(i))) && ((x(i+1))> (x(i))) ) %StepR _/
        if qisDown; y=y+1; end
        qisDown=false;
    end
end % for i=2:(length(x)-1)
quntNOsc= y;
end % function y = quntNOsc_s (x)
%---------- make at 17-Jul-07 10:16:59.44 
function rPMaxExtrapol= rPMaxExtrapol_s(rPM, rPm, quntL)
%rPMaxExtrapol_s                                             V1.00
%                                               build 2007407001
% Time-mirrored top extrema (Parabolic Maxs) extrapolation

%Init ------------------------------------
rPM= sortrows(rPM); %assumes nothing on rPM sort order
rPm= sortrows(rPm); %assumes nothing on rPm sort order

kTopTim1= rPM(:,1); kTopVal= rPM(:,2);
kDwnTim1= rPm(:,1); kDwnVal= rPm(:,2);

%Start extrapolation ---------------------
if ( (kTopTim1(1)== 1) && (kDwnTim1(1)== 1) )   
    disp ('rPMaxExtrapol_s: Poliextrema at signal''s start');
elseif ( (kTopTim1(1)<1) || (kDwnTim1(1)< 1) )   
    disp ('rPMaxExtrapol_s: Invalid extrema at signal''s start');
else
    kTopTim1=[2-kDwnTim1(1); kTopTim1];     % New first Top at the (one based) specular Min
    kTopVal=[kTopVal(1); kTopVal];          % Same Val as old first Top
end

% End extrapolation -----------------------
if ( (kTopTim1(end)== quntL) && (kDwnTim1(end)== quntL) )   
    disp ('rPMaxExtrapol_s: Poliextrema at signal''s end');
elseif ( (kTopTim1(end)> quntL) || (kDwnTim1(end)> quntL) )   
    disp ('rPMaxExtrapol_s: Invalid extrema at signal''s end');
else
    kTopTim1=[kTopTim1; (2*quntL - kDwnTim1(end))];     % New last Top at the specular Min
    kTopVal=[ kTopVal; kTopVal(end)];          % Same Val as old last Top 
end

% return value ------------------------
rPMaxExtrapol= sortrows([kTopTim1, kTopVal]); 

end
%-------------------------------------------------------------------------
%---------- make at 17-Jul-07 10:16:59.44 
function rPMinExtrapol= rPMinExtrapol_s(rPM, rPm, quntL)
%rPMinExtrapol_s                                           V1.00
%                                               build 2007407001
% Time-mirrored down extrema (Parabolic Mins) extrapolation

%Init ------------------------------------
rPM= sortrows(rPM); %assumes nothing on rPM sort order
rPm= sortrows(rPm); %assumes nothing on rPm sort order

kTopTim1= rPM(:,1); kTopVal= rPM(:,2);
kDwnTim1= rPm(:,1); kDwnVal= rPm(:,2);

%Start extrapolation ---------------------
if ( (kTopTim1(1)== 1) && (kDwnTim1(1)== 1) )
    disp ('rPMinExtrapol_s: Poliextrema at signal''s start');
elseif ( (kTopTim1(1)<1) || (kDwnTim1(1)< 1) )
    disp ('rPMinExtrapol_s: Invalid extrema at signal''s start');
else
    kDwnTim1=[2-kTopTim1(1); kDwnTim1];     % New first Dwn at the (one based) specular Max
    kDwnVal=[kDwnVal(1); kDwnVal];          % Same Val as old first Dwn
end

% End extrapolation -----------------------
if ( (kTopTim1(end)== quntL) && (kDwnTim1(end)== quntL) )
    disp ('rPMinExtrapol_s: Poliextrema at signal''s end');
elseif ( (kTopTim1(end)> quntL) || (kDwnTim1(end)> quntL) )
    disp ('rPMinExtrapol_s: Invalid extrema at signal''s end');
else
    kDwnTim1=[kDwnTim1; (2*quntL - kTopTim1(end))];     % New last Dwn at the specular Max
    kDwnVal=[ kDwnVal; kDwnVal(end)];          % Same Val as old last Dwn
end

% return value ------------------------
rPMinExtrapol= sortrows([kDwnTim1, kDwnVal]);

end
%-------------------------------------------------------------------------
%---------- make at 17-Jul-07 10:16:59.44 
function rPMax= rGetPMaxs_s(aS)         %Get Parabolic Maxs, plateaus out
%                                       build 20070612001
kS= aS(:);
quntLenS=length(kS); 
quntMaxCnt=0;
kSMNdx1= []; kSMVal=[];         %signal S Maxima indices and values
kSPMTim1= []; kSPMVal=[];       %signal S Parabolic Maxima times and values

if (quntLenS>2)     %if signal has enough length
    for Cnt=2:(quntLenS-1)  %search the Maxs
        if ( ((kS(Cnt) > kS(Cnt+1))) && ((kS(Cnt) >= kS(Cnt-1))) || ((kS(Cnt) >= kS(Cnt+1))) && ((kS(Cnt) > kS(Cnt-1))) )
            quntMaxCnt=quntMaxCnt+1;
            kSMNdx1= [kSMNdx1; Cnt];  kSMVal=[kSMVal; kS(Cnt)];
        end
    end
end

% Now we have the Maxs, lets get the Parabolic Maxs
oldxv= -Inf; oldyv= -Inf;
intGapMax= max(kS)-min(kS);
for jj=1:quntMaxCnt     %for all Maxs
    %xa= -1; xb= 0; xc= 1;
    ya= kS(kSMNdx1(jj)-1);  % Sample point before
    yb= kS(kSMNdx1(jj));    % Sample point, == kSMVal(jj)
    yc= kS(kSMNdx1(jj)+1);  % Sample point after
    D= (-4*yb+2*ya+2*yc);
    if (D==0), xv= kSMNdx1(jj);
    else xv= kSMNdx1(jj)+(ya-yc)/D; end; % Vertix abscissa
    D= (-16*yb+ 8*ya+ 8*yc);
    if (D==0), yv= yb;
    else yv= yb+ (2*yc*ya- ya*ya- yc*yc)/D; end;
    % Lets check for double maxima
    if ( (xv==oldxv)||(abs(yv-oldyv)/abs(xv-oldxv))> (2*intGapMax) )       
        xv= (xv+ oldxv)/2; yv= max(yv,oldyv);   %Double found
        kSPMTim1(length(kSPMTim1))= xv; kSPMVal(length(kSPMVal))= yv;
    else
        kSPMTim1= [kSPMTim1; xv];  kSPMVal=[kSPMVal; yv];
    end 
    oldxv= xv; oldyv= yv;
end % for jj=1:quntMaxCnt

if quntMaxCnt>0
    if ( kS(1) >= kSPMVal(1) )
        kSPMTim1= [1; kSPMTim1];  kSPMVal=[kS(1); kSPMVal ];    %Start must be included as a Max
    end
    if ( kS(end) >= kSPMVal(end))
        kSPMTim1= [kSPMTim1; quntLenS];  kSPMVal=[kSPMVal; kS(end)];   %End must be included as a Max
    end
end

if quntMaxCnt==0
    if ( kS(1) > kS(2) )
        kSPMTim1= [1; kSPMTim1];  kSPMVal=[kS(1); kSPMVal ];    %Start must be included as a Max
    end
    if ( kS(end) > kS(end-1))
        kSPMTim1= [kSPMTim1; quntLenS];  kSPMVal=[kSPMVal; kS(end)];   %End must be included as a Max
    end
end
if quntMaxCnt<0
    error('rGetPMaxs_s: Invalid MaxCnt value');
end


rPMax= sortrows([kSPMTim1, kSPMVal]);
end
%---------- make at 17-Jul-07 10:16:59.44 
function rPMin= rGetPMins_s(aS)         %Get Parabolic Mins, plateaus out
%                                       build 20070612001
kS= aS(:);
quntLenS=length(kS); 
quntMinCnt=0;
kSMNdx1= []; kSMVal=[];         %signal S Minima indices and values
kSPMTim1= []; kSPMVal=[];       %signal S Parabolic Minima times and values

if (quntLenS>2)     %if signal has enough length
    for Cnt=2:(quntLenS-1)  %search the Mins
        if ( ((kS(Cnt) < kS(Cnt+1))) && ((kS(Cnt) <= kS(Cnt-1))) || ((kS(Cnt) <= kS(Cnt+1))) && ((kS(Cnt) < kS(Cnt-1))) )
            quntMinCnt=quntMinCnt+1;
            kSMNdx1= [kSMNdx1; Cnt];  kSMVal=[kSMVal; kS(Cnt)];
        end
    end
end

% Now we have the Mins, lets get the Parabolic Mins
oldxv= -Inf; oldyv= -Inf;
intGapMax= max(kS)-min(kS);
for jj=1:quntMinCnt     %for all Mins
    %xa= -1; xb= 0; xc= 1;
    ya= kS(kSMNdx1(jj)-1);  % Sample point before
    yb= kS(kSMNdx1(jj));    % Sample point, == kSMVal(jj)
    yc= kS(kSMNdx1(jj)+1);  % Sample point after
    D= (-4*yb+2*ya+2*yc);
    if (D==0), xv= kSMNdx1(jj);
    else xv= kSMNdx1(jj)+(ya-yc)/D; end; % Vertix abscissa
    D= (-16*yb+ 8*ya+ 8*yc);
    if (D==0), yv= yb;
    else yv= yb+ (2*yc*ya- ya*ya- yc*yc)/D; end;
    % Lets check for double minima
    if ( (xv==oldxv)||(abs(yv-oldyv)/abs(xv-oldxv))> (2*intGapMax) )     
        xv= (xv+ oldxv)/2; yv= min(yv,oldyv);   %Double found
        kSPMTim1(length(kSPMTim1))= xv; kSPMVal(length(kSPMVal))= yv;
    else
        kSPMTim1= [kSPMTim1; xv];  kSPMVal=[kSPMVal; yv];
    end 
    oldxv= xv; oldyv= yv;
end % for jj=1:quntMinCnt

if quntMinCnt>0
    if ( kS(1) <= kSPMVal(1) )
        kSPMTim1= [1; kSPMTim1];  kSPMVal=[kS(1); kSPMVal ];    %Start must be included as a Min
    end
    if ( kS(end) <= kSPMVal(end))
        kSPMTim1= [kSPMTim1; quntLenS];  kSPMVal=[kSPMVal; kS(end)];   %End must be included as a Min
    end
end

if quntMinCnt==0
    if ( kS(1) < kS(2) )
        kSPMTim1= [1; kSPMTim1];  kSPMVal=[kS(1); kSPMVal];    %Start must be included as a Min
    end
    if ( kS(end) < kS(end-1))
        kSPMTim1= [kSPMTim1; quntLenS];  kSPMVal=[kSPMVal; kS(end)];   %End must be included as a Min
    end
end
if quntMinCnt<0
    error('rGetPMins_s: Invalid MinCnt value');
end


rPMin= sortrows([kSPMTim1, kSPMVal]);
end
%---------- make at 17-Jul-07 10:16:59.44 
