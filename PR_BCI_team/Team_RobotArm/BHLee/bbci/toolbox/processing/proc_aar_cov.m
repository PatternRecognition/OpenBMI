function   datAAR=proc_aar_cov(dat,p,N,UC,lag,c,varargin);
%function   dat=proc_aar_graz(Dat,Ord,WLen,UpCo,Lag,Cov <[ARTyp,ARWin]>)
%           
%  Dat : cnt/epo-Structure
%  Ord : order of the AAR-model
% WLen : length of the calculating window
% UpCo : update coefficent for calculating AR-model
%  Lag : lag in time-dimension of output-structure
%  Cov : Initial covariance-Matrices for calculating AAR
%ARTyp : Method for calculating initial ARparameters (default 'first')
%        'first' choose the parameters for first caclulation window
%        'mean' choose mean of ARprameters over all windows
%ARWin : Length of Calculating-Window for AR-Method (default '100')



%% Setting up varargin-parameters

    typ = 1;
    arl = 100;


%keyboard;

%% Setting up variables

nx       =      size(dat.x);
nx(1)    =      ceil(nx(1)/lag);

                        %% Size of input structure
                        
nxa      =      [p,nx];                       
                        %% Size of output sructure
nWin     =     nx(1);
                        %% Number of AAR-Coeff. per epoche/channel

datAAR    =      copyStruct(dat,'x');                        
datAAR.x  =      zeros(nxa);
                        %% Setting up output dat-array

                     
nCE       =     nx(2) ;
if ndims(dat.x)>2
  nCE      =     nCE*size(dat.x,3);
end
                        %% Number of channels*epoches



%% calculating inital AR-parameters                        
a0       =     zeros(p,nCE);
nARwin   =     floor((length(dat.x(:,1,1))-N)/arl);
                        %% Number of possible AR-model windows

%% switch lower(typ);
%%  case('first')
if typ==1
    for k = 1:nCE    
      a        =     aryule(dat.x(1:arl,k),p);
      a00(:,k) =     -a(2:end)';
      
%%      cov00(:,:,k) = cov;
    end

%%  case('mean')
elseif typ==2
    for k = 1:nCE
      for l = 1:nARwin
        ar(:,l) =     aryule(dat.x([1:arl]*l,nCE))
      end 
      a0(:,k)=mean(ar,2);
     end
%%    cov00 = zeros(p,p,nCE);
  else
    error('Stoped! Wrong calculating method for AR');
end



%% calculating AAR-parameters
for k = 1:nCE
%  k
%  datAAR.x(:,:,k) = aar(dat.x(:,k),[1 2],p,UC,a00(:,k)',cov)';
  
  
  
  AARmat          = aar(dat.x(:,k),[1 2],p,UC,a00(:,k)',c{k})';
%  AARmat          = AARmat(:,1:lag:end);
  datAAR.x(:,:,k) = AARmat(:,1:lag:end);
  clear AARmat;
  
end

  
  
  
    
  

