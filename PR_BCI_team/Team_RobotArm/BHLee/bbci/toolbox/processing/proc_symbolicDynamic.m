function   [dat] = proc_symbolicDynamic(dat, order, tau)
%  [SymDyn] = makeSymbolicDynamic(dat, embeddingDimension, embeddingDelay)
%  
% IN:   dat                - can be either a struct of epoched 
%                            or continues data 
%                            or a time series [Time x dimensions]
%       embeddingDimension - size of the state space, equals the number of lagged coordinates 
%       embeddingDelay     - delay between lagged coordinates
%
% OUT: SymDyn  -  same type and size as input 'dat' 
%                 the first order*tau elements are set to NaN, 
%                 as therefore exist no corresponding vector in state space,
%                 if 'dat' was a struct, it will additionally contains 
%                 the following fields
%       .isSymDym - boolean
%       .order    - integer (embeddingDimension)
%       .delay    - integer (embeddingDelay)
%
%
% NOTE : mutltiple channels and or trials will be coded into a symbolic
%        dynamic independently 
%
%
% Example:
%
%  xx = rand(1000,10) ;
%  m = 5; tau = 2 ;
%  [SymDyn] = makeSymbolicDynamic(xx, m, tau) ;
%
%  
%  epo.x = randn(1000,4,100) ;
%  m = 6; tau = 1;
%  [SymDyn] = makeSymbolicDynamic(epo, m, tau) ;
%  
%
% StL, Apr. 2005, Berlin
  
isTimeSeries = ~isstruct(dat) ;

% get the size of the series (Time, Channels, Epochs)
if isTimeSeries,
  [T, nChans, nEvt] = size(dat) ;
  dat.x = dat ;
else
  [T, nChans, nEvt] = size(dat.x) ;
end ;

% loop over each trial
for trial = 1:nEvt,
  % loop over each channel
  for cc = 1:nChans,
    % embed the one-dimensional time series given the channel and given
    % the epoch 
    y = embitTau(squeeze(dat.x(:,cc,trial))', order, tau) ;
    
    % get the size of the embedded series
    [dy,Ty] = size(y) ;

    % sort the coordinates, sorting index is stored in 'permutation'
    [dummy, permutation]=sort(y) ;

    % assign a symbol - bijectively map a permutation to {1, ... ,oder!} 
    [PermutationIndex] = fastPImapping(permutation');
    
    % replace the time series by its symbolic representation
    dat.x((end-Ty+1):end, cc, trial) = PermutationIndex ; 

    % set values of time points without a corresponding state space
    % vector to NaN, 
    dat.x(1:(end-Ty+1), cc, trial) = NaN ; 

  end ;
end ;

if isTimeSeries
  dat = dat.x ;
else      
  % define the additional fields of the struct
  dat.isSymDynamic    = true ; 
  dat.symDynamicOrder = order ;
  dat.symDynamicTau   = tau ;
end ;
  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%SUBFUNCTIONs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%embitTau%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  embeddedSeries = embitTau(serie, order, tau) 
% embedding of a  one-dimensional time series "serie" into 
% "order"-dimensional  space by the method of delays, 
% the delay is defined by "tau"


% delayed coordinates (0, tau, 2*tau, ..., (order-1)*tau)
tau = tau*(0:(order-1)) ;

% first time point that can be embedded (start_t -(order-1)*tau == 0)   
start_t	= max(tau)+1;

embeddedSeries = [];
% for each delayed coordinate  append the appropriately shifted series as  
% an additional row-vector to "embeddedSeries"  
for i = 1 : length(tau)
  embeddedSeries = [embeddedSeries; serie(start_t-tau(i):end-tau(i))];
end




%%%%%%%%%%%%%%%%%%%%%%% fastPImapping %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [PI] = fastPImapping(p)
% p - matrix (N,M) containing N permutation of (1..M)
%     each row is a permutation 
% 
% this function will quite fast, assign bijectively a symbol {1,..M!}
% to each given permutation, using M-adic number representation
% 

  % get the number of permutations and the length of each permutation
  [N,M] = size(p) ;
  
  % values of the M-adic "Bits" 
  m_pow = M.^(0:(M-1)) ;

  % all M-adic numbers of length M
  MtimesMtoMfac = zeros(M^M,1) ;
  
  % get faculty of M  M! = 1*2*...*M 
  mFaculty = prod(1:M) ;
  
  % get all possible permutations of (1,..,M) 
  allPermutations = perms(1:M);

  % get the mapping from a subset of all M-adic numbers to allPermutations
  for idx = 1: mFaculty ,
    % interprete a permutation as M-adic number 
    M_adic = sum((allPermutations(idx,:)-1).*m_pow) ;
    
    % assign to this number the Index (rank) of the permutation in the
    % list of all permutations 
    MtimesMtoPIc(M_adic) = idx ;
  end ;
    
  % now we are ready to map our given permutations

  PI = zeros(N, 1) ;
  % loop over each given permutation
  for idx = 1: N ,
    % express the permutation as M-adic number  
    PI(idx) = sum((p(idx,:)-1).*m_pow) ;
  end ;
  % use the mapping from M-adic to the permutation index
  PI = MtimesMtoPIc(PI) ;

