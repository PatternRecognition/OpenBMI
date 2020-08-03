function [cnt,state]= online_filterbank(cnt, state, filt_b, filt_a)
%[cnt,state]= online_filterbank(cnt, state, filt_b, filt_a)
%
% apply digital (FIR or IIR) forward filter(s)
%
% IN   cnt    - data structure of continuous data
%      filt_b, filt_a   - cell arrays of filter coefficients
%               as obtained by butters
%
% OUT  cnt    - updated data structure
%
% SEE butters, online_filt, proc_filterbank

% bb, ida.first.fhg.de


if isempty(state),
  state.nFilters= length(filt_b);
  state.filt_b= filt_b;
  state.filt_a= filt_a;
  persistent xo;       %% reserve memory only once
  [T, state.nChans]= size(cnt.x);
  xo= zeros([T, state.nChans*state.nFilters]);
  state.filtstate= cell([1 state.nFilters]);
%  state.clab= cell(1, state.nChans*state.nFilters);
%  cc= 1:state.nChans;
%  for ii= 1:state.nFilters,
%    state.clab(cc)= apply_cellwise(cnt.clab, 'strcat', ['_flt' int2str(ii)]);
%    cc= cc + state.nChans;
%  end
end

cc= 1:state.nChans;
for ii= 1:state.nFilters,
  [xo(:,cc), state.filtstate{ii}]= ...
     filter(state.filt_b{ii}, state.filt_a{ii}, cnt.x, state.filtstate{ii}, 1);
  cc= cc + state.nChans;
end
cnt.x= xo;
%cnt.clab= state.clab;
