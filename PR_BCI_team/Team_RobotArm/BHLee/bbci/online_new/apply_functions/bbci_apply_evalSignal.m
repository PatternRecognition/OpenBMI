function signal= bbci_apply_evalSignal(source, signal, bbci_signal)
%BBCI_APPLY_EVALSIGNAL - Process continuous acquired data
%
%Synopsis:
%  SIGNAL= bbci_apply_evalSignal(SOURCE, SIGNAL, BBCI_SIGNAL)
%
%Arguments:
%  SOURCE - Structure of one block of acquired data
%  SIGNAL - Structure buffering the continuous signal,
%      subfield of 'data' structure of bbci_apply
%  BBCI_SIGNAL - Structure specifying the processing of continuous signals,
%      subfield of 'bbci' structure of bbci_apply. 
%      BBCI_SIGNAL has the following fields
%      .proc - Cell Array of Strings which specify the processing function
%      .proc_param - Cell Array of Cell Arrays specifying the parameters
%            to the functions in field proc.
%      (Furthermore there is a field .source which is used by bbci_apply
%      to select which source is passed to this function.)
%
%Output:
%  SIGNAL - Updated structure buffering the continuous signals.

%Data in SIGNAL is stored in a ring buffer (SIGNAL.x).
%The pointer (SIGNAL.ptr) points to the last stored
%sample (in the time dimension, i.e., first dimension of the buffer).

% 02-2011 Benjamin Blankertz


if isempty(source.x),
  % if bbci.source.min_blocklength==0, then source may be empty
  return;
end

% Get the newly acquired block of data from SOURCE (thereby selecting the
% specified channels) and then sequentially apply processing functions as
% given in bbci_signal.fcn. Some functions require saving of a state.
cnt= signal.cnt;
cnt.x= source.x(:,signal.chidx);

for k= 1:length(bbci_signal.fcn),
  fcn= bbci_signal.fcn{k};
  param= bbci_signal.param{k};
  if signal.use_state(k),
    [cnt, signal.state{k}]= fcn(cnt, signal.state{k}, param{:});
  else
    cnt= fcn(cnt, param{:});
  end
end

if isempty(signal.x),
  % The buffer is initialized here (and not in bbci_apply_initData), since
  % the number of channels (+clab, fs) are not known beforehand, since they
  % depend of the processing steps in bbci_signal.fcn.
  signal.x= zeros(signal.size, size(cnt.x,2));
  signal.fs= cnt.fs;
  signal.clab= cnt.clab;
end

% Store the processed signals into the ring buffer and update the pointer.
T= size(cnt.x, 1);
idx= 1 + mod(signal.ptr + [0:T-1], signal.size);
signal.x(idx,:)= cnt.x;
%Due to the copy-on-write policy, buffer is copied in the line above.
%This slows down performance if the buffer size is large. In that case
%the following work-around could be used:
%bx= inplacearray(signal.x);
%bx(idx,:)= cnt.x;
%releaseinplace(bx);
signal.ptr= idx(end);
signal.time= source.time;
