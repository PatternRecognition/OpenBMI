function [mrk,trg] = getResponseMarker(resp,stim,time);
%GETRESPONSEMARKER GETS A DEFINED RESPONSE MARKER AFTER A STIMULUS
%
% usage:
%  mrk = getResponseMarker(resp,stim,<pos=1>);
%
% input:
%  resp      a marker structure with all response markers
%  stim      a marker structure with all stimulus marker
%  time      maximum time to response (note: a response must be within the next stimulus)
%
% output:
%  mrk       a marker structure as subset of resp where for each stimulus the pos'th response marker afterwards is used.
%  trg       the trigger structure with triggers removed without following marker
%
% Guido DOrnhege, 02/09/2004

if resp.fs~=stim.fs,
  error('stimulus and response markers must have the same sampling rate');
end

if ~exist('time','var') | isempty(time)
  time = inf;
end

time = resp.fs*time/1000;

resp = mrk_sortChronologically(resp);
stim = mrk_sortChronologically(stim);

ind = [];
ind2 = [];
for i = 1:length(stim.pos)
  aa = find(resp.pos>stim.pos(i));
  if length(aa)>=1 & (i==length(stim.pos)  | resp.pos(aa(1))<stim.pos(i+1)) & (resp.pos(aa(1))-stim.pos(i)<=time),
    ind = [ind,aa(1)];
    ind2 = [ind2,i];
  end
end

mrk = mrk_selectEvents(resp,ind);
if nargout>1
  trg = mrk_selectEvents(stim,ind2);
end
