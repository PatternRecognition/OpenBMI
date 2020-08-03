function data= proc_filtconcat(dat, b, a,varargin)
%dat= proc_filt(dat, b, a)
%
% apply digital (FIR or IIR) filter
%
% IN   dat   - data structure of continuous or epoched data
%      b, a     - filter coefficients
%
% OUT  dat      - updated data structure

% bb, ida.first.fhg.de

data = copyStruct(dat,'x');

data.x(:,:)= filter(b, a, dat.x(:,:));
for i = 1:length(varargin)/2
  data.x = cat(2,data.x,filter(varargin{2*i-1},varargin{2*i},dat.x(:,:)));
end

