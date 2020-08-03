function iv= getIvalIndices(ival, dat, varargin)
%iv= getIvalIndices(ival, fs)
%iv= getIvalIndices(ival, dat)
%
% IN   ival - time interval [start ms, end ms] (or just a point of time)
%             ival can also be a 2xN sized matrix. In that case the
%             concatenated indices of all ival-columns are returned.
%      fs   - sampling interval
%      dat  - data struct containing the fields .fs and .t
%
% OUT  iv   - indices of given interval [samples]

if length(varargin)==1 & ~isstruct(varargin{1}),
  opt= strukt('dim', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'ival_policy', 'maximal', ...
                  'dim', 1);

if prod(size(ival))>length(ival),
  if size(ival,1)~=2,
    error('IVAL is expected to be 2xN sized.');
  end
  iv= [];
  for kk= 1:size(ival,2),
    newiv= getIvalIndices(ival(:,kk), dat);
    iv= [iv, newiv];
  end
  return;
end

if isstruct(dat),
  if isfield(dat, 'dim') & length(dat.dim)>1,
    %% first dimension of dat.x comprises different 'virtual' dimensions
    %% that have been clashed
    dati= struct('fs', dat.fs);
    dati.t= dat.t{opt.dim};
    if isfield(dat, 'xUnit'),
      dati.xUnit= dat.xUnit{opt.dim};
    end
    iv= getIvalIndices(ival, dati);
    return;
  end
  
  if isfield(dat, 'xUnit') & strcmp(dat.xUnit, 'Hz'),
    switch(opt.ival_policy),
     case 'maximal',
      ibeg= max([1 find(dat.t<=ival(1))]);
      iend= min([find(dat.t>=ival(2)) length(dat.t)]);
     case 'minimal',
      ibeg= min([find(dat.t>=ival(1)) length(dat.t)]);
      iend= max([1 find(dat.t<=ival(2))]);
     case 'sloppy',
      dd= median(diff(dat.t));
      ibeg= min([find(dat.t>=ival(1)-0.25*dd) length(dat.t)]);
      iend= max([1 find(dat.t<=ival(2)+0.25*dd)]);
     otherwise
      error('ival_policy not known');
    end
    iv= ibeg:iend;
    return
  end
  
  iv= getIvalIndices(ival, dat.fs);
  if isfield(dat, 't'),
    segStart= dat.t(1);
  else
    segStart= 0;
  end
  iv= 1 + iv - segStart*dat.fs/1000;
  if (iv(1))~=round(iv(1))
    iv = ceil(iv);
   iv(end) = [];
  end
  
  %%added by rklein on may 16th 2008
    id0 = find(iv==0);
    iv(id0) = [];
  
  return;
end

fs= dat;
iv= floor(ival(1)*fs/1000):ceil(ival(end)*fs/1000);
