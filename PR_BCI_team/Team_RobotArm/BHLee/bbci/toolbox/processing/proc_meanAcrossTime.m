function epo= proc_meanAcrossTime(epo, varargin)
% dat= proc_meanAcrossTime(dat, <ival, chans>)
%
% calculate the average of the signals within a specified time interval
%
% IN   dat   - data structure of continuous or epoched data
%      ival  - interval in which the average is to be calculated,
%              default whole time range
%      chans - cell array of channels to be selected, default all
%
% OUT  dat   - updated data structure

% bb, ida.first.fhg.de

if length(varargin)>=1 & ~ischar(varargin{1}),
  opt= struct('ival', varargin{1});
  if length(varargin)>=2 & ~ischar(varargin{2}),
    opt.clab= varargin{2};
  end
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'std', 0, ...
                  'ival', [], ...
                  'clab', [], ...
		  'proc_NaNs', 0);

if ~isempty(opt.clab),
  epo= proc_selectChannels(epo, opt.clab);
end
if ~isempty(opt.ival),
  epo= proc_selectIval(epo, opt.ival);
end

if ~opt.proc_NaNs
	epo.x= mean(epo.x, 1);
	if opt.std,
  		epo.std= std(epo.x, [], 1);
	end

	if isfield(epo, 't'),
  		epo.t= mean(epo.t);
	end

else
	%this option is VERY slow
	%edited by rklein on june 4 2008
	% problem if data contains NaN values in old version
	sz = size(epo.x);
	n = prod(sz(2:end));
	x_new = zeros([1 n]);
	epo.x = reshape(epo.x,[sz(1) n]);
	for i=1:n
    		temp = epo.x(:,i);
    		idx_NaN = find(isnan(temp));
    		temp(idx_NaN) = [];
    		x_new(i) = mean(temp);
	end;
	x_new = reshape(x_new,[1 sz(2:end)]);
	epo.x = x_new;


	if opt.std,
  		epo.std= std(epo.x, [], 1);
	end

	if isfield(epo, 't'),
    	%edited by rklein on june 4 2008
  		if iscell(epo.t)
      			t = mean(epo.t{1});
      			epo.t=cat(2,{t},epo.t{2:end});
 	 	else
      			epo.t= mean(epo.t);
  		end;
	end

	if isfield(epo, 'dim'),
    		epo.dim=[1 epo.dim(2:end)];
	end;
end