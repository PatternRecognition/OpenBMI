function dat= proc_filtForth(dat, passBand,typ,varargin)
%dat= proc_filtForth(dat, passBand, <type='b'>)
%
% IN   dat      - data structure of continuous or epoched data
%      passBand - pass band [lowHz, highHz], or
%                 name of an EEG band as accepted by getFilterEEGband
%      typ      - 'b' or 'e' for butter or ellip
%
% OUT  dat      - updated data structure
%
% SEE  getFilterEEGband, proc_filtBackForth

% bb, ida.first.fhg.de


if ~exist('typ') | isempty(typ)
  typ = 'b';
end

if ischar(passBand),
  if strcmp(passBand, 'raw'),
    return;
  else
    [b, a]= getFilterEEGband(passBand, dat.fs);
  end
else
  if strcmp(typ,'e')
     [b,a] = getEllipFilter(passBand, dat.fs,varargin{:});  
  else
     [b,a] = getButterFilter(passBand, dat.fs,varargin{:});
  end  
end

dat.x(:,:)= filter(b, a, dat.x(:,:));









