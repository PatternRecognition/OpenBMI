function [dat,state]= online_filtForth(dat, state, passBand,typ,varargin)
%dat= online_filtForth(dat, passBand, <type='b'>)
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

if isempty(state)
  if ischar(passBand),
    if strcmp(passBand, 'raw'),
      return;
    else
      [state.b, state.a]= getFilterEEGband(passBand, dat.fs);
    end
  else
    if strcmp(typ,'e')
      [state.b,state.a] = getEllipFilter(passBand, dat.fs,varargin{:});  
    else
      [state.b,state.a] = getButterFilter(passBand, dat.fs,varargin{:});
    end  
  end
  [dat.x(:,:),state.his] = filter(state.b,state.a,dat.x(:,:));
else
  [dat.x(:,:),state.his] = filter(state.b,state.a,dat.x(:,:),state.his);
end













