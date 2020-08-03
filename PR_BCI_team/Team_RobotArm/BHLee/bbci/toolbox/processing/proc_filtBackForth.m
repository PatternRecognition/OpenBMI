function dat= proc_filtBackForth(dat, passBand)
%dat= proc_filtBackForth(dat, passBand)
%
% IN   dat      - data structure of continuous or epoched data
%      passBand - pass band [lowHz highHz], or
%                 name of an EEG band as accepted by getFilterEEGband
%
% OUT  dat      - updated data structure
%
% SEE  getFilterEEGband, proc_filtForth

% bb, ida.first.fhg.de


if ischar(passBand),
  if strcmp(passBand, 'raw'),
    return;
  else
    [b, a]= getFilterEEGband(passBand, dat.fs);
  end
end

dat.x(:,:)= filtfilt(b, a, dat.x(:,:));
