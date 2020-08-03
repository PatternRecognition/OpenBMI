function dat= proc_bipolarEOG(dat)
%dat= proc_bipolarEOG(dat)
%
% special use only

if ~isempty(chanind(dat, 'EOGh')),
  warning('EOGh already bipolar');
else
  EOGh= min(chanind(dat, 'EOGhp','F9'));
  EOGh= cat(2, EOGh, min(chanind(dat, 'EOGhn', 'F10')));
  if length(EOGh)~=2,
    error('monopolar EOGh channel(s) not found');
  end
  dat.x(:,EMGh(1),:)= dat.x(:,EOGh(1),:)-dat.x(:,EOGh(2),:);
  dat.x(:,EOGh(2),:)= [];
end

if ~isempty(chanind(dat, 'EOGv')),
  warning('EOGv already bipolar');
else
  EOGv= min(chanind(dat, 'EOGvp'));
  EOGv= cat(2, EOGv, min(chanind(dat, 'EOGvn', 'Fp2')));
  if length(EOGv)~=2,
    error('monopolar EOGv channel(s) not found');
  end
  dat.x(:,EOGv(1),:)= dat.x(:,EOGv(1),:)-dat.x(:,EOGv(2),:);
  dat.x(:,EOGv(2),:)= [];
end
