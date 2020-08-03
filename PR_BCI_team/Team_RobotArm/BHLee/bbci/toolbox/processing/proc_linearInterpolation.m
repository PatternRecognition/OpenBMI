function dat= proc_linearInterpolation(dat)
%

sz= size(dat.x);
nCE= prod(sz(2:end));
tt= 1:sz(1);

xx= zeros(2, nCE);
for ii= 1:nCE,
  C= train_linearPerceptron(tt, dat.x(:,ii)');
  xx(:,ii)= C.w;
end
dat.x= reshape(xx, [2 sz(2:end)]); ;

if isfield(dat, 't'),
  dat= rmfield(dat, 't');
end
