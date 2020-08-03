clear mnt
mnt.clab= cprintf('sA%d', 1:48)';
mnt.clab= cat(2, mnt.clab, cprintf('sB%d', 1:6)');
mnt.clab= cat(2, mnt.clab, cprintf('sC%d', 1:4)');

for no= 1:48,
  mnt.x(no)= -0.2 - 0.8*floor((no-1)/8)/7;
  mnt.y(no)= 0.5 - mod(no-1, 8)/7;
end
mnt.x(49:54)= [0 0 0 -0.025 -0.05 -0.1];
mnt.y(49:54)= [0.8 0.7 0.575 0.45 0.35 0.25];
mnt.x(55:58)= [0.1 0.05 0 -0.05];
mnt.y(55:58)= [0 0.1 0.2 0.3];

grd= sprintf('sB1,sB2,sB3,sB4,sB5,sB6,scale,legend\n_,_,_,_,sC4,sC3,sC2,sC1\n');
for ii= 0:5,
  for jj= 1:8,
    no= jj+8*ii;
    grd= [grd sprintf('sA%d,', no)];
  end
  grd= [grd(1:end-1) sprintf('\n')];
end
grd= [grd(1:end-1)];

opt_lmu= strukt('do_laplace', 0, ...
                'grd',grd, ...
                'clab',{'not','F*','E*','sA23','sA24','sA25','sA26','RS','Res*','sA41','sA48','sB1','sB5'}, ...
                'colDef',{'right II', 'right V'; [1 0 0], [0 0 1]}, ...
                'selband_opt', struct('areas', []), ...
                'selival_opt', struct('areas', []), ...
                'reject_artifacts', 1, ...
                'reject_channels', 1);

if ~isfield(bbci, 'setup_opts'),
  bbci.setup_opts= [];
end
bbci.setup_opts= set_defaults(opt_lmu, bbci.setup_opts);
