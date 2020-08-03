function out = apply_mkl(C,dat);

clear gf
gf('send_command', 'clean_kernels') ;
gf( 'clean_features', 'TRAIN' );

nF = length(C.dimensions);
train_feature = cell(1,nF);
test_feature = cell(1,nF);
start = 0;
for i = 1:nF
  test_feature{i} = dat(start+1:start+prod(C.dimensions{i}),:);
  train_feature{i} = C.SV(start+1:start+prod(C.dimensions{i}),:);
  start = start+prod(C.dimensions{i});
end
fea = C.feature;


cache_size = ceil(16*max(size(C.SV,2),size(dat,2))^2/1024/1024);

for i = 1:length(fea)
  gf('add_features','TRAIN', train_feature{fea(i)});
end


gf('set_labels','TRAIN', C.labels);

gf('send_command', 'new_svm LIGHT');
% $$$ gf('send_command', 'use_linadd 0');
% $$$ gf('send_command', 'use_mkl 1');
% $$$ gf('send_command', 'use_precompute 1');


gf('send_command', sprintf('set_kernel COMBINED %d', cache_size));

for i = 1:length(C.kernel)
  gf('send_command', C.kernel{i});
end


gf('set_svm',C.b,[C.w,C.alpha]);
gf('set_subkernel_weights',C.kw);
gf('send_command', 'init_kernel TRAIN');

gf( 'clean_features', 'TEST' );

for i = 1:length(fea)
  gf('add_features','TEST', test_feature{fea(i)});
end
gf('send_command', 'init_kernel TEST');

out=gf('svm_classify');


gf('send_command', 'clean_kernels') ;
gf( 'clean_features', 'TRAIN' );
gf( 'clean_features', 'TEST' );
