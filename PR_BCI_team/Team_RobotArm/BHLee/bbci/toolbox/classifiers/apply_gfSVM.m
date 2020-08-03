function out = apply_gfSVM(gfSVM, data);
%apply_gfSVM - applies data to classifier gfSVM trained by train_gfSVM.
%
% Synopsis:
%   out = train_gfSVM(gfSVM, data);
%   
% Arguments:
%   gfSVM:	gfSVM learned using train_gfSVM
%   data:	[n,d]	real valued data containing n examples of d dims
%
% Returns:
%  out:    -  real label for each test example (decision should be
%            done on the sign)
%
% See also: train_gfSVM
%
% Soeren Sonnenburg (2005)
% based on work from die Guido Dornhege (2003)

if isfield(gfSVM, 'properties') & isfield(gfSVM.properties, 'verbosity'),
  switch gfSVM.properties.verbosity,
    case 0
	  gf('send_command','loglevel ERROR');
    case 1
      gf('send_command','loglevel WARN');
    case 2
      gf('send_command','loglevel ALL');
    otherwise
      error('Unknown value for option ''verbosity''');
  end
end

if size(gfSVM.alphas,1) > 0,
	% we need some svm object, it does not matter which one
	gf('send_command', 'new_svm LIGHT');
	gf('set_features','TRAIN', gfSVM.suppvec);
	gf('set_features', 'TEST', data);

	gf('set_svm', gfSVM.b, gfSVM.alphas);

	gf('send_command', sprintf('set_kernel %s', gfSVM.kernel_string));
	gf('send_command', 'init_kernel TRAIN');
	gf('send_command', 'init_kernel TEST');

	out=gf('svm_classify');
else
	out=gfSVM.b*ones(1,size(data,2));
end
