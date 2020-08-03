function out = apply_SVM(SVM,data);
%APPLY_SVM applies data to classifier SVM trained by train_SVM.
%
% usage:
%  out = apply_SVM(SVM,data);
%
% input:
%  SVM    -  a classifier trained by train_SVM
%  data   -  data which should be applied
%
% output:
%  out    -  real label for each test example (decision should be
%            done on the sign)
%
% Guido Dornhege, 16/10/03
% based on programs written by Gunnar and Soeren
% changed by Benjamin to use gfSVM, see train_SVM

out= apply_gfSVM(SVM, data);
return;


%% old code:

if size(gfSVM.alphas,1) > 0,
	gf('set_features','TRAIN',SVM.SV);
	gf('set_labels', 'TRAIN', SVM.SVlab);
	gf('set_features', 'TEST', data);

	gf('set_svm',SVM.b,SVM.alphas);

	gf('send_command', ['set_kernel ' SVM.kernel]);
	gf('send_command', 'init_kernel TEST');

	out=gf('svm_classify');
else
	out=gfSVM.b*ones(1,size(data,2));
end
