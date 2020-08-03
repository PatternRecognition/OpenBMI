function [fv,w,la] = proc_csp_multiclass(epo,n);


if ~exist('n','var') | isempty(n)
  n = 1;
end


Sigma = zeros(size(epo.x,2),size(epo.x,2),size(epo.y,1));

for i = 1:size(epo.y,1)
  ind = find(epo.y(i,:) >0);
  x = permute(epo.x(:,:,ind),[2 1 3]);
  x = x(:,:);
  Sigma(:,:,i) = x*x'/size(x,2);
end

sig = Sigma;

[v,d] = eig(sum(Sigma,3));

v = v';
v = diag(1./sqrt(diag(d)))*v;

for k = 1:size(epo.y,1)
  Sigma(:,:,k) = v*Sigma(:,:,k)*v';
end

opts = optimset('display','off');

str1 = 'eigs(A';
for i = 1:size(epo.y,1)-2;
  str1 = [str1,sprintf('-lamb(%i)*B(:,:,%i)',i,i)];
end
str2 = 'eigs(';
for i = 1:size(epo.y,1)-2;
  str2 = [str2,sprintf('+lamb(%i)*B(:,:,%i)',i,i)];
end
str2 = [str2,'-A'];

str1 = [str1,',1,''''LR'''',struct(''''disp'''',0))'',''lamb'',''A'',''B'');'];
str2 = [str2,',1,''''LR'''',struct(''''disp'''',0))'',''lamb'',''A'',''B'');'];
str1 = ['f1=inline(''', str1];
str2 = ['f2=inline(''', str2];
eval(str1);
eval(str2);

w = [];
la = [];


for i = 1:size(epo.y,1);
  A = Sigma(:,:,i);
  in = setdiff(1:size(epo.y,1),i);
  B = repmat(Sigma(:,:,in(1)),[1 1 length(in)-1]);
  B = B-Sigma(:,:,in(2:end));
  
  warning off backtrace
  warning off verbose

  x1 = fminunc(f1,1/size(epo.y,1),opts,A,B);
  x2 = fminunc(f2,1/size(epo.y,1),opts,A,B);

  warning on backtrace
  warning on verbose
  
  C = sum(repmat(reshape(x1,[1 1 size(epo.y,1)-2]),[size(epo.x,2),size(epo.x,2)]).*B,3);
  
  [v1,l1] = eigs(A-C,n,'LR',struct('disp',0));
  v1 = v1/sqrt(v1'*v1);
   
  v1 = v'*v1;
  
  w = [w,v1];
  la = [la,l1];
  
  C = sum(repmat(reshape(x2,[1 1 size(epo.y,1)-2]),[size(epo.x,2),size(epo.x,2)]).*B,3);
  
  [v1,l1] = eigs(C-A,n,'LR',struct('disp',0));
  v1 = v1/sqrt(v1'*v1);
   
  v1 = v'*v1;
  
  w = [w,v1];
  la = [la,-l1];
  
%  vv = null([v1,v2]');
%  vold = vold*vv;
  
%  sig = zeros(size(sigma)-[2 2 0]);
%  
%  for k = 1:4
%    sig(:,:,k) = vv'*sigma(:,:,k)*vv;
%    sig(:,:,k) = 0.5*(sig(:,:,k)+sig(:,:,k)');
%  end
%  
%  sigma = sig;
%end


end



fv=proc_linearDerivation(epo,w);

% $$$ 
% $$$ 
% $$$ for k = 1:size(epo.y,1)
% $$$   disp(diag(w'*sig(:,:,k)*w));
% $$$ end
