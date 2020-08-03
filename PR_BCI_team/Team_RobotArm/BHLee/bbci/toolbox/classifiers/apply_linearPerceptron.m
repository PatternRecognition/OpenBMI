function out= apply_linearPerceptron(C, x)
%out= apply_linearPerceptron(C, x)

out= real(C.w' * [x; C.bias*ones(1,size(x,2))]);

if size(out,1)==2,
  out= [-1 1]*out;
end
