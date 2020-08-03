function C= train_linearPerceptron(xTr, yTr)
%C= train_linearPerceptron(xTr, yTr)
%
% least squares regression to the labels with constant bias input,
% i.e. linear perceptron solution
% Works with the moore-penrose-inverse

if isreal(xTr),
  C.bias= 1;
else
  C.bias= [1; i];
end

C.w= (yTr*pinv([xTr; C.bias*ones(1,size(xTr,2))]))';
