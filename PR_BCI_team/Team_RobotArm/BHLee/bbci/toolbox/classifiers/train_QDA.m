function C= train_QDA(xTr, yTr)
%C= train_QDA(xTr, yTr)
%
% quadratic discriminant analysis, see train_RDA with parameters lambda and gamma equal to zero.
%
% see J.H. Friedman, Regularized Discriminant Analysis,
%                    J Am Stat Assoc 84(405), 1989

C = train_RDA(xTr, yTr, 0, 0);
