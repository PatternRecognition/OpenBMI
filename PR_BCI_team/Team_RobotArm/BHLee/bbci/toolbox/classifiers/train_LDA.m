function C= train_LDA(xTr, yTr)
%C= train_LDA(xTr, yTr)
%
% linear discriminant analysis
%
% see J.H. Friedman, Regularized Discriminant Analysis,
%                    J Am Stat Assoc 84(405), 1989

C = train_RLDA(xTr, yTr, 0);