%This folder contains all classification and regression algorithms.
%For each classifier there must be a training function with prefix
%'train_' and an apply function with prefix 'apply_'. If the classifier
%adheres to a specific separating hyperplane formulation, the apply
%function can be omitted and 'apply_separatingHyperplane' will be
%used.
%
% LDA: linear discriminant analysis 
%      Train a multiclass linear classifier (see J.H. Friedman,
%      Regularized Discriminant Analysis, J Am Stat Assoc 84(405),
%      1989)
%      Optimal for gaussian distribution with same covariance
%      matrices and equal priors
% 
% RLDA: Regularised linear discriminant analysis
%      Train a multiclass linear classifier (see J.H. Friedman,
%      Regularized Discriminant Analysis, J Am Stat Assoc 84(405),
%      1989) in a regularised version
%      Optimal (without regularisation) for gaussian distribution
%      with same covariance matrices and equal priors
%
% RDA: Regularised discriminant analysis
%      Train a multiclass quadratic classifier (see J.H. Friedman,
%      Regularized Discriminant Analysis, J Am Stat Assoc 84(405),
%      1989) in a regularised version
%      Optimal (without regularisation) for gaussian distribution
%      with equal priors
%
% QDA: Quadratic discriminant analysis
%      Train a multiclass quadratic classifier (see J.H. Friedman,
%      Regularized Discriminant Analysis, J Am Stat Assoc 84(405),
%      1989) in a regularised version
%      Optimal (without regularisation) for gaussian distribution
%      with equal priors
%
% FisherDiscriminant: Fisher Discriminant analysis. 
%     Train a multiclass FisherDiscriminant (see Duda, Hart,
%     Pattern Recognition). Notice that b will calculate as optimal
%     frontier between Gaussian distribution in the projected hyper
%     space (one against the rest)
%     Optimal for gaussian distribution with same covariances
%     matrices, good assumption for optimal linear classification
%     for different covariances matrices. 
%
% FDqwqx: regularised FisherDiscriminant, which minimize sum of
%      2-norm w plus 2-norm slack-variables (distance to 1 or -1
%      respectively). Only two-class. Without regularisation (w)
%      it's the same as LSR.
%
% FDlwqx: regularised FisherDiscriminant, which minimize sum of
%      1-norm w plus 2-norm slack-variables (distance to 1 or -1
%      respectively). Only two-class. Without regularisation (w)
%      it's the same as LSR.
%
% FDqwlx: regularised FisherDiscriminant, which minimize sum of
%      2-norm w plus 1-norm slack-variables (distance to 1 or -1
%      respectively). Only two-class. 
%
% FDlwlx: regularised FisherDiscriminant, which minimize sum of
%      1-norm w plus 1-norm slack-variables (distance to 1 or -1
%      respectively). Only two-class.
%
% SWLDA: stepwise linear discriminant analysis
%        train a stepwise LDA classifier (see N.R. Draper, H. Smith, 
%        Applied Regression Analysis, 2nd Edition, John Wiley and Sons, 
%        1981.)
%
% LSR: Pseudoinverse. Also known as linearPerceptron. Only
%      two-class. No calculation of a good threshold. 
%
% RLSR: regularised lest mean squares regression. It is a regularised
%      variant to LSR. Equal to FDqwqx. 
%
% MSR: Mean Square regression. The algorithm is described in DUDA/ 
%      HART. The same linear part as in FisherDiscriminant. The threshold
%      is maybe different. Input: nothing. Similar to FDqwqx.
%
% LPM: This method is very similar to the linear support vector machine 
%      for binary classification. Instead of minimizing the 2-norm of 
%      the linear factor here the 1-norm is minimized. 
%
% minimax: Minimax probability machine (see paper of Jordan et al.): 
%      implements a linear classifier for a binary classification 
%      problem. The means and covariances will estimated for both 
%      sides of the training set. For each probability with this 
%      means and covariances the hyperplane is choosen who divides
%      the classes best (maximal score probability). Under this all 
%      the one with minimal score probability is choosen (saves 
%      against the worst case). The classifier is calculated in an 
%      iteration. Therefore some iteration constant can be choosen.
%
% Rminimax: Minimax probability machine (see paper of Jordan et al.) 
%      in a regularised version: implements a linear classifier for a 
%      binary classification problem. The means and covariances will 
%      estimated for both sides of the training set. In the same way 
%      like RDA the covariance matrices are balanced by two constant, 
%      one for the balance between the two covariance matrices, one 
%      for the the balance between the eigenvalues for each covariance. 
%      Then it is the same like minimax.
%
% MSM1: The classifier is desribed in ROBUST LINEAR PROGRAMMING
%      DISCRIMINATION OF TWO LINEARLY INSEPARABLE SETS by Bennett 
%      and Mangasarian. It is the same as the LPM machine with high
%      regularisation constant, i.e. without minimzing the 1-norm of
%      w. It is therefore also the same as a linear support vector
%      machine without noticing of the minimizing of the 2-norm of the
%      linear factor. 
%
% kNearestNeighbor: Search for each test point the k nearest
%      neighbor in the training set and decides on their label
%      which label is good for it.
%
% SVM: support vector machine using the genefinder of Gunnar and Soeren.
%
%
%There are three simple implementations of linear SVMs based on cplex.
%They are not useful for bigger data sets. Please use SVM based on
% Soeren's and Gunnar's gene finder.
%
% linSVM: this is the well-known linear support vector machine. It minimizes 
%      the length of the linear vector regarding to minimal slack variables 
%      (distance to 1 or -1 respectively). Support vector machines try to 
%      make a lot of slack variables to get 0. Therefore only a few traning 
%      vectors are important (the so called support vectors). In the linear 
%      case this is not so important since you can calculate the interesting 
%      values of the classifier directly. In the multiclass case
%      the algorithm is given by On the Algorithmic Implementation
%      of Multiclass Kernel-based Vector Machines by Crammer and Singer.
%
% linSVM_nu: nu-support vector machine (see Schmola, Schölkopf, 
%      nu-SV regression): this is another kind of support vectors. 
%      Instead of a regularisation constant before the slack variables 
%      you have a variable rho to which your slack variables are the 
%      distance of the training set and one new goal is to maximize 
%      this. A new regularisation constant nu manages how important 
%      this is done. You can prove that nu is an upper bound for the 
%      fraction of positive slack variables and a lower bound for the 
%      fraction of support vectors. 
% 
% linSVM_cost: another kind of a linear support vector machine for 
%      binary classification where you can punish fault for one side 
%      harder than for other side. This will be done with a cost factor 
%      given to the slack variables. Note that you need only the ration 
%      between the costs for both sides. Therefore the cost for side 
%      two is fixed to 1. 
%
%




