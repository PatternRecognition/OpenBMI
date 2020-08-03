function C = train_libsvm(x_Tr, y_Tr, varargin)
% TRAIN_LIBSVM - wrapper for training an SVM model (classification or regression) via libsvm.
%
%Synopsis:
%   C = train_libsvm(X, LABELS, <OPT>)
%
%Arguments:
%   X_TR: DOUBLE [NxM] - Data matrix, with N feature dimensions, and M training points/examples.
%   Y_TR: INT [CxM] - Class membership labels of points in X_TR. C by M matrix of training
%                     labels, with C representing the number of classes and M the number of training examples/points.
%                     Y_TR(i,j)==1 if the point j belongs to class i.
%                     BEWARE: currently a membership of a single example to several classes is
%                     NOT supported by libsvm and wil lead to an error.
%   OPT: PROPLIST - Structure or property/value list of optional properties:
%     's' - INT (default 0): svm_type - sets the type of SVM. Options are:
%        0 -- C-SVC
%        1 -- nu-SVC
%        2 -- one-class SVM
%        3 -- epsilon-SVR
%        4 -- nu-SVR
%     't' - INT (default 2): kernel_type - sets type of kernel function. Options are:
%        0 -- linear: u'*v
%        1 -- polynomial: (gamma*u'*v + coef0)^degree
%        2 -- radial basis function: exp(-gamma*|u-v|^2)
%        3 -- sigmoid: tanh(gamma*u'*v + coef0)
%        % 4 -- precomputed kernel - kernel values in training_set_file
%     'd' - INT (default 3): degree - set degree in kernel function
%     'g' - DOUBLE (default 1/num_features): gamma - set gamma in kernel function
%     'r' - DOUBLE (default 0): coef0 - set coef0 in kernel function
%     'c' - DOUBLE (default 1 if s==0): cost - set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
%     'n' - DOUBLE (default 0.5): nu - set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
%     'p' - DOUBLE (default 0.1): epsilon - set the epsilon in loss function of epsilon-SVR
%     'm' - INT (default 100): cachesize - set cache memory size in MB
%     'e' - DOUBLE (default 0.001): epsilon - set tolerance of termination criterion
%     'h' - BOOL (default 1): shrinking - whether to use the shrinking heuristics
%     'b' - BOOL (default 0): probability_estimates - whether to train a
%           SVC or SVR model for probability estimates. If activated, the trained
%           model can deliver both, distances to hyperplanes and probabilities
%           (as the output of apply_libsvm will depend again on parameter opt.b).
%     'wi' - DOUBLE (default 1): penalty weight - set the parameter C of
%           class i to weight*C, for C-SVC. Use larger penalty for the misclassification of members of the smaller
%           class in unbalanced data sets. As matlab field names do not
%           like "-" chars, the class names are re-mapped to 1:n. Thus use
%           w1 to wn for setting penalty weights.
%     'q' - BOOL: if provided, quiet mode (no outputs).  Omit q completely to activate outputs.
%Returns:
%   C: STRUCT - Trained model (classifier or regressor) including the fields:
%     'Parameters' : parameters
%     'nr_class' : number of classes; = 2 for regression/one-class svm
%     'totalSV' : total #SV
%     'rho' : -b of the decision function(s) wx+b
%     'Label' : label of each class; empty for regression/one-class SVM
%     'ProbA' : pairwise probability information; empty if -b 0 or in one-class SVM
%     'ProbB' : pairwise probability information; empty if -b 0 or in one-class SVM
%     'nSV' : number of SVs for each class; empty for regression/one-class SVM
%     'sv_coef' : coefficients for SVs in decision functions
%     'SVs' : support vectors
%     'w' : weights of the primal (only for linear models)
%     'b' : bias of the primal (only for linear models)
%
%       If you do not use the option '-b 1', ProbA and ProbB are empty
%       matrices. 
%
%
%Description:
%   TRAIN_LIBSVM is a wrapper function for the BBCI toolbox to train
%   a SVM classifier (one-, two-, or multiclass) or regression model
%   via the libsvm matlab interface.
%
%   Prior to use with the bbci toolbox, the libsvm package needs to be
%   downloaded and the matlab interface needs to be compiled (mex).
%
%   More details about this model can be found in the LIBSVM FAQ:
%   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html)
%   and the LIBSVM implementation document:
%   (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).
%
% References:
%   This interface was initially written by Jun-Cheng Chen, Kuan-Jen Peng,
%   Chih-Yuan Yang and Chih-Huai Cheng from Department of Computer
%   Science, National Taiwan University. The current version was prepared
%   by Rong-En Fan and Ting-Fan Wu. If you find this tool useful, please
%   cite LIBSVM as follows
%
%   Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
%   vector machines. ACM Transactions on Intelligent Systems and
%   Technology, 2:27:1--27:27, 2011. Software available at
%   http://www.csie.ntu.edu.tw/~cjlin/libsvm
%
%   For any question, please contact Chih-Jen Lin <cjlin@csie.ntu.edu.tw>,
%   or check the FAQ page:
%
%   http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q9:_MATLAB_interface
%
%
% Example:
%   %train_libsvm(X, labels)
% % 
%   x=0:0.15:45;
%   y=sin(x)+0.2*randn(1,length(x))
% 
%   % Permute data points
%   idx= randperm(length(x));
%   x= x(:,idx); y= y(:,idx);
%   figure ; plot(x,y,'*')
% 
%   % Train regression with gaussian Kernel and C-SVM, but default values
%   % for kernel width and regularization
%   C=train_libsvm(x(:,1:50),y(:,1:50), 's',3, 't',2);
% 
%   % Test trained model with new test points
%   [y_Out]= apply_libsvm(C,x(:,151:300) ,'b',0); 
% 
%   % Plot result
%   figure;
%   plot(x(:,151:300),y_Out,'*'); 
%   hold on ; grid on ; 
%   plot(x(:,151:300),y(:,151:300),'r*'); 
%   plot(x(:,1:50),y(:,1:50),'kx')
%   title('SVM regression result - default hyperparameters')
%   legend({'estimate for test points','true test points','training points'});
% 
%   % Now we want to do better: train with optimized hyperparameters
%   % Typically, these have to be determined in extra xv step or within xv
%   C=train_libsvm(x(:,1:50),y(:,1:50), 's',3, 't',2, 'c', 36, 'g', 0.063,'b', 0,'q',1);
%   [y_Out]= apply_libsvm(C,x(:,151:300) ,'b',0); % completely new test points
% 
%   % Plot result
%   figure;
%   plot(x(:,151:300),y_Out,'*'); 
%   hold on ; grid on ; 
%   plot(x(:,151:300),y(:,151:300),'r*'); 
%   plot(x(:,1:50),y(:,1:50),'kx')
%   title('SVM regression result - hyperparameters optimized')
%   legend({'estimate for test points','true test points','training points'});
%
%
%
% See also APPLY_LIBSVM, XVALIDATION, 

% ToDo:
% Check for one-class SVM - conversion of labels necessary?
% Merge checks for classification / regression cases with label conversions
%

% Michael Tangermann 2012_06_07

opt= propertylist2struct(varargin{:});

[opt, isdefault]= ...
    set_defaults(opt, ...
    's', 0, ...
    't', 2, ...
    'd', 3, ...
    'n', 0.5);




%% Sanity checking and conversions

% Dimensionality of labels

switch opt.s
    case 0 % Classification
        if size(y_Tr,1)==1 %Dimensionality of labels too low
            error('train_libsvm: For classification problems, please use bbci format when defining labels (not just one vector)');
        else %Dimensionality OK
            if sum(sum(y_Tr)>1)
                error('train_libsvm: multi-label training data points (points belonging to several classes) are not supported by libsvm');
            end
            % label conversion
            libsvm_labels=([1:size(y_Tr,1)]*y_Tr)'; % m by 1 vector of training labels
        end
        
    case 1 % Classification
        if size(y_Tr,1)==1 %Dimensionality of labels too low
            error('train_libsvm: For classification problems, please use bbci format when defining labels (not just one vector)');
        else %Dimensionality OK
            if sum(sum(y_Tr)>1)
                error('train_libsvm: multi-label training data points (points belonging to several classes) are not supported by libsvm');
            end
            % label conversion
            libsvm_labels=([1:size(y_Tr,1)]*y_Tr)'; % m by 1 vector of training labels
        end
        
    case 2 % One-class
        if size(y_Tr,1)~=1
            warning('train_libsvm: For one-class problems, only the first row of labels is considered');
        end
        % label conversion
        libsvm_labels= y_Tr(1,:)'; % m by 1 vector of training labels
        
    case 3 % Regression
        if size(y_Tr,1)~=1
            warning('train_libsvm: For regression problems, only first row of labels is considered');
        end
        % label conversion
        libsvm_labels= y_Tr(1,:)'; % m by 1 vector of training labels
        
    case 4 % Regression
        if size(y_Tr,1)~=1
            warning('train_libsvm: For regression problems, only first row of labels is considered');
        end
        % label conversion
        libsvm_labels= y_Tr(1,:)'; % m by 1 vector of training labels
    otherwise
        warning('No libsvm options given - you better do not expect reasonable results from training...');
        
end


% Convert data for libsvm
libsvm_data=x_Tr'; % m by n matrix of m training instances with n features

%% Conversion of options into a string
% default options are set by the libsvm matlab interface. No need to do it here via set_defaults
libsvm_opts='';

for Fld= fieldnames(opt)',
    fld = Fld{1};
    libsvm_opts = [libsvm_opts ' -' num2str(fld) ' ' num2str(opt.(fld))] ;
end

% Result of libsvm training depends on label of first data point. This is
% considered and corrected to match the BBCI standard (i.e. first row of
% labels is class -1, second row is class 1) in apply_libsvm. Information
% about the class orders is found in trained model under "model.Label"
% Earlier workaround:
% if opt.s==0
%     [libsvm_labels, sortIndex] = sort(libsvm_labels, 'descend');
%     libsvm_data=libsvm_data(sortIndex,:);
% end

%% Train classifier / regressor
C = svmtrain(libsvm_labels, libsvm_data, libsvm_opts);


%% Post processing: weight and bias (only for linear models)
if opt.t==0 && (opt.s==0 || opt.s==1) && not(isfield(opt,'v'))  % valid only for linear model and for classification
    C.w=[];
    C.b=[];
    C.w = C.SVs' * C.sv_coef;
    C.b = -C.rho;
    if C.Label(1) == -1
        C.w = -C.w;
        C.b = -C.b;
    end
end

end
