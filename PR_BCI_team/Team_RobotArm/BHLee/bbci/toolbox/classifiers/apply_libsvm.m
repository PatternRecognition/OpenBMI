function [y_Out]= apply_libsvm(model, dat, varargin)
% APPLY_LIBSVM - wrapper for applying an SVM model (classification or regression) via libsvm.
%
%Synopsis:
%   C = apply_libsvm(MODEL, DAT, <OPT>)
%
%Arguments:
%   MODEL: [STRUCT] - SVM model, that has been trained via train_libsvm (a classifier, regressor or distribution estimator).
%   DAT: DOUBLE [NxM] - Data matrix, with N feature dimensions, and M test points/examples.
%   OPT: PROPLIST - Structure or property/value list of optional properties:
%     'b' - BOOL (default 0): probability_estimates - whether to train a SVC or SVR model for probability estimates
%     'q' - BOOL: if provided, quiet mode (no outputs).  Omit q completely to activate outputs (opt.b==0 is not working).
%
%Returns:
%   Y_OUT: INT/DOUBLE [CxM] - Estimated class labels / class label probabilities /
%                             regression values for the example points contained in DAT.
%                             C is number of classes and M the number of test points/examples.
%                  The data type depends on the kind of SVM, number of classes and values of opt.b in training and application:
%                      - epsilon-SVR and nu-SVR: estimated regression values (numeric, DOUBLE)
%                      - 2-class C-SVC, 2-class nu-SVC, one-class SVM:
%                        estimated signed distance to decision hyperplane
%                        (DOUBLE), with negative values for the first, and
%                        positive for the second class.
%                        If OPT.B==1, the class probabilities are given (DOUBLE).
%                      - n-class SVC (nu-type or C-type): estimated class labels (INT) in BBCI label format:
%                        Y_OUT(i,j)==1 if the point j belongs to class i.
%                        With option OPT.B==1 the class probabilities are
%                        returned instead (DOUBLE)
%
%Description:
%   APPLY_LIBSVM is a wrapper function for the BBCI toolbox to apply
%   a trained SVM classifier (one-, two-, or multiclass) or regression model
%   via the libsvm matlab interface to new data.
%
%   Multi-class SVM is realized by 1-vs-rest. If decision values are delivered (opt.b==0),
%   the interface of libsvm.
%   returns numeric values also for multi-class cases, but their interpretation is
%   unclear. For opt.b==1, the probability values are OK, but need
%   re-ordering according to model.Label
%
%   Prior to use with the bbci toolbox, the libsvm package needs to be
%   downloaded and the matlab interface needs to be compiled (mex).
%
%   More details about this model can be found in the LIBSVM FAQ:
%   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html)
%   and the LIBSVM implementation document:
%   (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).

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
% Examples:
%   (see TRAIN_LIBSVM)
%
%
% See also TRAIN_LIBSVM

% ToDo:
% Check for conversion of (non-probabilistic) output of multi-class
%

% Michael Tangermann 2012_06_12

opt= propertylist2struct(varargin{:});

[opt, isdefault]= ...
    set_defaults(opt, ...
    'b', 0);

% Remove inf values from test data
%ind = find(sum(abs(dat),1)~=inf);
%dat = dat(:,ind);

%% Conversion of options into a string
% more default options are set by the libsvm matlab interface. No need to do it here via set_defaults
libsvm_opts='';
for Fld= fieldnames(opt)',
    fld = Fld{1};
    libsvm_opts = [libsvm_opts ' -' num2str(fld) ' ' num2str(opt.(fld))] ;
end

libsvm_labels = zeros(1,size(dat',1))';


%% Actual application of the model to data
[y_PredClassLabels, acc, y_PredVals] = svmpredict(libsvm_labels, dat', model, libsvm_opts);
% The function 'svmpredict' has three outputs. The first one,
% y_PredClassLabels, is a vector of predicted labels. The second output,
% acc, is a vector including accuracy (for classification), mean
% squared error, and squared correlation coefficient (for regression).
% The third (y_PredVals) is a matrix containing decision values or probability
% estimates (if '-b 1' is specified). If k is the number of classes
% in training data, for decision values, each row includes results of
% predicting k(k-1)/2 binary-class SVMs. For classification, k = 1 is a
% special case. Decision value +1 is returned for each testing instance,
% instead of an empty vector. For probabilities, each row contains k values
% indicating the probability that the testing instance is in each class.
% 
% For b==0 and b==1: Note that the order of classes here is the same as 'Label' field
% in the model structure.
%
% (MT:) Strange enough, libsvm allows for providing labels for the data, such
% that an accuracy can be calculated. I ignored it, as
% e.g. the BBCI xvalidation routine can also take care of this.





%% Some output formatting
y_Out=[];

% y_ClassOut=[];


% Regression case
if (model.Parameters(1)==3 || model.Parameters(1)==4)
    y_Out = y_PredVals';
    return;
end


% One class SVM, same output as for classification with two classes
if (model.Parameters(1)==2)
    y_Out = y_PredVals';
    return;
end

% Classification case
if (model.Parameters(1)<2)
    
    if model.nr_class<3
        % one or two-classes only
        % return one-dimensional vector only, which contains signed distances to the hyperplane
        % Attention: check for correct sign! For y_PredVals, it depends on the class
        % order in model.Labels (as for opt.b==0). For y_PredClassLabels,
        % it does not!
        
        if model.Label(1)==1
            % flip sign for continuous outputs
            y_Out=-y_PredVals';
        else
            y_Out=y_PredVals';
        end
    else
        %multi-class classification:
        % class-wise probabilities if opt.b==1, otherwise binary label
        % format.
        % Distances to decision hyperplanes could be provided in case of
        % opt.b==0, but their interpretation remained unclear to me
        % (re-ordering was of no use). Have to check this with the libsvm
        % FAQ or authors...
        if opt.b==0
            % Class labels requested
            y_Out = zeros(model.nr_class,length(y_PredClassLabels));
            for i=1:model.nr_class
                y_Out(i,y_PredClassLabels==i) = 1; % return binary label matrix (instead of distances to any of the hyperplanes)
            end
        else
            % Probabilities requested:
            % As the order of classes is quite strange in y_PredVals,
            % re-ordering has to be done according to model.Labels:
            y_Out(model.Label',:) = y_PredVals';
        end
    end
end

end

