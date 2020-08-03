function r= train_Rminimax(data, labels, la, ga, opt)
%TRAINMINIMAX implement MiniMax
% The algorithm initialisizes the MiniMax Probaility machine
% Input: data: a set of observed data
%        labels: a logical array
%        la, ga: regularisation constant (see Regularised Discriminant Analysis, train_RDA), should be between 0 and 1 both
%        opt: see train_minimax
% Output: A trained Minimax for a two-point-classfier w'z=-b
%         r is a structure:
%           r.w : w from w'z=-b
%           r.b : b from w'z=-b
%           r.alpha = expected likelihood for right classfication of future datas
%
% Guido Dornhege
% 09.01.02

%check the input
if size(labels,1) == 2 labels = [-1 1]*labels;end
if ~exist('la') | isempty(la) la = 0;end
if ~exist('ga') | isempty(ga) ga = 0;end


% some values
[n,m] = size(data);
traindata1 = data(:,find(labels==-1));
traindata2 = data(:,find(labels==1));
EW1 = mean(traindata1,2);
EW2 = mean(traindata2,2);
COV1 = cov(traindata1');
COV2 = cov(traindata2');
COV = cov(data');
COV1 = (1-la)*COV1 + la*COV;
COV1 = (1-ga)*COV1 + ga/n*trace(COV1)*eye(n);
COV2 = (1-la)*COV2 + la*COV;
COV2 = (1-ga)*COV2 + ga/n*trace(COV2)*eye(n);

% permitted mistake
if ~exist('opt','var') | ~isstruct(opt)
    opt.fault= n*eps;
    opt.maxnumber=1E+5;
end
if ~isfield(opt,'fault')
    opt.fault = n*eps;
end
if ~isfield(opt,'maxnumber')
    opt.maxnumber = 1E+4;
end
fault = opt.fault;
maxnumber = opt.maxnumber;

% Calculate a0
a0 = (EW2-EW1);
a0 = a0/(a0'*a0);

F= null(a0');

% initial step in iterativing calculating of u
uold = zeros(n-1,1);
delta=sqrt(a0'*COV1*a0);
epsilon=sqrt(a0'*COV2*a0);
sol = -F'*COV1*a0/delta -F'*COV2*a0/epsilon;
mat = (F'*COV1*F)/delta+(F'*COV2*F)/epsilon;
uneu = mat\sol;
number=0;
% the iteration
while norm(uneu-uold,2)/norm(uneu,2)>fault & number <maxnumber
    number=number+1;
    uold=uneu;
    delta=sqrt((a0+F*uold)'*COV1*(a0+F*uold));
    epsilon=sqrt((a0+F*uold)'*COV2*(a0+F*uold));
    sol = -F'*COV1*a0/delta - F'*COV2*a0/epsilon;
    mat = (F'*COV1*F)/delta+(F'*COV2*F)/epsilon;
    uneu = mat\sol;
end

if number == maxnumber
    warning('maximal number of iteration exceeded');
end

% Calculate the interesting values
a = a0+F*uneu;
kappa = 1/(sqrt(a'*COV1*a)+sqrt(a'*COV2*a));
b = -a'*EW1-kappa*sqrt(a'*COV1*a);
%c = a'*EW2+kappa*sqrt(a'*COV2*a);   %controlling value, ths value must be equal to b
alpha= (kappa^2)/(1+kappa^2);
r.w=a; r.b=b; r.alpha = alpha;



