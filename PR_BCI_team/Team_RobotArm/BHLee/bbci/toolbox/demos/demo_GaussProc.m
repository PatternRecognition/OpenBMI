% Demonstrate simple 1-D regression problem with a Gaussian process model
%
% Train GP model with two different kernel functions (RBF and Matern) on a
% tiny dataset, plot prediction plus error bars

randn('state', 0);
% Two clusters of points with some space in between
x = [0.1 0.15 0.2 0.25  0.65 0.7 0.75 0.8 0.85 0.9];
ndata = length(x);
t = sin(2*pi*x) + 0.1*randn([1 ndata]);
xtest = linspace(-0.2, 1, 50);
lineOpt = {'LineWidth', 3};

fh1 = figure;
clf;
set(gca, 'FontSize', 14);
hold on
plot(x, t, '*k', 'MarkerSize', 8);
xlabel('Input')
ylabel('Target')
plot(xtest, sin(2*pi*xtest), 'r-', lineOpt{:});

% Standard GP model with RBF kernel
net = train_GaussProc(x, t);
% Also retrieve the predictive variances
[ytest, yvar] = apply_GaussProc(net, xtest);
sig = sqrt(yvar);

plot(xtest, ytest, '-m', lineOpt{:});
plot(xtest, ytest+(2*sig), ':m', 'LineWidth', 2);
plot(xtest, ytest-(2*sig), ':m', 'LineWidth', 2);
axis([-0.2 1 -1.5 1.5]);
title('Prediction of a GP model with RBF kernel')
legend('Training data', 'True function', 'GP prediction', 'Error bars (2std)');
hold off

opt = [];
% Now also output the progress of evidence maximization
opt.optimizer = {'opt_BFGS', 'verbosity', 2, 'checkGradient', 1};
% Use a Matern kernel with fixed degree of smoothness
opt.kernel = {'matern', 'smoothness', 1};
opt.clamped = {'smoothness'};
net = train_GaussProc(x, t, opt);
[ytest, yvar] = apply_GaussProc(net, xtest);
sig = sqrt(yvar);

fh1 = figure;
clf;
set(gca, 'FontSize', 14);
hold on
plot(x, t, '*k', 'MarkerSize', 8);
xlabel('Input')
ylabel('Target')
plot(xtest, sin(2*pi*xtest), 'r-', lineOpt{:});
plot(xtest, ytest, '-m', lineOpt{:});
plot(xtest, ytest+(2*sig), ':m', 'LineWidth', 2);
plot(xtest, ytest-(2*sig), ':m', 'LineWidth', 2);
axis([-0.2 1 -1.5 1.5]);
title('Prediction of a GP model with Matern kernel')
legend('Training data', 'True function', 'GP prediction', 'Error bars (2std)');
hold off

