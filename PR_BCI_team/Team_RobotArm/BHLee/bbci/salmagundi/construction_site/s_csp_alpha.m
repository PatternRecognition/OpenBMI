file = {'Guido_04_03_18/imag_moveGuido', 'Guido_04_03_18/imag_lettGuido'};
label = 'Guido_04_03_18_imag_move_imag_lett_RF'

opt.clab = {'not','E*','Fp*','FAF*','I*','AF*'};
opt.band = [7 30];
opt.ival = [500 3500];
opt.filtOrder= 5;

%% Preprocess data
[cnt, mrk] = loadProcessedEEG(file);
mrk = mrk_selectClasses(mrk, {'right', 'foot'});
cnt= proc_selectChannels(cnt, opt.clab);
[b,a]=butter(opt.filtOrder, opt.band/cnt.fs*2);
cnt_flt = proc_filt(cnt, b, a);
fv= makeEpochs(cnt_flt, mrk, opt.ival);

%% CSP
epo=proc_selectEpochs(fv,1:ceil(size(fv.x,3)/2));
[T,d,n]=size(epo.x);
epocv=proc_covariance(epo);
[fea, W, la]=proc_csp(epo);
fea=proc_logarithm(proc_variance(fea));
[la ix]=sort(la);
W=W(:,ix);
fea.x=fea.x(1,ix,:);


%% Classwise Average covariance matrices
I1=find(epo.y(1,:)>0);
I2=find(epo.y(2,:)>0);
V1=reshape(mean(epocv.x(1,:,I1),3), [d,d]);
V2=reshape(mean(epocv.x(1,:,I2),3), [d,d]);

%% Normalize the filter vectors to norm=1
Wn=W*diag(sqrt(diag(W'*W)).^(-1));
L1=diag(Wn'*V1*Wn);
L2=diag(Wn'*V2*Wn);

%% Response to the first class vs. that to the second class
nc=ceil(d/2);
figure, plot(L1(1:nc), L2(1:nc), 'x', L1(end-nc:end), L2(end-nc:end), 'o', 'linewidth', 2);
set(gca,'xscale','log','yscale','log');
hold on; plot(L1(1:3), L2(1:3), 'om', 'markersize', 10, 'linewidth', 2);
hold on; plot(L1(end-2:end), L2(end-2:end), 'om', 'markersize', 10, 'linewidth', 2);
hold on; plot([.1 10], [.1 10], 'm--', 'linewidth', 2);
grid on;
xlabel(epo.className{1});
ylabel(epo.className{2});
legend({'filters for the right class', 'filters for the foot class'})

%% Plot Trials by CSP features
ic=[1:3, d:-1:d-2];
figure, plot(1:n, squeeze(fea.x(1,ic,:)));
legend([repmat('csp',[6,1]), ('123456')']);
xlabel('trials');

%% Filters and Patterns
mnt=setElectrodeMontage(epo.clab);
A=inv(W);
figure, plotCSPanalysis(epo, mnt, W, A, la, 'nComps', 3);

