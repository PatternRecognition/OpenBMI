%select methods for the competion
%methods{1}='jade';
methods{1}='fast';
methods{2}='pham';
methods{3}='acdc';
%methods{4}='fasto';
Ntrials=10;
maxiter=500;

error_curve=zeros(3,Ntrials,maxiter)*NaN;

for trial=1:Ntrials,   
%generate the target matrices.
%for running with old data set newdata=0.
newdata=1;
if newdata
    N=100;
    K=150;
   % [TrueA,ss,vv]=svd(randn(N,N));
   TrueA= randn(N,N);
   %the 1st matrix is p.d. to allow whitening
    TrueL=[rand(N,1) rand(N,K-1)]; %positive definite data
    M=zeros(N,N,K);
    for k=1:K
        M(:,:,k)=TrueA*diag(TrueL(:,k))*TrueA';
        Noise=0*randn(N,N);
        %Noise has to be Hermitian. Also, make
        %it p.d. to keep the 1st matrix p.d.:
        Noise=Noise*Noise';
        M(:,:,k)=M(:,:,k)+Noise;
    end
end



%run competion
for k=1:3,
 [VV,moni]=simDiag(M,methods{k},maxiter);
 %res{k}=moni;
 error_curve(k,trial,1:moni.iter)=moni.errdiag;
 time_per_itr(k,trial)=  moni.etime/moni.iter;
 isi(k,trial) =  score2(inv(VV),TrueA);
end


 trial

figure(2)
subplot(311)
 %display results

 semilogy(squeeze(error_curve(:,trial,1:200))')
 hold on;legend(methods);
 drawnow;

 %display results

 % semilogy(squeeze(error_curve(:,2,1:35))')

end

figure(2),

subplot(312)
boxplot(time_per_itr')
subplot(313)
boxplot(isi') 

% figure(3)

% subplot(312)
% hh=bar(mean(isi'))
% ch=get(hh,'Children');
% set(ch(1),'LineWidth',1);set(ch(1),'FaceColor','r');

save result_non_ortho_no_noise_100_150 
