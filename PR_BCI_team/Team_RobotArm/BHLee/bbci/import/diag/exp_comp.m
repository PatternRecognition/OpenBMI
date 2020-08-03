%select methods for the competion
methods{1}='jade';
methods{2}='pham';
methods{3}='acdc';
methods{4}='fasto';
Ntrials=10;
maxiter=200;

error_curve=zeros(4,Ntrials,maxiter)*NaN;

for trial=1:Ntrials,   
%generate the target matrices.
%for running with old data set newdata=0.
newdata=1;
if newdata
    N=10;
    K=100;
    [TrueA,ss,vv]=svd(randn(N,N));
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
for k=1:4,
 [VV,moni]=simDiag(M,methods{k},maxiter);
 %res{k}=moni;
 error_curve(k,trial,1:moni.iter)=moni.errdiag;
 time_per_itr(k,trial)=  moni.etime/moni.iter;
 isi(k,trial) =  score2(inv(VV),TrueA);
end


 trial


 %display results

 semilogy(squeeze(error_curve(:,trial,1:50))')
 hold on;legend(methods);
 drawnow;

 %display results

 % semilogy(squeeze(error_curve(:,2,1:35))')

end

figure(2),
clf
subplot(211)
bar(time_per_itr)
subplot(212)
bar(isi) 
 save result_ortho_no_noise_10_100 
