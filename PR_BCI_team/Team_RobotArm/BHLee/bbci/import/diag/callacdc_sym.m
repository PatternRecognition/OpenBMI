%callacdc_sym (call acdc_sym)- a small demo 
%showing the way to call acdc_sym (the 
%symmetric version). 

%generate the target matrices.
%for running with old data set newdata=0.
newdata=1;
if newdata
    N=3;
    K=5;
    TrueA=randn(N,N)+1j*randn(N,N);;
    TrueL=randn(N,K)+1j*randn(N,K);
    M=zeros(N,N,K);
    for k=1:K
        M(:,:,k)=TrueA*diag(TrueL(:,k))*transpose(TrueA);
        Noise=.01*(randn(N,N)+1j*randn(N,N));
        %Noise has to be symmetric.
        Noise=Noise+transpose(Noise);
        M(:,:,k)=M(:,:,k)+Noise;
    end
end

%call acdc_sym without initialization
A1=acdc_sym(M);
%show the resulting "demixing":
%(note that the phase ambiguity is inherent
%and cannot be resolved without further
%information on the source/mixing)
disp('demixing attained w/o initialization:')
disp(A1\TrueA)
