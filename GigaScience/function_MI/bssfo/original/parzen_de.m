function [ pd ] = parzen_de( X,d )
%PAZEN_DE Summary of this function goes here
%   Detailed explanation goes here
    %Parzen Density Estiamation using Gaussian Kernel
     % FUNCTION
     %   [px] =parzen_de(X,d,h)
     % % INPUT ARGUMENTS:
     %   X:  Training Data for the class
     %   d:  the data we need to estimate the density for
     %   h:  the kernel width
     % OUTPUT ARGUMENTS:
     %   pd: parzen pdf estimate for d
     [Nm l]=size(X);
     pd=0; 
     %!-------------------------------flexible
     yhap = 0;
     for i=1:Nm
        xii=X(i,:);
        yi(i) = d - xii;
        yhap = yhap + yi(i);
     end
     ymean = yhap./Nm;
     ystd = (sum((yi - ymean).^2)./Nm).^(1/2);
     %!------------------------------------------------
     
     for i=1:Nm
        xi=X(i,:);
        y = d - xi;
        stdv = ystd;
%         stdv = 0.25;
        h = (4/(3*Nm)).^(1/5).* stdv;

        pd=pd+exp((-1/2).*(y^2)./(h^2));
     end
     pd=pd.*((1/Nm)*(1/sqrt(2*pi)));

end

