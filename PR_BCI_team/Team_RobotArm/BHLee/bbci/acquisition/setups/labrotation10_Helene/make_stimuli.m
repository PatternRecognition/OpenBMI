
clear all
clc
close all

%addpath ~lena/svn_int/investigation/personal/treder/matlab/fourier
%addpath ~lena/svn_int/investigation/personal/treder/matlab/functions
addpath('D:\svn\bbci\investigation\personal\treder\matlab\fourier')
addpath('D:\svn\bbci\investigation\personal\treder\matlab\functions')
addpath('D:\svn\bbci\investigation\personal\treder\matlab\web')

width = 1000;
%path = '/home/lena/Desktop/Symmetrie/stimuli/';
path = 'D:\stimuli\labrotation10_Helene\';

for NUM = 1:10000
    %% I STIMULI
    
    % 2 Achsen
    
    % q1= randn(width/2,width/2);
    % sym = cat(2, q1, fliplr(q1));
    % sym = cat(1, sym, flipud(sym));
    % sym_nat = spectralSlope(sym,2);
    % figure, imshow(sym_nat,[])
    
    
    % 3 Achsen
    q11 = zeros(width/2,width/2);
    for i = 1:size(q11,1)
        q11(i,1:i) = randn(1,i);
    end
    q12 = q11';
    
    for i = 1:size(q12,1)
        q(i,i)= 0;
    end
    q1 = q11 + q12;
    sym = cat(2, q1, fliplr(q1));
    sym = cat(1, sym, flipud(sym));
    sym_nat = spectralSlope(sym,2);
%     figure, imshow(sym_nat,[])
    
    
    ran = randn(width,width);
    ran_nat = spectralSlope(ran,2);
%     figure, imshow(ran_nat,[])
    
    % normalization
    sigma = 32/256;
    middle = 0.5;
    img_sym = normalize(sym_nat, sigma, middle);
    img_ran = normalize(ran_nat, sigma, middle);
    
    
    % save
    imwrite(img_sym,[path,'symA',num2str(NUM),'.jpg'])
    imwrite(img_ran,[path,'ranA',num2str(NUM),'.jpg'])
    
    
    
    %% II STIMULI
    
    f = fbp(width,width,15,30);
    
    FF_sym = fftshift(fft2(sym));
    sym2 = real(ifft2(ifftshift(FF_sym.*f)));
    sym2 = sym2 - mean(mean(sym2));
    sym2 = sym2 > 0;
%     figure, imshow(sym2,[])
    
    FF_ran = fftshift(fft2(ran));
    ran2 = real(ifft2(ifftshift(FF_ran.*f)));
    ran2 = ran2 - mean(mean(ran2));
    ran2 = ran2 > 0;
%     figure, imshow(ran2,[])
    
    
    % save
    imwrite(sym2,[path,'symB',num2str(NUM),'.jpg'])
    imwrite(ran2,[path,'ranB',num2str(NUM),'.jpg'])
    
    
end
