function [RMAOV2] = RMAOV2(X,alpha)
% RMAOV2 Repeated Measures Two-way Analysis of Variance Test.
%   ANOVA with two within-subject variables is used to analyze the relationship 
%   between two independent variables and the dependent variable. This procedure
%   requires fewer participants but can give rise to research confounds such as 
%   order or practice effects and participant bias.
%   For a more detailed explanation on repeated measures designs, we suggest you 
%   read the help section of the RMAOV1 file which you can find on this Matlab File
%   Exchange site.
%   
%   Syntax: function [RMAOV2] = RMAOV2(X,alpha) 
%      
%     Inputs:
%          X - data matrix (Size of matrix must be n-by-4;dependent variable=column 1;
%              independent variable 1=column 2;independent variable 2=column 3;
%              subject=column 4). 
%      alpha - significance level (default = 0.05).
%    Outputs:
%            - Complete Analysis of Variance Table.
%            - Strength of the relationships.
%
%    Example: From the example of Schuyler W. Huck other's on-line resources Chapter 16 (within-subjects ANOVA)
%             of the Readings Statistics and Research book [http://www.readingstats.com/third/index.html]. 
%             Considering the following hypothetical experiment. A total of four rats are run in a maze 
%             three trials (T) a day for two days (D). The number of wrong turns on each trial are shown
%             on the below table. Use a significance level = 0.05.
%
%                                                      D1                         D2
%                                          ---------------------------------------------------
%                               Subject       T1       T2       T3       T1       T2       T3
%                             ----------------------------------------------------------------
%                                  1         10        9        7        8        7        5
%                                  2          9        8        5        5        5        3           
%                                  3          9        7        7        7        5        5
%                                  4          8        6        4        3        2        1
%                             ----------------------------------------------------------------
%                                       
%     Data matrix must be:
%     X=[10 1 1 1;9 1 1 2;9 1 1 3;8 1 1 4;9 1 2 1;8 1 2 2;7 1 2 3;6 1 2 4;7 1 3 1;5 1 3 2;7 1 3 3;4 1 3 4;
%     8 2 1 1;5 2 1 2;7 2 1 3;3 2 1 4;7 2 2 1;5 2 2 2;5 2 2 3;2 2 2 4;5 2 3 1;3 2 3 2;5 2 3 3;1 2 3 4];
%
%     Calling on Matlab the function: 
%             RMAOV2(X)
%
%       Answer is:
%
%    The number of IV1 levels are: 2
%
%    The number of IV2 levels are: 3
%
%    The number of subjects are: 4
%
%    Repeated Measures Two-Way Analysis of Variance Table.
%    ---------------------------------------------------------------------------
%    SOV                  SS          df           MS             F        P
%    ---------------------------------------------------------------------------
%    Subjects           43.458         3         14.486[       24.716   0.0000]
%    IV1                45.375         1         45.375        33.000   0.0105
%    Error(IV1)          4.125         3          1.375
%    IV2                30.333         2         15.167        24.818   0.0013
%    Error(IV2)          3.667         6          0.611
%    IV1xIV2             1.000         2          0.500         3.000   0.1250
%    Error(IV1xIV2)      1.000         6          0.167
%    [Error              8.792        15          0.586]
%    Total             128.958        23
%    ---------------------------------------------------------------------------
%    If the P result are smaller than 0.05
%    the corresponding Ho's tested result statistically significant. Otherwise, are not significative.
%    [Generally speaking, no Mean Square is computed for the variable "subjects" since it is assumed
%    that subjects differ from one another thus making a significance test of "subjects" superfluous.
%    However, for all the interested people we are given it anyway].
%  
%    The percentage of the variability in the DV associated with the IV1 is 91.67
%    (After the effects of individual differences have been removed).
%
%    The percentage of the variability in the DV associated with the IV2 is 89.22
%    (After the effects of individual differences have been removed).
%
%    Created by A. Trujillo-Ortiz, R. Hernandez-Walls and R.A. Trujillo-Perez
%               Facultad de Ciencias Marinas
%               Universidad Autonoma de Baja California
%               Apdo. Postal 453
%               Ensenada, Baja California
%               Mexico.
%               atrujo@uabc.mx
%
%    Copyright.July 25, 2004.
%
%    To cite this file, this would be an appropriate format:
%    Trujillo-Ortiz, A., R. Hernandez-Walls and R.A. Trujillo-Perez. (2004). RMAOV2:Two-way repeated
%      measures ANOVA. A MATLAB file. [WWW document]. URL http://www.mathworks.com/matlabcentral/
%      fileexchange/loadFile.do?objectId=5578
%
%    References:
%    Huck, S. W. (2000), Reading Statistics and Research. 3rd. ed. 
%             New-York:Allyn&Bacon/Longman Pub. Chapter 16.
%

if nargin < 2,
   alpha = 0.05; %(default)
end; 

if (alpha <= 0 | alpha >= 1)
   fprintf('Warning: significance level must be between 0 and 1\n');
   return;
end;

if nargin < 1, 
   error('Requires at least one input argument.');
   return;
end;

a = max(X(:,2));
b = max(X(:,3));
s = max(X(:,4));

fprintf('The number of IV1 levels are:%2i\n\n', a);
fprintf('The number of IV2 levels are:%2i\n\n', b);
fprintf('The number of subjects are:%2i\n\n', s);

indice = X(:,2);
for i = 1:a
    Xe = find(indice==i);
    eval(['A' num2str(i) '=X(Xe,1);']);
end;

indice = X(:,3);
for j = 1:b
    Xe = find(indice==j);
    eval(['B' num2str(j) '=X(Xe,1);']);
end;

indice = X(:,4);
for k = 1:s
    Xe = find(indice==k);
    eval(['S' num2str(k) '=X(Xe,1);']);
end;

C = (sum(X(:,1)))^2/length(X(:,1));  %correction term
SSTO = sum(X(:,1).^2)-C;  %total sum of squares
dfTO = length(X(:,1))-1;  %total degrees of freedom
   
%procedure related to the IV1 (independent variable 1).
A = [];
for i = 1:a
    eval(['x =((sum(A' num2str(i) ').^2)/length(A' num2str(i) '));']);
    A = [A,x];
end;
SSA = sum(A)-C;  %sum of squares for the IV1
dfA = a-1;  %degrees of freedom for the IV1
MSA = SSA/dfA;  %mean square for the IV1

%procedure related to the IV2 (independent variable 2).
B = [];
for j = 1:b
    eval(['x =((sum(B' num2str(j) ').^2)/length(B' num2str(j) '));']);
    B =[B,x];
end;
SSB = sum(B)-C;  %sum of squares for the IV2
dfB = b-1;  %degrees of freedom for the IV2
MSB = SSB/dfB;  %mean square for the IV2

%procedure related to the within-subjects.
S = [];
for k = 1:s
    eval(['x =((sum(S' num2str(k) ').^2)/length(S' num2str(k) '));']);
    S = [S,x];
end;
SSS = sum(S)-C;  %sum of squares for the within-subjects
dfS = k-1;  %degrees of freedom for the within-subjects
MSS = SSS/dfS;  %mean square for the within-subjects

%procedure related to the IV1-error.
for i = 1:a
    for k = 1:s
        Xe = find((X(:,2)==i) & (X(:,4)==k));
        eval(['IV1S' num2str(i) num2str(k) '=X(Xe,1);']);
    end;
end;
EIV1 = [];
for i = 1:a
    for k = 1:s
        eval(['x =((sum(IV1S' num2str(i) num2str(k) ').^2)/length(IV1S' num2str(i) num2str(k) '));']);
        EIV1 = [EIV1,x];
    end;
end;
SSEA = sum(EIV1)-sum(A)-sum(S)+C;  %sum of squares of the IV1-error
dfEA = dfA*dfS;  %degrees of freedom of the IV1-error
MSEA = SSEA/dfEA;  %mean square for the IV1-error


%procedure related to the IV2-error.
for j = 1:b
    for k = 1:s
        Xe = find((X(:,3)==j) & (X(:,4)==k));
        eval(['IV2S' num2str(j) num2str(k) '=X(Xe,1);']);
    end;
end;
EIV2 = [];
for j = 1:b
    for k = 1:s
        eval(['x =((sum(IV2S' num2str(j) num2str(k) ').^2)/length(IV2S' num2str(j) num2str(k) '));']);
        EIV2 = [EIV2,x];
    end;
end;
SSEB = sum(EIV2)-sum(B)-sum(S)+C;  %sum of squares of the IV2-error
dfEB = dfB*dfS;  %degrees of freedom of the IV2-error
MSEB = SSEB/dfEB;  %mean square for the IV2-error

%procedure related to the IV1 and IV2.
for i = 1:a
    for j = 1:b
        Xe = find((X(:,2)==i) & (X(:,3)==j));
        eval(['AB' num2str(i) num2str(j) '=X(Xe,1);']);
    end;
end;
AB = [];
for i = 1:a
    for j = 1:b
        eval(['x =((sum(AB' num2str(i) num2str(j) ').^2)/length(AB' num2str(i) num2str(j) '));']);
        AB = [AB,x];
    end;
end;
SSAB = sum(AB)-sum(A)-sum(B)+C;  %sum of squares of the IV1xIV2
dfAB = dfA*dfB;  %degrees of freedom of the IV1xIV2
MSAB = SSAB/dfAB;  %mean square for the IV1xIV2

%procedure related to the IV1xIV2-error.
SSEAB = SSTO-SSA-SSEA-SSB-SSEB-SSAB-SSS;  %sum of squares of the IV1xIV2-error
dfEAB = dfTO-dfA-dfEA-dfB-dfEB-dfAB-dfS;  %degrees of freedom of the IV1xIV2-error
MSEAB = SSEAB/dfEAB;  %mean square for the IV1xIV2-error

%procedure related to the within-subject error.
SSSE = SSEA+SSEB+SSEAB;
dfSE = dfEA+dfEB+dfEAB;
MSSE = SSSE/dfSE;

%F-statistics calculation
F1 = MSA/MSEA;
F2 = MSB/MSEB;
F3 = MSAB/MSEAB;
F4 = MSS/MSSE;

%degrees of freedom re-definition
v1 = dfA;
v2 = dfEA;
v3 = dfB;
v4 = dfEB;
v5 = dfAB;
v6 = dfEAB;
v7 = dfS;
v8 = dfSE;
v9 = dfTO;

%Probability associated to the F-statistics.
P1 = 1 - fcdf(F1,v1,v2);    
P2 = 1 - fcdf(F2,v3,v4);   
P3 = 1 - fcdf(F3,v5,v6);
P4 = 1 - fcdf(F4,v7,v8);

eta21 = SSA/(SSA+SSEA)*100;
eta22 = SSB/(SSB+SSEB)*100;

disp('Repeated Measures Two-Way Analysis of Variance Table.')
fprintf('---------------------------------------------------------------------------\n');
disp('SOV                  SS          df           MS             F        P')
fprintf('---------------------------------------------------------------------------\n');
fprintf('Subjects      %11.3f%10i%15.3f[%13.3f%9.4f]\n\n',SSS,v7,MSS,F4,P4);
fprintf('IV1           %11.3f%10i%15.3f%14.3f%9.4f\n\n',SSA,v1,MSA,F1,P1);
fprintf('Error(IV1)    %11.3f%10i%15.3f\n\n',SSEA,v2,MSEA);
fprintf('IV2           %11.3f%10i%15.3f%14.3f%9.4f\n\n',SSB,v3,MSB,F2,P2);
fprintf('Error(IV2)    %11.3f%10i%15.3f\n\n',SSEB,v4,MSEB);
fprintf('IV1xIV2       %11.3f%10i%15.3f%14.3f%9.4f\n\n',SSAB,v5,MSAB,F3,P3);
fprintf('Error(IV1xIV2)%11.3f%10i%15.3f\n\n',SSEAB,v6,MSEAB);
fprintf('[Error        %11.3f%10i%15.3f]\n\n',SSSE,v8,MSSE);
fprintf('Total         %11.3f%10i\n\n',SSTO,v9);
fprintf('---------------------------------------------------------------------------\n');

fprintf('If the P results are smaller than% 3.2f\n', alpha );
disp('the corresponding Ho''s tested result statistically significant. Otherwise, are not significative.');
disp('[Generally speaking, no Mean Square is computed for the variable "subjects" since it is assumed');
disp('that subjects differ from one another thus making a significance test of "subjects" superfluous.');
disp('However, for all the interested people we are given it anyway].');
disp('  ');
fprintf('The percentage of the variability in the DV associated with the IV1 (eta squared) is% 3.2f\n', eta21);
disp('(After the effects of individual differences have been removed).');
disp('  ');
fprintf('The percentage of the variability in the DV associated with the IV2 (eta squared) is% 3.2f\n', eta22);
disp('(After the effects of individual differences have been removed).');disp('  ');

return;