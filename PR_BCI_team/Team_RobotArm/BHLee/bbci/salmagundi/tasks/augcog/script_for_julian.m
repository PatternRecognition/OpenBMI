% Einfach dieses Skript ausführen, und entsprechend die Optionen für folgende Variablen variieren:

% proc_variance, d.h. matlab/bci/processing bitte neu auschecken!!!

fil = 6     % alternative: 8
feature = 'ar';    % alternative: bandPowerbyvariance
task = '*calc';
ival = [1000 30000]; % damit die Fenster nicht überlappen
freq = [4 20];
spatial = 'car';
channels = {'not','E*','M*','Fp*'}; 
param = {6};   % für bandPowerbyvariance {}




setup_augcog;

fv = feval(['get_augcog_' feature],augcog(fil).file,task,ival,freq,spatial,param{:},channels);


fv = proc_flaten(fv);



% In fv.x stehen nun die Daten als Spaltenvektoren drin.
% VIEL SPASS DAMIT!!!