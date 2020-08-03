function [name, exp_type, exp_name]= expbase_filename(exp)
%[filename, exptype, expname]= expbase_filename(exp)
%
% IN  exp  - struct with experiment infos, as each element of the
%            struct array which is obtained from generate_experiment_database.
%      .subject  - name of the subject
%      .date     - date as string yy_mm_dd
%      .paradigm - base name of the experiment, e.g., 'imag'
%      .appendix - appendix to the expriment name, e.g., 'fb2'

%% bb ida.first.fhg.de 07/2004

name= strcat(exp.subject, '_', exp.date, '/', exp.paradigm, ...
             exp.appendix, exp.subject);

if nargout>1,
  if iscell(exp.paradigm),
    exp_type= vec2str(exp.paradigm, '%s', '_');
  else
    exp_type= exp.paradigm;
  end
  exp_name= strcat(exp.subject, '_', exp.date, '_', exp_type);
end
