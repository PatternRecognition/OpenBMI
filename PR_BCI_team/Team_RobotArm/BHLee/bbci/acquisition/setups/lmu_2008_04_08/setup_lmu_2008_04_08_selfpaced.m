opt= struct('maxtime', 1*15);
opt.bv_host= 'localhost';
opt.validKeys= 'jklö';

fprintf('for testing:\n  stim_selfpaced(opt, ''test'',1);\n');

fprintf('stim_selfpaced(opt);\n');
