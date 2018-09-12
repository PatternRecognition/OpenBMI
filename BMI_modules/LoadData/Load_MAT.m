function [train, test] = Load_MAT(filename)
    dat = importdata(filename);
    fields = sort(fieldnames(dat));
    
    test = dat.(fields{1});
    train = dat.(fields{2});
end