function bool= iscluster

[res, hn]= system('hostname');
bool= strncmp(hn, 'node', 4);
