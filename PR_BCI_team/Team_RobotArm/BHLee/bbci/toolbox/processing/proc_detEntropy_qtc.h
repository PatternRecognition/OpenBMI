 #ifndef PROC_DET_ENTROPY_QTC_H
 #define PROC_DET_ENTROPY_QTC_H
 
 void initQtc(int size);
 void releaseQtc();
 
 double invlogint(double x);
 double ftdSelAugment ( char *buffer, long bufflen);
 
 #include "proc_detEntropy_qtc.c"
 
 #endif /* PROC_DET_ENTROPY_QTC_H */
