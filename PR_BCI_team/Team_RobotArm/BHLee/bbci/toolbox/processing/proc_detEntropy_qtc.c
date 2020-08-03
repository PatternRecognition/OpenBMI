/*
 * We got the file from <name>. I have changed nothing at the logic of the 
 * code. I made some changes to get the code to standard c because it didn't
 * compiled in matlab(Visual C++).
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>

/*#include "proc_detEntropy_qtc.h"*/


#define MAX_STEPSAMPLE_INTERVAL 65536  /* The maximum interval (2^16) for step information sample */
 /* match lists links same codewords based on IDs. string list links all codewords */
typedef struct CODEWORDLIST 
{  
    unsigned long CodeWordID;
    struct CODEWORDLIST *NextCodeWordInMatchList;  
    struct CODEWORDLIST *PreviousCodeWordInMatchList;    
    struct CODEWORDLIST *NextCodeWordInStringList;    
    struct CODEWORDLIST *PreviousCodeWordInStringList;  
    
}CODEWORD; 

typedef struct MATCHLISTENTRY_DEF  /* entry list stores heads of all match lists. These are dummy heads, no real data is stored here */
{  
    unsigned long CodeWordID;
    struct CODEWORDLIST *NextCodeWordInMatchList;   
    struct CODEWORDLIST *PreviousCodeWordInMatchList;
    
}MATCHLISTENTRY;



#define EMPTY -1


#define FALSE 0
#define TRUE 1


#define MAXIT 100
#define EULER 0.5772156649
#define FPMIN 5.0e-308
#define EPS 5.0e-16

 int	Info = FALSE, Entropy = FALSE, Comp = FALSE;
 int	Binary = FALSE;
 int	Normalise = FALSE;

  /* Command Line Flags */
  int FromFile = FALSE;
  char *FileName = NULL;
  

  /* Data Array */
	long DataLen;
	char *Dynamics;

	long Head, Tail;
	long *Next, *Last;
	long AugCount = 0;
	long Prefix = EMPTY, K = 0, LastK;
	double Complexity, LastComplexity = 0.0;


    CODEWORD *PrimeListEnd;
    MATCHLISTENTRY *EntryList;
	long HashTableSize;


    long Tprefix_N_Indicate;  
    long NextAvailableEntry; 
    long FirstNewEntry_PerPass;  
    CODEWORD *PrimeList;
    
    float V[7]={5.5427e-01f, 2.0248e-01f, -2.1048e-01f, 7.9845e-02f, -1.5054e-02f, 1.4153e-03f, -5.2962e-05f};


  /* T-information */
  long OldPos = 0;

long FileSize(char *FileName) {
  struct stat stbuf;

  if (stat(FileName, &stbuf) == -1)
    return(-1);
  else
    return(stbuf.st_size);
}


double ei(double x) 
{
        void nrerror(char error_text[]);
        int k;
        double fact,prev,sum,term;
                
        if (x <= 0.0) nrerror("Bad argument in ei");
        if (x < FPMIN) return log(x)+EULER;
        if (x <= -log(EPS)) {
                sum=0.0;
                fact=1.0;
                for (k=1;k<=MAXIT;k++) {
                        fact *= x/k;
                        term=fact/k;
                        sum += term;
                        if (term < EPS*sum) break;
                }
                if (k > MAXIT) nrerror("series failed in ei");
                return sum+log(x)+EULER;
        } else {
                sum=0.0;
                term=1.0;
                for (k=1;k<=MAXIT;k++) {
                        prev=term;
                        term *= k/x;
                        if (term < EPS) break;
                        if (term < prev) sum += term;
                        else {
                                sum -= prev;
                                break;
                        }
                }
                return exp(x)*(1.0+sum)/x;
        }
}

void nrerror(char error_text[])
{
        printf("%s\n", error_text);
        exit(1);
}


double logint(double x) {
        return ei(log(x));
}       

double invlogint(double x) {
        double lb, ub, tmp1, tmp2, g1;
        lb = 1.0+EPS;   
        if (x < logint(lb)) {
                nrerror("argument too small for fast algorithm");
        }
        ub = 2.0;
        while (logint(ub) < x) {
                lb = ub;
                ub *= 2;
        }
        tmp1 = logint(lb);
        tmp2 = logint(ub);
        g1 = 1/log(lb);
        /* x is now between logint(lb) and logint(ub) */
        /* printf("lb:%g ub:%g tmp1:%g tmp2:%g g1:%g\n",lb,ub,tmp1,tmp2,g1); */
        while (tmp2-tmp1 > EPS) {       
                /* printf("lb:%g ub:%g tmp1:%g tmp2:%g g1:%g\n",lb,ub,tmp1,tmp2,g1);
                printf("Iteration\n"); */
                ub = (x - tmp1) * (ub - lb)/(tmp2 - tmp1) + lb;
                tmp2 = logint(ub);
                lb += (x - tmp1)/g1;
                tmp1 = logint(lb);
                g1 = 1/log(lb);
        }
        if (Binary) return (ub-.7582221)/log(2.0);
        else return (ub-.7582221);
}

/**********************************************************************************/
/* New quickcalc
/*
/**********************************************************************************/

double ftdSelAugment ( char *buffer, long bufflen)
{  

    long Loop, StepCount, HashCodeTemp, Length_SuffixString_ParsedIntoOneCodeword, StepInfoPrintCount;
	long TprefixLength, HashCode_Tprefix, Count_TprefixMatch, TempID, TempEntry;

	char *pc_TprefixAddress;
	CODEWORD *CodeWordBegin, *PrimeListTemp, *RealEndOfPrimeList, *Node_Tprefix, *HeadOfCurrentMatchList;
	CODEWORD *CodeWordInMatchList;
	
    Complexity = 0;
    Tprefix_N_Indicate = 256;												/* entry reserved for p^k_n */
    NextAvailableEntry = 257; 												/* points to next available entry */
    FirstNewEntry_PerPass = NextAvailableEntry;  							/* entry for first new codeword in each pass */
    EntryList[Tprefix_N_Indicate].CodeWordID = 0;

    PrimeListEnd = NULL; 


/********* Initialize head node *********/
    PrimeList[0].NextCodeWordInStringList = PrimeList + 1;   
    PrimeList[0].PreviousCodeWordInStringList = NULL;
    
/********* Initialize end node *********/    
    PrimeList[bufflen].NextCodeWordInStringList = NULL;
    PrimeList[bufflen].PreviousCodeWordInStringList = PrimeList + bufflen - 1; 
    
    for (Loop = 0; Loop < 256; Loop++)  									/* initialize first 256 entries (0--255) */
    {
        EntryList[Loop].PreviousCodeWordInMatchList = (CODEWORD*)(EntryList + Loop); 
        EntryList[Loop].CodeWordID = 0; 
    }
    
    for (Loop= 1; Loop < bufflen ; Loop++)   							/* create lists. Pointers in NextCodeWordInMatchList and NextCodeWordInStringList should have same values */
    {
        PrimeList[Loop].NextCodeWordInStringList = PrimeList + Loop + 1;
        PrimeList[Loop].PreviousCodeWordInStringList = PrimeList + Loop - 1;
    }
    PrimeListEnd = PrimeList + bufflen;

    StepCount = 1;   													/* includes literal character */

    PrimeListTemp = PrimeList;
    for (Loop= 0; Loop < bufflen - 1; Loop++)  
    {
        HashCodeTemp = (unsigned char)(buffer[Loop]);  
        PrimeListTemp->NextCodeWordInStringList->CodeWordID = HashCodeTemp;
        
																			/*********** Add to the list **********/
        PrimeListTemp->NextCodeWordInStringList->PreviousCodeWordInMatchList =EntryList[HashCodeTemp].PreviousCodeWordInMatchList;
        PrimeListTemp->NextCodeWordInStringList->NextCodeWordInMatchList = (CODEWORD*)&(EntryList[HashCodeTemp]);
        EntryList[HashCodeTemp].PreviousCodeWordInMatchList->NextCodeWordInMatchList = PrimeListTemp->NextCodeWordInStringList;
        EntryList[HashCodeTemp].PreviousCodeWordInMatchList = PrimeListTemp->NextCodeWordInStringList;
																			/*************************************/

        PrimeListTemp++;
    }
    PrimeListTemp = PrimeList;

    Length_SuffixString_ParsedIntoOneCodeword = 0;   					/* compute length of substring (suffix of string) that has been parsed as one codeword after each run */
    RealEndOfPrimeList = PrimeListEnd;

    StepInfoPrintCount = 0;   											/* To work with StepInfoDisplayInterval */

    				/*********** start T-decomposition ***********/

    for (;;)  
    {
        if (PrimeList->NextCodeWordInStringList->NextCodeWordInStringList == NULL) /* Check if process has finished (i.e. if these is only one codeword in string) */
            break;

        K = 1;   														/* count the number of the adjacent codewords that match the T-prerix. */
        StepCount++;

         			/*********** (1)Get the T-prefix ********** */
        Node_Tprefix = PrimeListEnd->PreviousCodeWordInStringList;
        pc_TprefixAddress = buffer + (Node_Tprefix - PrimeList) - 1;  
        TprefixLength = RealEndOfPrimeList - Node_Tprefix;
        HashCode_Tprefix = (Node_Tprefix->CodeWordID);
        HeadOfCurrentMatchList = Node_Tprefix->NextCodeWordInMatchList;  /*Point to the dummy head of the hash list.*/

        			/*********** (2)Find how many adjacent copies preceding the T-prefix at the end of the string. So we can got the T-expansion parameter ***********/
        while (Node_Tprefix->PreviousCodeWordInStringList->PreviousCodeWordInStringList != NULL)
        {
            if (Node_Tprefix->PreviousCodeWordInMatchList == Node_Tprefix->PreviousCodeWordInStringList)
            {  
                Node_Tprefix = Node_Tprefix->PreviousCodeWordInStringList;
                K++;
            }
            else
                break;
        }
        RealEndOfPrimeList = Node_Tprefix; 
  
        Node_Tprefix->PreviousCodeWordInMatchList->NextCodeWordInMatchList = HeadOfCurrentMatchList;
        HeadOfCurrentMatchList->PreviousCodeWordInMatchList = Node_Tprefix->PreviousCodeWordInMatchList;
        Node_Tprefix->PreviousCodeWordInStringList->NextCodeWordInStringList = PrimeListEnd;
        PrimeListEnd->PreviousCodeWordInStringList = Node_Tprefix->PreviousCodeWordInStringList;

        Complexity += log((double)(K + 1)); 
        
        			/*********** (3)Search from the beginning of the match list to find the codeword that matches the T-prefix ***********/
        CodeWordInMatchList = HeadOfCurrentMatchList;  /*The head of the match list.*/

        if (CodeWordInMatchList ->NextCodeWordInMatchList == HeadOfCurrentMatchList) 	/* if list does not exist */
          	continue;
        CodeWordInMatchList = CodeWordInMatchList->NextCodeWordInMatchList;
        Count_TprefixMatch = 0;  													/* The number of the adjacent copies of T-prefix found. It will be set to 0 again */
                                                                             			/* once the new codeword does not mathc the T-prefix */
        																				/* Now search really begins from begining of corresponding match list */
        																				/* The match list helps to locate the codeword in the string list */
        while (CodeWordInMatchList != HeadOfCurrentMatchList)  
        {
            PrimeListTemp = CodeWordInMatchList;
            Count_TprefixMatch = 1;  													/* number of adjacent  codewords matching the T-prefix */
            CodeWordBegin = PrimeListTemp; 									/* The node that pointing to the possible beginning of a new codeword */
            PrimeListTemp = PrimeListTemp->NextCodeWordInStringList;

            for (;;)  																	/* matched codeword found. Here we need remove some codewords from the match list and string list */
                        																/* New merged codeword will be added to the corresponding match list. Check in the string list */
            {    
                if (HashCode_Tprefix == PrimeListTemp->CodeWordID)
                { 																		/* The current codeword DOES match the T-prefix */
                    Count_TprefixMatch++;  
                    if (Count_TprefixMatch <= K)
                    {
                        PrimeListTemp = PrimeListTemp->NextCodeWordInStringList;
                    }
                    else  																/* number of the adjacent matching codewords exceeds the T-expansion parameter */
                    {
                        																/* move the pointer to the next code in the match list */
                        CodeWordInMatchList = PrimeListTemp->NextCodeWordInMatchList;
                        																/* delete the node in the string list */
                        CodeWordBegin->NextCodeWordInStringList = PrimeListTemp->NextCodeWordInStringList;
                        PrimeListTemp->NextCodeWordInStringList->PreviousCodeWordInStringList = CodeWordBegin;
                        if ( EntryList[Tprefix_N_Indicate].CodeWordID < FirstNewEntry_PerPass) /* no entry for this new codeword */
                        {
                            EntryList[Tprefix_N_Indicate].CodeWordID = NextAvailableEntry;
                            TempID = NextAvailableEntry;
                            EntryList[NextAvailableEntry].PreviousCodeWordInMatchList = (CODEWORD*)(EntryList + NextAvailableEntry);
                            EntryList[NextAvailableEntry].CodeWordID= 0;
                            NextAvailableEntry++;
                        }
                        else
                        {
                            TempID = EntryList[Tprefix_N_Indicate].CodeWordID;
                        }
                        CodeWordBegin->CodeWordID = TempID;

																						/*********** Add to the lists ***********/
                        CodeWordBegin->PreviousCodeWordInMatchList =EntryList[TempID].PreviousCodeWordInMatchList;
                        CodeWordBegin->NextCodeWordInMatchList = (CODEWORD*)(&(EntryList[TempID]));
                        EntryList[TempID].PreviousCodeWordInMatchList->NextCodeWordInMatchList = CodeWordBegin;
                        EntryList[TempID].PreviousCodeWordInMatchList = CodeWordBegin;

                        break;
                    }
                }
                else   																	/* current codeword does not match the T-prefix */
                {
                    TempEntry = PrimeListTemp->CodeWordID;
                    CodeWordInMatchList = PrimeListTemp->PreviousCodeWordInStringList->NextCodeWordInMatchList;
                    for (Loop = 0; Loop < Count_TprefixMatch; Loop++)  					/* Remove codewords from string list, the first matched codeword remains */
                    {
                        if (EntryList[TempEntry].CodeWordID < FirstNewEntry_PerPass) 	/* entry for the new codeword */
                        {
                            EntryList[TempEntry].CodeWordID = NextAvailableEntry;
                            TempEntry = NextAvailableEntry;
                            EntryList[TempEntry].PreviousCodeWordInMatchList = (CODEWORD*)(EntryList + TempEntry);
                            EntryList[TempEntry].CodeWordID= 0;
                                
                            NextAvailableEntry++;
                        }
                        else
                        {
                            TempEntry = EntryList[TempEntry].CodeWordID;
                        }
                    }
                    CodeWordBegin->CodeWordID = TempEntry;  							/* new ID */
                    { 																	/* if not the last node in the match list */
                        PrimeListTemp->PreviousCodeWordInMatchList->NextCodeWordInMatchList = PrimeListTemp->NextCodeWordInMatchList; 
                        PrimeListTemp->NextCodeWordInMatchList->PreviousCodeWordInMatchList = PrimeListTemp->PreviousCodeWordInMatchList;
                    }
                    
																						/*********** Add to the lists ***********/
                    CodeWordBegin->PreviousCodeWordInMatchList =EntryList[TempEntry].PreviousCodeWordInMatchList;
                    CodeWordBegin->NextCodeWordInMatchList = (CODEWORD*)(&(EntryList[TempEntry]));
                    EntryList[TempEntry].PreviousCodeWordInMatchList->NextCodeWordInMatchList = CodeWordBegin;
                    EntryList[TempEntry].PreviousCodeWordInMatchList = CodeWordBegin;
																						/****************************************/
                    CodeWordBegin->NextCodeWordInStringList = PrimeListTemp->NextCodeWordInStringList;
                    PrimeListTemp->NextCodeWordInStringList->PreviousCodeWordInStringList = CodeWordBegin;
                
                    break;
                }
            }																			/* end of for (;;) */
        }
        FirstNewEntry_PerPass = NextAvailableEntry;
    }


    return Complexity/log(2.0);
}








/******************* main *********************/
int main(int argc, char *argv[]) {

  int Loop, DataOut=FALSE ; 
  long	i, offset;
  long	Window = 0, Shift;
  float Scale=1.0;
  float xscale = 9.0, yscale = 1.0 ,zscale = 0.0 , xoffset = 0.0, yoffset = 0.0 ,zoffset = 0.0;
  FILE *f;

  /* Read arguments from command line */
  for (Loop = 1; Loop < argc; Loop++) {
		if ((strcmp(argv[Loop], "?")== 0) || (strcmp(argv[Loop], "-h")== 0)) {
			 printf("\t qtc computes T-complexity (taugs)  T-information (nats or bits) and T-entropy (nats/char or bits/char)\n");
			printf("\t using the T-decomposition algorithm by M.R.Titchener\n");
			printf("\t\tQuick-tcalc (qtc)  incorporates a modified version of the implementation by Speidel and Yang \n");
			printf("\t\t\t format: qtc -f filename\n");
			printf("\t\t\t T-Information only: qtc -f filename -i\n");
			printf("\t\t\t Entropy only: qtc -f filename -i\n");
			printf("\t\t\t nats -> bits : qtc -f filename -b\n");
			printf("\t\t\t -f filename \t\n");
			printf("\t\t\t -i \t outputs T-information only\n");
			printf("\t\t\t -e \t outputs T-entropy only\n");
			printf("\t\t\t -b \t information units (bits) instead of (nats)\n");
			printf("\t\t\t -ud \t deci-nats/bits\n");
			printf("\t\t\t -uc \t centi-nats/bits\n");
			printf("\t\t\t -um \t milli-nats/bits\n");
			printf("\t\t\t -w W S \t assume a window W and step S\n");
			printf("\t\t\t -h \t prints this info\n");
			printf("\t \n");
			return 0;
		}
		else if ((strcmp(argv[Loop], "-f") == 0) || (strcmp(argv[Loop], "-F")== 0 )){
			  FromFile = TRUE;     
			  if (argc > (Loop+1)) {      
				FileName = argv[Loop + 1];
				Loop++;
			  }
			  else
			  {
				printf("No filenname specified!\n");
				return -1;
			  }
		}
		else if (strcmp(argv[Loop], "-c") == 0)  { /*utput T-information */
		  Comp = TRUE; Info = Entropy = FALSE;
		}
		else if (strcmp(argv[Loop], "-i") == 0)  { /*output T-information */
		  Info = TRUE; Entropy = Comp=FALSE; 
		}
		else if (strcmp(argv[Loop], "-e") == 0)  { /*output T-information */
		  Entropy = TRUE;
		  Comp = Info = FALSE;
		}
		else if (strcmp(argv[Loop], "-b") == 0)  {
		  Binary = TRUE;
		}
		else if (strcmp(argv[Loop], "-n") == 0)  { /* normalise entropy */
		  Normalise = TRUE;
		}
		else if (strcmp(argv[Loop], "-ud") == 0)  { /*deci-range */
		  Scale=10.0;
		}
		else if (strcmp(argv[Loop], "-uc") == 0)  { /*centi-range */
		  Scale=100.0;
		}
		else if (strcmp(argv[Loop], "-um") == 0)  { /*milli-range */
		  Scale=1000.0;
		}
		else if (strcmp(argv[Loop], "-w") == 0) {  
		  sscanf(argv[Loop + 1], "%d", &Window);
		  Loop++;
		  sscanf(argv[Loop + 1], "%d", &Shift);
		  Loop++;
		}
		else if (strcmp(argv[Loop], "-xyz") == 0) {   
		  sscanf(argv[Loop + 1], "%f", &xscale);
		  Loop++;
		  sscanf(argv[Loop + 1], "%f", &yscale);
		  Loop++;
		  sscanf(argv[Loop + 1], "%f", &zscale);
		  Loop++;
		  sscanf(argv[Loop + 1], "%f", &xoffset);
		  Loop++;
		  sscanf(argv[Loop + 1], "%f", &yoffset);
		  Loop++;
		  sscanf(argv[Loop + 1], "%f", &zoffset);
		  Loop++;
		}
    }
    
 	if (FromFile == TRUE) { 
	  	struct stat stbuf;

	  	if (stat(FileName, &stbuf) == -1) return(-1);
		else {
      float loglength, reference;
      float Information;
			DataLen = stbuf.st_size;
			if (Window == 0) Window = Shift = DataLen;
			else if (Shift > Window) Shift = Window;


			
			
			if (Normalise == TRUE) {loglength = (float)(log(Window)/log(10)); reference = V[0]+(V[1] +(V[2]+(V[3]+(V[4]+ (V[5] +V[6]*loglength)*loglength)*loglength)*loglength)*loglength)*loglength; 
				if (Binary) reference=reference/(float)log(2.0);
			}
			else reference = 1.0;
			

			Dynamics = (char *)malloc(DataLen);
			f = fopen(FileName, "r");
			fread(Dynamics, 1, DataLen, f);
			fclose(f);
			
		
			HashTableSize = DataLen + 255;
			EntryList = (MATCHLISTENTRY*) malloc(sizeof (MATCHLISTENTRY) * HashTableSize);
			PrimeList = (CODEWORD *)malloc ((DataLen + 2) * sizeof (CODEWORD)); /* two extra nodes, head, end nodes in string list. */
				
			if ((EntryList == NULL) || (PrimeList == NULL) || (Dynamics == NULL) ) {printf("Error Allocating Memory for linked list\n"); return -1;}
			
			

			if ((DataLen-Window)/Shift > 1.0) {
				i = offset = 0;
				while (offset <= DataLen-Window) {offset+=Shift; i++;}
				printf ("Y:g:graph\n%d 1\n%f %f %f\n%f %f %f\n", i ,xscale, yscale ,zscale , xoffset, yoffset, zoffset);
			}

			offset = 0; 			
			while (offset <= DataLen-Window) {
				Complexity = ftdSelAugment(&Dynamics[offset], Window);

				if ((Comp == FALSE) && (Info == FALSE) && (Entropy == FALSE) ) printf("%.2f %.2f %.5f\n", Complexity, Scale*Information, Scale*(Information=invlogint(Complexity))/Window/reference); 
				else if (Comp == TRUE) printf("%.2f ", Complexity );
				else if (Info == TRUE) printf("%.2f ", Scale*invlogint(Complexity) );
				else if (Entropy == TRUE) printf("%.5f ", Scale*invlogint(Complexity)/Window/reference );
				offset+=Shift; 
			}
			printf ("\n");

			if ((DataLen-Window)/Shift > 1.0) {
				printf ("G 0.700000 0.000000 0.900000  9.000000 6.000000 6.000000  0.000000 0.000000 0.000000  1.000000 1.000000 0.000000\n");
				printf ("I -1.0 0.000010 %f 0.0 -1.0\n", 500.0/(DataLen/Shift));
			}
			free (Dynamics);
			free (PrimeList);
			free (EntryList);
		}
	}

    return 0;
}

int initQtc(int dataSetSize) {
/* setup for qtc (for reasons, ask Mark :-) */
  HashTableSize = dataSetSize + 255;
  EntryList = (MATCHLISTENTRY*) malloc(sizeof (MATCHLISTENTRY) * HashTableSize);
  PrimeList = (CODEWORD *)malloc ((dataSetSize + 2) * sizeof (CODEWORD)); /* two extra nodes, head, end nodes in string list. */
  
  return EntryList != NULL && PrimeList != NULL;
}

void releaseQtc() {
  if(PrimeList != 0) {
    free (PrimeList);PrimeList = 0;
  }
  if(EntryList != 0) {
    free (EntryList);EntryList = 0;
  }
}
  


