#ifndef GLOBALS_H
#define GLOBALS_H
#include <string>
using namespace std;


/**
	enum of the types of amplifiers currently available
*/
enum AmpType {File, Gtec, BrainAmp};
enum BinaryDataFormat {INT_16, INT_32, FLOAT_, DOUBLE_};
enum FileFormat {BRAIN_VISION, STANDARD};

static const string BV_HDR_EXT = ".VHDR";
static const string BV_MRK_EXT = ".VMRK";
static const string BV_EEG_EXT = ".EEG";
static const string STD_EXT = ".EEG";
static const string STD_MRK_EXT = ".MRK";



#endif

