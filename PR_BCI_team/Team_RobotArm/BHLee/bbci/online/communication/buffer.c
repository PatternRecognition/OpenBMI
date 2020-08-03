/*
 * buffer.c
 *
 * The buffer is for the data we receive from the gtec controllers.
 * For each controller we store the data in a different track.
 *
 * The buffer has the following organisation to make it easier to use the
 * buffer with filter.h and the gtec controllers.
 *
 * trackSize = 16
 *
 *                1. track                2. track           ...
 *        | 1  2  3  4  .  .  . 16| 1  2  3  4  .  .  . 16| ...    Data in track
 * -----------------------------------------------------------------------------
 * 1.line |00 02 04 08  .  .  . 30|32 34 36 38  .  .  . 62| ...  byte number in the buffer
 * 2.line |
 * 3.line |
 *   .
 *   .
 *   .
 *
 * 2009/01/16 - Max Sagebaum
 *              - file created
 */

#include "buffer.h"

static double* buffer;       /* The buffer for the data */
static int    readPos;      /* The position where the new data starts in the buffer */
static int*   writePos;     /* The writing position for each track in the buffer */

static int numberOfTracks;  /* The number of tracks in the buffer */
static int trackSize;       /* The size of each track */
static int bufferLines;     /* The number of lines in the buffer */
static int lineSize;      /* The size of each line in the buffer */

/*********************************************
 *
 * Creates the arrays for the buffer. Sets the reading and writing
 * positions.
 * INPUT: lines         - The number of lines for the buffer
 *        tracks        - The number of tracks for the buffer
 *        sizePerTrack  - The number of values for each track
 * OUTPUT: -
 * RETURN: 1 if the buffer was created, false otherwise
 *
 **********************************************/
static int bufferCreate(int lines, int tracks, int sizePerTrack) {
  int i;
  
  /* set the size values for the buffer */
  numberOfTracks = tracks;
  bufferLines = lines;
  trackSize = sizePerTrack;
  lineSize = trackSize * numberOfTracks;
  
  /* init the track positions */
  readPos = 0;
  writePos = (int*)malloc(numberOfTracks * sizeof(int));
  for(i = 0; i < numberOfTracks; ++i) {
    writePos[i] = 0;
  }
  
  /* create the buffer and check if enough space was left. */
  buffer = (double*)malloc(bufferLines * numberOfTracks * trackSize * sizeof(double));  
  if(NULL == buffer) {
    return 0;
  } else {
    return 1;
  }
}

/*********************************************
 *
 * Writes the values in data into the given track.
 * INPUT: data          - The values which will be written to the track.
 *        dataLines     - The number of lines in data.
 *        writeTrack    - Specifies the track for the data.
 * OUTPUT: -
 *
 **********************************************/
static void bufferWrite(float *data, int dataLines, int writeTrack) {
  int trackPos;
  int linePos;
  
  int dataLineStart;
  
  /* get the offset for the write position in the buffer */
  dataLineStart = writePos[writeTrack];
  
  for(linePos = 0; linePos < dataLines; ++linePos) {
    /* get the position for the line in the buffer, we have a torodical access */
    int linePosInBuffer = (dataLineStart + linePos) % bufferLines;
    linePosInBuffer = linePosInBuffer * lineSize;
    
    for(trackPos = 0; trackPos < trackSize; ++trackPos) {
      int dataPos;
      int bufferPos;
      
      /* get the position in the data and the position in the current line
       * of the buffer */
      dataPos = trackPos + linePos * trackSize;
      bufferPos = linePosInBuffer + writeTrack * trackSize + trackPos; 

      buffer[bufferPos] = (double)data[dataPos];
    }
  }
  
  /* set the new position for the current track */
  writePos[writeTrack] = (dataLineStart + dataLines) % bufferLines;
}

/*********************************************
 *
 * Gets the position in the buffer were the new data begins and the size of
 * new data.
 * INPUT: -
 * OUTPUT: dataStart    - The position where the new data begins
 *         dataSize     - The size of the new data
 *
 **********************************************/
static void bufferRead(double** dataStart, int* dataSize) {
  int curTrack;
  int maxBufferEndPos;
  
  /* The maximum size for the size of the data */
  maxBufferEndPos = bufferLines;
  
  for(curTrack = 0; curTrack < numberOfTracks; ++curTrack) {
    int newMaxBufferEndPos;
    if(writePos[curTrack] < readPos) {
      /* The write pos is less than the read pos, the ring buffer has
       * crossed the end of the buffer while writing. This means that the
       * data to the end of the buffer is new.
       */
      newMaxBufferEndPos = bufferLines;
    } else {
      newMaxBufferEndPos = writePos[curTrack];
    }
    
    if(maxBufferEndPos > newMaxBufferEndPos) {
      maxBufferEndPos = newMaxBufferEndPos;
    }
  }
  
  /* set the ouput variables */
  *dataStart = buffer + readPos * lineSize; /* the pointer to the start of the data */
  *dataSize = maxBufferEndPos - readPos; /* set the size of the new data */
  
  /* Set the new read position for the next call */
  readPos = maxBufferEndPos % bufferLines;  
}

/*********************************************
 *
 * Deletes all arrays which the buffer uses.
 *
 **********************************************/
static void bufferClose() {
  if(NULL != buffer) { free(buffer); buffer = NULL;}
  if(NULL != writePos) { free(writePos); writePos = NULL;}
}